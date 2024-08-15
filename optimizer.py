import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import optuna

from pvlib import pvsystem, location, modelchain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS as PARAMS
from pvlib.bifacial.pvfactors import pvfactors_timeseries

from Albedo_TimeSeries.monthly_albedo import calculate_monthly_avg_albedo
from SnowHeight.monthly_snow import calculate_new_row_height
from pvtune import process_pvtune_output

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pvlib')
warnings.filterwarnings(action='ignore', module='pvfactors')

def optimize_axistilt(
        pvtune_filepath, albedodata_filepath, lat, lon, tz, gcr, max_angle, pvrow_width,
        bifaciality, temp_model_parameters, cec_modules, cec_module, cec_inverters, cec_inverter, site_name):
    """
    Optimize the axis tilt angle of solar panels on a solar farm to maximize total energy output.
    
    Parameters:
    - pvtune_filepath: File path to the pvtune output CSV.
    - albedodata_filepath: File path to the albedo time series data.
    - lat, lon, tz: Latitude, longitude, and timezone of the site.
    - gcr: Ground coverage ratio.
    - max_angle: Maximum allowable tilt angle for solar panels.
    - pvrow_width: Width of the PV rows.
    - bifaciality: Bifaciality factor of the solar panels.
    - temp_model_parameters: Temperature model parameters.
    - cec_modules, cec_module: Module data for the system.
    - cec_inverters, cec_inverter: Inverter data for the system.
    - site_name: Name of the site.
    
    Returns:
    - optimal_axistilt: The optimal axis tilt angle that maximizes energy output.
    - maximized_total_energy: The maximum total energy output corresponding to the optimal tilt.
    - summary_df: DataFrame summarizing the PV tune output.
    
    This function uses Optuna to optimize the tilt angle of the solar panels to maximize energy
    output, considering factors like albedo, snow height, and bifacial gain.
    """

    # Suppress Optuna logging output
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # Load the albedo data
    albedo_data = calculate_monthly_avg_albedo(albedodata_filepath)
    
    # Function to get the albedo value for a specific date
    def get_albedo(date):
        month = date.month
        albedo_row = albedo_data[albedo_data['Month'] == month]
        if not albedo_row.empty:
            return albedo_row['Surface Albedo'].values[0]
        else:
            return 0.2  # Default albedo value if no data is available
    
    # Create a location object for the solar farm site
    site_location = location.Location(lat, lon, tz=tz, name=site_name)
    
    # Load the row height data from the CSV and calculate the weighted average reveal height
    summary_df, weighted_avg_reveal_height = process_pvtune_output(pvtune_filepath)

    # Function to calculate the total energy output for a given axis tilt
    def calculate_total_energy(axis_tilt):
        # Adjust row heights based on snow height for each month
        adjusted_heights_df = calculate_new_row_height(lat, lon, weighted_avg_reveal_height)
        
        # Function to get the adjusted row height for a specific date
        def get_adjusted_row_height(date):
            month = date.month
            height_row = adjusted_heights_df[adjusted_heights_df['month'] == month]
            if not height_row.empty:
                return height_row['adjusted_row_height'].values[0]
            else:
                return weighted_avg_reveal_height  # Default to weighted average reveal height

        total_energy_bi = 0  # Initialize the total energy for the year
        start_date = pd.Timestamp('2021-01-01')  # Start date for the simulation
        end_date = pd.Timestamp('2021-12-31')  # End date for the simulation
        count = 0  # Initialize count to track the number of days simulated

        # Loop through the entire year, simulating every 10 days
        while start_date <= end_date:
            # Generate a time series for the current day
            times = pd.date_range(start_date, start_date + pd.Timedelta(days=1), freq='20min', tz=tz)
            albedo_values = times.to_series().apply(get_albedo)  # Get the albedo values for the day
            pvrow_height = get_adjusted_row_height(start_date)  # Get the adjusted row height for the day
            solar_position = site_location.get_solarposition(times)  # Get the solar position data
            cs = site_location.get_clearsky(times)  # Get the clear sky irradiance

            # Define the tracker mount with the current axis tilt
            sat_mount = pvsystem.SingleAxisTrackerMount(axis_tilt=axis_tilt,
                                                        axis_azimuth=180,  # South-facing panels
                                                        max_angle=max_angle,
                                                        backtrack=True,
                                                        gcr=gcr)
            # Get the orientation of the solar panels
            orientation = sat_mount.get_orientation(solar_position['apparent_zenith'],
                                                    solar_position['azimuth'])

            # Calculate irradiance using the pvfactors timeseries model
            irrad = pvfactors_timeseries(solar_position['azimuth'],
                                         solar_position['apparent_zenith'],
                                         orientation['surface_azimuth'],
                                         orientation['surface_tilt'],
                                         180,  # Fixed axis azimuth
                                         times,
                                         cs['dni'],
                                         cs['dhi'],
                                         gcr,
                                         pvrow_height,
                                         pvrow_width,
                                         albedo_values,
                                         n_pvrows=3,  # Number of PV rows
                                         index_observed_pvrow=1)  # Index of observed row

            irrad = pd.concat(irrad, axis=1)  # Combine irradiance data
            # Calculate effective irradiance considering bifaciality
            irrad['effective_irradiance'] = (
                irrad['total_abs_front'] + (irrad['total_abs_back'] * bifaciality)
            )

            # Create the PV array and system using the calculated irradiance
            array = pvsystem.Array(mount=sat_mount,
                                   module_parameters=cec_module,
                                   temperature_model_parameters=temp_model_parameters)
            system = pvsystem.PVSystem(arrays=[array],
                                       inverter_parameters=cec_inverter)
            # Create the model chain and run the simulation
            mc_bifi = modelchain.ModelChain(system, site_location, aoi_model='no_loss')
            mc_bifi.run_model_from_effective_irradiance(irrad)

            ac_power = mc_bifi.results.ac  # Get the AC power output
            energy_kwh = ac_power.sum() / 3000  # Convert power from watts to kWh
            total_energy_bi += energy_kwh  # Accumulate the total energy output

            # Move to the next 10-day period
            start_date += pd.Timedelta(days=10)
            count += 1

        # Scale the total energy to represent the full year
        return total_energy_bi * (365 / count)

    # Initialize lists to store the trial results
    trial_axis_tilts = []
    trial_energies = []

    # Define the objective function for the Optuna optimization
    def objective(trial):
        # Suggest a value for axis tilt within the range 0 to 60 degrees
        axis_tilt = trial.suggest_float('axis_tilt', 0, 60)
        # Calculate the total energy for the suggested axis tilt
        total_energy = calculate_total_energy(axis_tilt)
        
        # Save the trial results for plotting
        trial_axis_tilts.append(axis_tilt)
        trial_energies.append(total_energy)
        
        return total_energy  # Maximize energy output

    # Create an Optuna study and perform the optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)  # Run the optimization for 20 trials

    # Plot the results of the optimization
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(trial_axis_tilts, trial_energies, c=trial_energies, cmap='viridis', s=50)

    ax.set_xlabel('Axis Tilt (degrees)')
    ax.set_ylabel('Total Energy (kWh)')
    ax.set_title('Total Energy vs Axis Tilt')

    # Add a color bar for energy output
    plt.colorbar(sc, ax=ax, label='Total Energy (kWh)')

    plt.show()

    # Get the best result from the optimization
    optimal_axistilt = study.best_params['axis_tilt']
    maximized_total_energy = study.best_value

    return optimal_axistilt, maximized_total_energy, summary_df
