import pandas as pd
import numpy as np

from pvlib import pvsystem, location, modelchain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS as PARAMS
from pvlib.bifacial.pvfactors import pvfactors_timeseries

from Albedo_TimeSeries.monthly_albedo import calculate_monthly_avg_albedo
from SnowHeight.monthly_snow import calculate_new_row_height

def calculate_total_energy(
        height, ext_or_int, opt_tilt, albedodata_filepath, lat, lon, tz, gcr, 
        max_angle, pvrow_width, bifaciality, temp_model_parameters, cec_modules, cec_module,
        cec_inverters, cec_inverter, site_name):
    """
    Calculate the total energy output of a solar farm over a year, considering different parameters
    such as albedo, snow height, and bifacial gain.

    Parameters:
    - height: Original reveal height of the PV rows.
    - ext_or_int: Specifies whether the tracker row is external ("Ext") or internal ("Int").
    - opt_tilt: Optimal axis tilt angle.
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
    - total_energy_bi: Total energy output of the bifacial solar system in kWh over the year.
    
    This function calculates the total energy output for a bifacial solar system, adjusting
    for seasonal factors such as albedo and snow height. The function handles both external and
    internal tracker rows and simulates energy production over a full year.
    """
    
    # Set the number of PV rows based on ext_or_int input
    if ext_or_int == "Ext":
        n_pvrows = 2
    elif ext_or_int == "Int":
        n_pvrows = 3
    else:
        raise ValueError("Invalid value for ext_or_int. Must be 'Ext' or 'Int'.")

    # Load the albedo data
    albedo_data = calculate_monthly_avg_albedo(albedodata_filepath)
    
    def get_albedo(date):
        # Get the albedo value based on the month
        month = date.month
        albedo_row = albedo_data[albedo_data['Month'] == month]
        if not albedo_row.empty:
            return albedo_row['Surface Albedo'].values[0]
        else:
            return 0.2  # Default albedo value if no data is available
    
    # Create a location object for the site
    site_location = location.Location(lat, lon, tz=tz, name=site_name)
    
    # Calculate adjusted row heights based on snow conditions
    adjusted_heights_df = calculate_new_row_height(lat, lon, height)
        
    def get_adjusted_row_height(date):
        # Get the adjusted row height based on the month and snow data
        month = date.month
        height_row = adjusted_heights_df[adjusted_heights_df['month'] == month]
        if not height_row.empty:
            return height_row['adjusted_row_height'].values[0]
        else:
            return height  # Default to original reveal height if no adjustment is needed
    
    total_energy_bi = 0  # Initialize total energy accumulator
    start_date = pd.Timestamp('2021-01-01')  # Start date of the simulation
    end_date = pd.Timestamp('2021-12-31')  # End date of the simulation
    count = 0  # Initialize the counter for iterations

    # Loop through the entire year with 10-day increments
    while start_date <= end_date:
        times = pd.date_range(start_date, start_date + pd.Timedelta(days=1), freq='20min', tz=tz)
        albedo_values = times.to_series().apply(get_albedo)  # Get albedo values for the current time range
        pvrow_height = get_adjusted_row_height(start_date)  # Adjust the row height for the current date
        solar_position = site_location.get_solarposition(times)  # Get the solar position for the current time range
        cs = site_location.get_clearsky(times)  # Get the clear sky irradiance data
        
        # Define the single-axis tracker mount configuration
        sat_mount = pvsystem.SingleAxisTrackerMount(axis_tilt=opt_tilt,
                                                    axis_azimuth=180,
                                                    max_angle=max_angle,
                                                    backtrack=True,
                                                    gcr=gcr)
        # Calculate the panel orientation based on solar position and mount configuration
        orientation = sat_mount.get_orientation(solar_position['apparent_zenith'],
                                                solar_position['azimuth'])

        # Calculate the irradiance using the pvfactors model
        irrad = pvfactors_timeseries(solar_position['azimuth'],
                                     solar_position['apparent_zenith'],
                                     orientation['surface_azimuth'],
                                     orientation['surface_tilt'],
                                     180,  # Fixed axis_azimuth
                                     times,
                                     cs['dni'],
                                     cs['dhi'],
                                     gcr,
                                     pvrow_height,
                                     pvrow_width,
                                     albedo_values,
                                     n_pvrows=n_pvrows,
                                     index_observed_pvrow=1)

        irrad = pd.concat(irrad, axis=1)
        irrad['effective_irradiance'] = (
            irrad['total_abs_front'] + (irrad['total_abs_back'] * bifaciality)
        )

        # Define the PV array and system configuration
        array = pvsystem.Array(mount=sat_mount,
                               module_parameters=cec_module,
                               temperature_model_parameters=temp_model_parameters)
        system = pvsystem.PVSystem(arrays=[array],
                                   inverter_parameters=cec_inverter)
        
        # Run the ModelChain simulation
        mc_bifi = modelchain.ModelChain(system, site_location, aoi_model='no_loss')
        mc_bifi.run_model_from_effective_irradiance(irrad)

        ac_power = mc_bifi.results.ac  # Get the AC power output
        energy_kwh = ac_power.sum() / 3000  # Convert from watts to kWh
        total_energy_bi += energy_kwh  # Accumulate total energy

        start_date += pd.Timedelta(days=10)  # Move to the next 10-day period
        count += 1  # Increment the counter

    # Calculate the average total energy over the year, scaling up by the number of periods
    return total_energy_bi * (365 / count)

