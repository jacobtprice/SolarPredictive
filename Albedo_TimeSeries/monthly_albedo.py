"""
    monthly_albedo.py
    
    Description: Takes surface albedo data files from NSRDB database and creates a dataframe
    of the monthly averages at that site. Dataframe is used as a timeseries in the bifacial
    modelchain simulation.

    Created: 7/29/24
    Last Updated: 7/30/24
    Author: Jacob Price
"""

import pandas as pd
import os

def calculate_monthly_avg_albedo(folder_path):
    """
    Reads CSV files containing surface albedo data for multiple years, calculates the average
    surface albedo for each month for each year, and then calculates the average monthly albedo
    across all years.

    CSV Download Instructions:
        1. Go to https://nsrdb.nrel.gov/data-viewer
        2. Enter the coordinates for the proposed solar site
        3. Select "USA and Americas (10, 30, 60 min / 2km / 2019-2022)" from the datasets tab
        4. Select "Surface Albedo" from the attributes tab
        5. Select all years
        6. Select any interval
        7. Send to email, download zip of csv file
        8. Place folder in relevant folder location, rename if needed
        
    Parameters:
        folder_path (str): The path to the folder containing the CSV files with albedo data.
        
    Returns:
        pd.DataFrame: A DataFrame containing the average surface albedo for each month across all years.
    """
    
    # Initialize a list to store the DataFrames for each year
    all_yearly_data = []

    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            # Read each CSV file
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, skiprows=2)
            
            # Group by 'Year' and 'Month' and calculate the average 'Surface Albedo'
            monthly_avg_albedo = df.groupby(['Year', 'Month'])['Surface Albedo'].mean().reset_index()
            
            # Append the DataFrame to the list
            all_yearly_data.append(monthly_avg_albedo)
    
    # Concatenate all the yearly data into a single DataFrame
    combined_data = pd.concat(all_yearly_data)
    
    # Group by 'Month' and calculate the average 'Surface Albedo' across all years
    avg_monthly_albedo = combined_data.groupby('Month')['Surface Albedo'].mean().reset_index()
    
    # Return the resulting DataFrame
    return avg_monthly_albedo

