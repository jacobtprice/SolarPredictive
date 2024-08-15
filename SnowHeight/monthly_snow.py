from datetime import datetime
import pandas as pd
from meteostat import Daily, Point

def calculate_new_row_height(lat, lon, row_height):
    # Define the location
    location = Point(lat, lon)

    # Set the time period
    start = datetime(2019, 1, 1)
    end = datetime(2021, 12, 31)

    # Get daily data
    data = Daily(location, start, end)
    data = data.fetch()

    # Ensure the index is a datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Extract the month from the date index
    data['month'] = data.index.month

    # Fill NaN values in the 'snow' column with 0
    data['snow'] = data['snow'].fillna(0)

    # Group by month and calculate the average snowfall (convert from mm to meters)
    if 'snow' in data.columns:
        average_snowfall = data.groupby('month')['snow'].mean().reset_index()
        average_snowfall['average_snowfall'] = average_snowfall['snow'] / 1000

        # Drop the original 'snow' column to avoid confusion
        average_snowfall = average_snowfall.drop(columns=['snow'])

        # Rename columns for clarity
        average_snowfall.columns = ['month', 'average_snowfall']

        # Calculate the adjusted row height
        average_snowfall['adjusted_row_height'] = row_height - average_snowfall['average_snowfall']

        # Return the new DataFrame with month and adjusted row height
        return average_snowfall[['month', 'adjusted_row_height']]
    else:
        print("No 'snow' data found.")
        return pd.DataFrame(columns=['month', 'adjusted_row_height'])