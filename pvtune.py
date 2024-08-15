import pandas as pd
import numpy as np

def process_pvtune_output(file_path):
    '''
    The process_pvtune_output function processes a PVTune output CSV file to extract relevant data for solar tracker rows. It 
    performs several calculations, including identifying rows that correspond to the ends of tracker arrays (either external 
    or internal), determining the number of modules per row based on the difference in northing (N) coordinates, and calculating 
    the maximum reveal height for each row. The function then aggregates this data into a summary DataFrame, which groups tracker 
    rows by their reveal height and whether they are external or internal. Additionally, the function computes a weighted average 
    reveal height, considering the number of modules associated with each height. The function returns both the summary DataFrame 
    and the rounded weighted average reveal height, which can be used in further solar energy modeling and simulations.
    
    Parameters: 
    - file_path: file path to pvtune output
    
    Returns:
    - summary_df: dataframe which contains the buckets of modules
    - weighted_average_reveal_height: the average pv row height across all modules
    '''
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path, usecols=range(23))

    # Filter the DataFrame to only include rows where Description is "Ext_Array_END" or "Int_Array_END"
    array_end_df = df[(df['Description'] == 'Ext_Array_END') | (df['Description'] == 'Int_Array_END')]

    # Group by 'Tracker Row Id' and calculate the absolute difference in 'N' values
    abs_diff_n = array_end_df.groupby('Tracker Row Id')['N'].apply(lambda x: abs(x.iloc[0] - x.iloc[1]) if len(x) > 1 else None)

    # Map the calculated differences to the original DataFrame
    df['Difference in N'] = df['Tracker Row Id'].map(abs_diff_n)

    # Create a new column "Number of Modules" based on the conditions
    df['Number of Modules'] = df['Difference in N'].apply(
        lambda x: 78 if 250 < x < 270 else (104 if 380 < x < 400 else None)
    )

    # Move "Number of Modules" to be the first column
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Number of Modules')))
    df = df[cols]

    # Calculate the maximum "Reveal Height" for each "Tracker Row Id"
    max_reveal_height = df.groupby('Tracker Row Id')['Reveal Height'].transform('max')

    # Round up the "Max Reveal Height" to the nearest quarter foot
    rounded_reveal_height = 0.75 + (np.ceil(max_reveal_height * 4) / 4)

    # Add this rounded value back into the DataFrame
    df['Max Reveal Height'] = rounded_reveal_height

    # Move "Max Reveal Height" to be the first column
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Max Reveal Height')))
    df = df[cols]

    # Sort the DataFrame by "Tracker Row Id" in increasing order
    df = df.sort_values(by='Tracker Row Id', ascending=True)

    # Aggregate data to create the new DataFrame
    new_df = df.groupby('Tracker Row Id').agg({
        'Max Reveal Height': 'max',
        'Number of Modules': 'first',  # Assuming the first non-null value is what you want to retain
        'E': 'first',  # Take the first value, assuming E is consistent per Tracker Row Id
        'Z (Existing Grade)': 'first',  # Take the first value
        'Description': lambda x: 'Ext' if x.str.contains('Ext').any() else 'Int'  # Determine if any "Ext" or "Int" exists
    }).reset_index()

    # Group by 'Max Reveal Height' and 'Description', then aggregate the number of rows and total number of modules
    summary_df = new_df.groupby(['Max Reveal Height', 'Description']).agg(
        Number_of_Rows=('Tracker Row Id', 'count'),
        Number_of_Total_Modules=('Number of Modules', 'sum')
    ).reset_index()

    # Calculate the weighted sum of Max Reveal Height * Number of Total Modules
    summary_df['Weighted Height'] = summary_df['Max Reveal Height'] * summary_df['Number_of_Total_Modules']

    # Calculate the weighted average reveal height
    total_modules = summary_df['Number_of_Total_Modules'].sum()
    weighted_average_reveal_height = summary_df['Weighted Height'].sum() / total_modules

    # Return the final DataFrame and the rounded weighted average reveal height
    return summary_df, weighted_average_reveal_height
