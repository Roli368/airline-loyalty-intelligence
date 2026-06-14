import pandas as pd
import numpy as np
import os

def preprocess_data(data_dir):
    print("Loading datasets...")
    history_path = os.path.join(data_dir, 'Customer Loyalty History.csv')
    activity_path = os.path.join(data_dir, 'Customer Flight Activity.csv')
    
    df_history = pd.read_csv(history_path)
    df_activity = pd.read_csv(activity_path)
    
    print("Cleaning History Data...")
    # Impute missing Salary with the median salary for their Education level
    df_history['Salary'] = df_history.groupby('Education')['Salary'].transform(lambda x: x.fillna(x.median()))
    # There might still be a few missing if an entire category is missing, so fill remaining with overall median
    df_history['Salary'] = df_history['Salary'].fillna(df_history['Salary'].median())
    
    print("Engineering Behavioral Features from Activity...")
    # Create a sortable 'Date' column for recency calculation (using end of month)
    df_activity['Date'] = pd.to_datetime(df_activity['Year'].astype(str) + '-' + df_activity['Month'].astype(str) + '-01') + pd.offsets.MonthEnd(0)
    
    # Calculate recency: Months since last flight. 
    # We define the "current" date as the max date in the dataset
    max_date = df_activity['Date'].max()
    
    # Filter only rows where a flight was booked to find the last flight date
    flights_only = df_activity[df_activity['Total Flights'] > 0]
    last_flight_dates = flights_only.groupby('Loyalty Number')['Date'].max().reset_index()
    last_flight_dates.columns = ['Loyalty Number', 'Last Flight Date']
    
    # Aggregate overall metrics
    agg_activity = df_activity.groupby('Loyalty Number').agg({
        'Total Flights': 'sum',
        'Distance': 'sum',
        'Points Accumulated': 'sum',
        'Points Redeemed': 'sum',
        'Date': 'count' # total months active
    }).reset_index()
    
    agg_activity.rename(columns={'Date': 'Months Active'}, inplace=True)
    
    # Merge recency back
    agg_activity = pd.merge(agg_activity, last_flight_dates, on='Loyalty Number', how='left')
    
    # Calculate Recency in Months (if they never flew, assign a large number, e.g., max months + 12)
    agg_activity['Recency (Months)'] = ((max_date - agg_activity['Last Flight Date']) / np.timedelta64(1, 'D') / 30.44).fillna(999).astype(int)
    
    print("Merging Datasets...")
    df_final = pd.merge(df_history, agg_activity, on='Loyalty Number', how='left')
    
    # If there are members in history with no activity records at all, fill their stats with 0
    df_final['Total Flights'] = df_final['Total Flights'].fillna(0)
    df_final['Distance'] = df_final['Distance'].fillna(0)
    df_final['Points Accumulated'] = df_final['Points Accumulated'].fillna(0)
    df_final['Points Redeemed'] = df_final['Points Redeemed'].fillna(0)
    df_final['Months Active'] = df_final['Months Active'].fillna(0)
    df_final['Recency (Months)'] = df_final['Recency (Months)'].fillna(999)
    
    output_path = os.path.join(data_dir, 'processed_segmentation_data.csv')
    df_final.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    preprocess_data('c:/Users/rolir/Desktop/airlines')
