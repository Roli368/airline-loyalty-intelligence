import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_segmentation(data_dir):
    print("Loading preprocessed data...")
    input_path = os.path.join(data_dir, 'processed_segmentation_data.csv')
    df = pd.read_csv(input_path)
    
    # Select features for clustering
    features = [
        'Recency (Months)', 
        'Total Flights', 
        'Distance', 
        'Points Redeemed', 
        'Salary'
    ]
    
    print("Scaling features...")
    # Fill any lingering NaNs just in case
    df[features] = df[features].fillna(df[features].median())
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    print("Running K-Means Clustering (k=5)...")
    # Setting a random state for reproducibility
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # Calculate average feature values for each cluster to assign personas
    cluster_centers = df.groupby('Cluster')[features].mean()
    print("Cluster Centers:")
    print(cluster_centers)
    
    # A simple heuristic rule to assign persona names based on the centers
    # (In a real scenario, this would be manually curated after reviewing the centers)
    def assign_persona(row):
        c = row['Cluster']
        if c == 0: return "High-Value Redeemers"
        elif c == 1: return "Inactive / Zero Flights"
        elif c == 2: return "Occasional / Declining Flyers"
        elif c == 3: return "Legacy High-Income Churned"
        elif c == 4: return "Active Points Hoarders"
        return "Unknown"
        
    df['Persona'] = df.apply(assign_persona, axis=1)
    
    import joblib
    
    # Save the models for future production deployment
    model_dir = os.path.join(data_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(kmeans, os.path.join(model_dir, 'kmeans_model.pkl'))
    print(f"Models serialized and saved to: {model_dir}")

    # Save the segments
    output_path = os.path.join(data_dir, 'customer_segments.csv')
    # Save just the loyalty number and the segmentation results for easy merging
    df[['Loyalty Number', 'Cluster', 'Persona', *features]].to_csv(output_path, index=False)
    print(f"Segmentation complete. Saved to: {output_path}")

if __name__ == "__main__":
    run_segmentation('c:/Users/rolir/Desktop/airlines')
