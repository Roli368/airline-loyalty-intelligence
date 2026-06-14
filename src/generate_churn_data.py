import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
import joblib
def main():
    print("Loading data...")
    # Adjust paths based on where the script is run
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    flight = pd.read_csv(os.path.join(data_dir, "Customer Flight Activity.csv"))
    loyalty = pd.read_csv(os.path.join(data_dir, "Customer Loyalty History.csv"))
    
    print("Preprocessing flight data...")
    flight["Date"] = pd.to_datetime(dict(year=flight["Year"], month=flight["Month"], day=1))
    flight = flight.sort_values(["Loyalty Number", "Date"])
    
    customer_month = flight.groupby(["Loyalty Number", "Date"], as_index=False).agg({
        "Total Flights": "sum",
        "Distance": "sum",
        "Points Accumulated": "sum",
        "Points Redeemed": "sum"
    })
    
    print("Feature engineering...")
    customer_month["Flights_Last_3M"] = customer_month.groupby("Loyalty Number")["Total Flights"].transform(lambda x: x.rolling(3, min_periods=1).sum())
    customer_month["Distance_Last_3M"] = customer_month.groupby("Loyalty Number")["Distance"].transform(lambda x: x.rolling(3, min_periods=1).sum())
    customer_month["Points_Last_3M"] = customer_month.groupby("Loyalty Number")["Points Accumulated"].transform(lambda x: x.rolling(3, min_periods=1).sum())
    
    customer_month["Flight_Frequency"] = customer_month.groupby("Loyalty Number")["Total Flights"].transform("mean")
    customer_month["Avg_Distance_Per_Month"] = customer_month.groupby("Loyalty Number")["Distance"].transform("mean")
    customer_month["Avg_Points_Per_Month"] = customer_month.groupby("Loyalty Number")["Points Accumulated"].transform("mean")
    
    customer_month["Points_Redeem_Ratio"] = customer_month["Points Redeemed"] / (customer_month["Points Accumulated"] + 1)
    
    customer_month["Flight_Trend_3M"] = customer_month["Total Flights"] - customer_month.groupby("Loyalty Number")["Total Flights"].shift(3)
    customer_month["Flight_Trend_3M"] = customer_month["Flight_Trend_3M"].fillna(0)
    
    customer_month["Distance_Trend_3M"] = customer_month["Distance"] - customer_month.groupby("Loyalty Number")["Distance"].shift(3)
    customer_month["Distance_Trend_3M"] = customer_month["Distance_Trend_3M"].fillna(0)
    
    latest_date = customer_month["Date"].max()
    last_flight = customer_month.groupby("Loyalty Number")["Date"].max().reset_index()
    last_flight["Months_Since_Last_Flight"] = (latest_date - last_flight["Date"]).dt.days / 30
    
    customer_month = customer_month.merge(last_flight[["Loyalty Number", "Months_Since_Last_Flight"]], on="Loyalty Number", how="left")
    
    print("Generating target variable (Churn)...")
    future_flights = customer_month.groupby("Loyalty Number")["Total Flights"].transform(lambda x: x.shift(-1).fillna(0) + x.shift(-2).fillna(0) + x.shift(-3).fillna(0))
    customer_month["Churn"] = (future_flights == 0).astype(int)
    
    print("Merging with loyalty data...")
    data = customer_month.merge(loyalty, on="Loyalty Number", how="left")
    
    data["Membership_Age"] = 2018 - data["Enrollment Year"]
    data["Cancelled"] = np.where(data["Cancellation Year"].notna(), 1, 0)
    data["CLV_Per_Year"] = data["CLV"] / (data["Membership_Age"] + 1)
    
    features = [
        "Flights_Last_3M", "Distance_Last_3M", "Points_Last_3M",
        "Flight_Frequency", "Avg_Distance_Per_Month", "Avg_Points_Per_Month",
        "Points_Redeem_Ratio", "Flight_Trend_3M", "Distance_Trend_3M",
        "Months_Since_Last_Flight", "CLV", "CLV_Per_Year",
        "Salary", "Membership_Age", "Cancelled",
        "Gender", "Education", "Marital Status", "Loyalty Card", "Enrollment Type"
    ]
    
    num_cols = [
        "Flights_Last_3M", "Distance_Last_3M", "Points_Last_3M",
        "Flight_Frequency", "Avg_Distance_Per_Month", "Avg_Points_Per_Month",
        "Points_Redeem_Ratio", "Flight_Trend_3M", "Distance_Trend_3M",
        "Months_Since_Last_Flight", "CLV", "CLV_Per_Year",
        "Salary", "Membership_Age", "Cancelled"
    ]
    
    cat_cols = [
        "Gender", "Education", "Marital Status", "Loyalty Card", "Enrollment Type"
    ]
    
    print("Training XGBoost Pipeline...")
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])
    
    X = data[features]
    y = data["Churn"]
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X, y) # Training on full data to generate final predictions
    
    print("Generating predictions...")
    data["Churn_Probability"] = pipeline.predict_proba(X)[:,1]
    
    print("Assigning Risk and Interventions...")
    def risk_segment(prob):
        if prob >= 0.80: return "Critical Risk"
        elif prob >= 0.60: return "High Risk"
        elif prob >= 0.40: return "Medium Risk"
        else: return "Low Risk"
    
    data["Risk_Level"] = data["Churn_Probability"].apply(risk_segment)
    
    q1 = data["CLV"].quantile(0.25)
    q2 = data["CLV"].quantile(0.50)
    q3 = data["CLV"].quantile(0.75)
    
    def value_segment(clv):
        if clv >= q3: return "Premium"
        elif clv >= q2: return "Gold"
        elif clv >= q1: return "Silver"
        else: return "Bronze"
    
    data["Value_Segment"] = data["CLV"].apply(value_segment)
    
    def retention_action(row):
        risk = row["Risk_Level"]
        value = row["Value_Segment"]
        if risk == "Critical Risk" and value == "Premium": return "Personal Relationship Manager Call"
        elif risk == "Critical Risk" and value == "Gold": return "Free Upgrade Voucher"
        elif risk == "Critical Risk": return "Double Loyalty Points Campaign"
        elif risk == "High Risk" and value in ["Premium", "Gold"]: return "Priority Lounge Access"
        elif risk == "High Risk": return "20% Bonus Miles Offer"
        elif risk == "Medium Risk": return "Targeted Email Campaign"
        else: return "No Action Required"
    
    data["Recommended_Action"] = data.apply(retention_action, axis=1)
    
    def communication_channel(action):
        if action == "Personal Relationship Manager Call": return "Phone Call"
        elif action == "Free Upgrade Voucher": return "Email + SMS"
        elif action == "Priority Lounge Access": return "Email"
        elif action == "Double Loyalty Points Campaign": return "App Notification"
        elif action == "20% Bonus Miles Offer": return "Email"
        else: return "None"
    
    data["Communication_Channel"] = data["Recommended_Action"].apply(communication_channel)
    
    def intervention_timing(risk):
        if risk == "Critical Risk": return "Within 24 Hours"
        elif risk == "High Risk": return "Within 7 Days"
        elif risk == "Medium Risk": return "Within 30 Days"
        else: return "No Immediate Action"
    
    data["Action_Timeline"] = data["Risk_Level"].apply(intervention_timing)
    
    data["Revenue_At_Risk"] = data["CLV"] * data["Churn_Probability"]
    data["Priority_Score"] = data["Revenue_At_Risk"]
    
    # We only want the latest record for each customer so the dashboard isn't inflated with historical months
    latest_data = data.sort_values("Date").groupby("Loyalty Number").tail(1)
    
    retention_dashboard = latest_data[[
        "Loyalty Number", "CLV", "Churn_Probability", "Risk_Level", "Value_Segment",
        "Revenue_At_Risk", "Priority_Score", "Recommended_Action", 
        "Communication_Channel", "Action_Timeline"
    ]]
    
    out_file = os.path.join(data_dir, "churn_results.csv")
    retention_dashboard.to_csv(out_file, index=False)
    print(f"Success! Saved to {out_file}")

    # Save pipeline
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    pipeline_path = os.path.join(models_dir, "churn_pipeline.pkl")
    joblib.dump(pipeline, pipeline_path)
    print(f"Pipeline saved to {pipeline_path}")
    
    # Save raw features for simulation
    features_out = os.path.join(data_dir, "raw_features.csv")
    # We save the features exactly as expected by the pipeline
    latest_X = X.loc[latest_data.index].copy()
    latest_X.insert(0, "Loyalty Number", latest_data["Loyalty Number"])
    latest_X.to_csv(features_out, index=False)
    print(f"Features saved to {features_out}")

if __name__ == "__main__":
    main()
