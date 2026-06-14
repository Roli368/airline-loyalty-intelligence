# Airline Loyalty Analytics

This repository contains the codebase for analyzing airline loyalty data. It focuses on customer segmentation and churn prediction to help identify high-risk members and suggest targeted retention strategies.

## Project Structure

- `app.py`: Streamlit dashboard for data visualization and model interaction.
- `src/generate_churn_data.py`: Pipeline for training the XGBoost churn model and exporting results.
- `src/data_preprocessing.py`: Script for cleaning raw data and engineering RFM features.
- `src/segmentation.py`: K-Means clustering script for assigning customer personas.
- `data/`: Directory containing raw and processed CSV files.
- `models/`: Directory storing serialized model artifacts (`churn_pipeline.pkl`, `kmeans_model.pkl`).

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run data processing and model training:**
   If the `models/` directory is empty or you've added new raw data, generate the models by running:
   ```bash
   python src/generate_churn_data.py
   ```

3. **Start the dashboard:**
   ```bash
   streamlit run app.py
   ```
   The application will be available at `http://localhost:8501`.

## Core Components

* **Churn Prediction:** An XGBoost classifier that predicts the probability of a customer leaving based on their recent flight and points history.
* **Segmentation:** K-Means clustering implementation that groups users into behavioral personas.
* **Dashboard Features:** Includes a scenario simulator to test how changes in customer behavior impact churn probability, and a baseline ROI calculator for evaluating retention interventions.
