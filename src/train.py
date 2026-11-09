import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model():
    print("Loading cleaned dataset...")
    df = pd.read_csv("data/processed/cleaned_retail.csv")
    
    # Ensure InvoiceDate is datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Set a reference date for Recency calculation (max date in dataset + 1 day)
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    print("Aggregating data to Customer RFM level...")
    # Group by Customer ID to create customer-level features
    customer_df = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency: Days since last purchase
        'Invoice': 'nunique',                                    # Frequency: Total unique orders/invoices
        'TotalSpend': ['sum', 'mean']                            # Monetary: Total Spend & Average Order Value
    })
    
    # Flatten multi-index column names
    customer_df.columns = ['Recency', 'Frequency', 'TotalSpend', 'AvgOrderValue']
    customer_df = customer_df.reset_index()
    
    # Define Features (X) and Target (y)
    # Target: Predict Total Spend (or future spend)
    X = customer_df[['Recency', 'Frequency', 'AvgOrderValue']]
    y = customer_df['TotalSpend']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Evaluation -> RMSE: {rmse:.2f} | R2 Score: {r2:.4f}")
    
    # Save the trained model artifact
    os.makedirs("models", exist_ok=True)
    model_path = "models/retail_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_model()