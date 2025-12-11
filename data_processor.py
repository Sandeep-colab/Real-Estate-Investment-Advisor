import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # Added import for joblib

# --- Configuration ---
FILE_PATH = 'Data/india_housing_prices.csv'
# Assume a conservative average annual appreciation rate for the Regression target
ASSUMED_GROWTH_RATE = 0.08  # 8% per annum
GROWTH_PERIOD = 5           # 5 years
# FIX: Increased threshold from 0.40 to 0.60 to create balanced classes.
GOOD_INVESTMENT_THRESHOLD_RATE = 0.60 

def load_data(path):
    """Loads the dataset."""
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        # Handle Windows path separator in the original path if the file is in 'Data' folder
        try:
             path = path.replace('\\', '/')
             df = pd.read_csv(path)
             print(f"Data loaded successfully (via adjusted path). Shape: {df.shape}")
             return df
        except FileNotFoundError:
            print(f"Error: File not found at {path}")
            return None

def engineer_targets(df):
    """
    Creates the two target variables:
    1. Future_Price_5Y (Regression)
    2. Good_Investment (Classification)
    """
    
    print("Engineering Target Variables...")
    
    # 1. Regression Target: Future Price
    future_price_factor = (1 + ASSUMED_GROWTH_RATE) ** GROWTH_PERIOD
    df['Future_Price_5Y'] = df['Price_in_Lakhs'] * future_price_factor
    print(f"Created 'Future_Price_5Y' using {ASSUMED_GROWTH_RATE*100}% annual growth.")

    # 2. Classification Target: Good Investment
    median_price_per_sqft = df['Price_per_SqFt'].median()
    
    df['simulated_growth_rate'] = (
        1.08 +  # Base growth
        (df['BHK'] > 3).astype(int) * 0.01 + 
        (df['Age_of_Property'] >= 10).astype(int) * 0.005 + 
        (df['Price_per_SqFt'] < median_price_per_sqft).astype(int) * 0.015
    )
    
    # Calculate simulated appreciation
    df['simulated_appreciation'] = (df['simulated_growth_rate'] ** GROWTH_PERIOD) - 1
    
    # Classification Rule: Good Investment if appreciation is above the high threshold
    df['Good_Investment'] = (
        (df['simulated_appreciation'] >= GOOD_INVESTMENT_THRESHOLD_RATE) &
        (df['Availability_Status'] != 'Sold')
    ).astype(int)
    
    print(f"Created 'Good_Investment' (1/0). Good investments count: {df['Good_Investment'].sum()}")
    
    return df

def feature_engineer(df):
    """Performs custom feature engineering."""
    print("Performing Feature Engineering...")
    
    # 1. Interaction Feature: Price per BHK
    df['Price_per_BHK'] = np.where(df['BHK'] > 0, df['Price_in_Lakhs'] / df['BHK'], df['Price_in_Lakhs'])
    
    # 2. Density/Accessibility Score (Higher is better)
    df['School_Score'] = df['Nearby_Schools'] / df['Nearby_Schools'].max()
    df['Hospital_Score'] = df['Nearby_Hospitals'] / df['Nearby_Hospitals'].max()
    
    accessibility_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df['Transport_Score'] = df['Public_Transport_Accessibility'].map(accessibility_map) / 3
    
    # Combined Locality Score (weighted average)
    df['Infrastructure_Score'] = (
        0.4 * df['School_Score'] + 
        0.3 * df['Hospital_Score'] + 
        0.3 * df['Transport_Score']
    )
    
    # Drop intermediate columns
    df = df.drop(columns=['School_Score', 'Hospital_Score', 'Transport_Score', 'ID'])
    
    print(f"Engineered 'Price_per_BHK' and 'Infrastructure_Score'.")
    return df

def preprocess_data(df):
    """Handles missing values, duplicates, and sets up the preprocessing pipeline."""
    
    # 1. Handle Duplicates and Missing Values (Initial check)
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_rows - df.shape[0]} duplicates.")
    
    # 2. Feature and Target Selection
    cols_to_drop = [
        'Price_in_Lakhs',       
        'Price_per_SqFt',       
        'Year_Built',           
        'Availability_Status',  
        'simulated_growth_rate',
        'simulated_appreciation' 
    ]
    df_processed = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Define features (X) and targets (y)
    X = df_processed.drop(columns=['Future_Price_5Y', 'Good_Investment'])
    y_reg = df_processed['Future_Price_5Y']
    y_cls = df_processed['Good_Investment']

    # 3. Define Preprocessing Pipeline
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    print("Preprocessing pipeline set up (StandardScaler + OneHotEncoder).")
    return X, y_reg, y_cls, preprocessor

def main():
    """Main function to run the data processing pipeline."""
    
    df = load_data(FILE_PATH)
    if df is None:
        return

    # Step 1: Feature Engineering & Target Creation
    df = engineer_targets(df)
    df = feature_engineer(df)
    
    # Step 2: Preprocessing Setup and Splitting
    X, y_reg, y_cls, preprocessor = preprocess_data(df)
    
    # Split data into training and testing sets
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42
    )
    
    print("\n--- Data Split Summary ---")
    print(f"X_train shape: {X_train.shape}, y_reg_train shape: {y_reg_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_reg_test shape: {y_reg_test.shape}")
    print(f"Classification Positive Class (Good Investment) in Train: {y_cls_train.mean():.2f}")
    
    # Save the processed data and preprocessor for the model training step
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_reg_train.to_csv('y_reg_train.csv', index=False)
    y_reg_test.to_csv('y_reg_test.csv', index=False)
    y_cls_train.to_csv('y_cls_train.csv', index=False)
    y_cls_test.to_csv('y_cls_test.csv', index=False)
    
    # Save the preprocessor
    joblib.dump(preprocessor, 'preprocessor.pkl')
    
    # Save the original features for Streamlit input validation
    # FINAL FIX: Use orient='index' to match the loading logic in app.py
    X.head(1).to_json('feature_template.json', orient='index')
    
    print("\nData preprocessing and split complete.")
    print("Files saved: X_train.csv, y_reg_train.csv, preprocessor.pkl, etc.")
    print("Next step: Run 'model_trainer.py'.")

if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None) 
    main()