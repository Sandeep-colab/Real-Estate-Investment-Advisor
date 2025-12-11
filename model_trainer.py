import pandas as pd
import numpy as np
import joblib
import mlflow
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError # Added for robustness, though not strictly needed here

# --- Configuration ---
REG_EXPERIMENT_NAME = "Real_Estate_Price_Forecast"
CLS_EXPERIMENT_NAME = "Real_Estate_Investment_Classification"

# Load the data and preprocessor
def load_assets():
    """Loads processed data and the saved ColumnTransformer."""
    try:
        X_train = pd.read_csv('X_train.csv')
        X_test = pd.read_csv('X_test.csv')
        # Use .iloc[:, 0] to get the Series from the single-column DataFrame
        y_reg_train = pd.read_csv('y_reg_train.csv').iloc[:, 0]
        y_reg_test = pd.read_csv('y_reg_test.csv').iloc[:, 0]
        y_cls_train = pd.read_csv('y_cls_train.csv').iloc[:, 0]
        y_cls_test = pd.read_csv('y_cls_test.csv').iloc[:, 0]
        preprocessor = joblib.load('preprocessor.pkl')
        
        print("Assets loaded successfully.")
        return X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, preprocessor
    except FileNotFoundError as e:
        print(f"Error loading assets: {e}")
        print("Please ensure you have run 'data_processor.py' first.")
        return None, None, None, None, None, None, None

def evaluate_regression(y_test, y_pred):
    """Calculates and logs regression metrics."""
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    mlflow.log_metrics(metrics)
    print("Regression Metrics Logged:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    return metrics

def evaluate_classification(y_test, y_pred, y_prob):
    """Calculates and logs classification metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    }
    mlflow.log_metrics(metrics)
    print("Classification Metrics Logged:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    return metrics

def train_regression_model(X_train, X_test, y_train, y_test, preprocessor):
    """Trains and logs the XGBoost Regression model."""
    
    mlflow.set_experiment(REG_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="XGBoost_Price_Predictor"):
        print("\n--- Training Regression Model (Future Price) ---")
        
        # Model definition
        xgb_regressor = xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=7, 
            random_state=42,
            n_jobs=-1
        )
        
        # Full Pipeline (Preprocessor + Model)
        reg_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb_regressor)
        ])
        
        # Train
        reg_pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = reg_pipeline.predict(X_test)
        
        # Log Parameters
        mlflow.log_params(xgb_regressor.get_params())
        
        # Evaluate and Log Metrics
        evaluate_regression(y_test, y_pred)
        
        # Save and Register Model (for local MLflow tracking)
        mlflow.sklearn.log_model(
            sk_model=reg_pipeline, 
            artifact_path="future_price_model", 
            registered_model_name="FuturePricePredictor"
        )
        
        print("Regression Model logged and registered successfully.")
        return reg_pipeline

def train_classification_model(X_train, X_test, y_train, y_test, preprocessor):
    """Trains and logs the XGBoost Classification model."""
    
    mlflow.set_experiment(CLS_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="XGBoost_Investment_Classifier"):
        print("\n--- Training Classification Model (Good Investment) ---")
        
        # Model definition
        xgb_classifier = xgb.XGBClassifier(
            objective='binary:logistic', 
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42,
            eval_metric='logloss',
            # use_label_encoder=False is generally required for newer versions
            n_jobs=-1
        )
        
        # Full Pipeline (Preprocessor + Model)
        cls_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', xgb_classifier)
        ])
        
        # Train
        cls_pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = cls_pipeline.predict(X_test)
        y_prob = cls_pipeline.predict_proba(X_test)[:, 1]
        
        # Log Parameters
        mlflow.log_params(xgb_classifier.get_params())
        
        # Evaluate and Log Metrics
        evaluate_classification(y_test, y_pred, y_prob)
        
        # Save and Register Model (for local MLflow tracking)
        mlflow.sklearn.log_model(
            sk_model=cls_pipeline, 
            artifact_path="investment_classifier_model", 
            registered_model_name="InvestmentClassifier"
        )
        
        print("Classification Model logged and registered successfully.")
        return cls_pipeline

def main():
    assets = load_assets()
    if assets is None:
        return

    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, preprocessor = assets

    # Ensure MLflow tracking URI is set (default is local `./mlruns`)
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Train and log Regression Model
    reg_model = train_regression_model(X_train, X_test, y_reg_train, y_reg_test, preprocessor)
    
    # Train and log Classification Model
    cls_model = train_classification_model(X_train, X_test, y_cls_train, y_cls_test, preprocessor)
    
    # --- FIX: Save final models using joblib for streamlined cloud deployment ---
    print("\nSaving final model artifacts for deployment...")
    joblib.dump(reg_model, 'final_reg_model.pkl')
    joblib.dump(cls_model, 'final_cls_model.pkl')
    print("Model artifacts saved: final_reg_model.pkl, final_cls_model.pkl")
    # --- END FIX ---
    
    print("\n--- Training Complete ---")
    print("To view MLflow UI, run 'mlflow ui' in your terminal and navigate to http://localhost:5000")
    print("Next step: Update 'app.py' to load these PKL files, update 'requirements.txt', commit, and push.")

if __name__ == '__main__':
    # Fix for a warning related to data serialization
    pd.set_option('mode.chained_assignment', None) 
    main()