import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import os
import mlflow.sklearn

# --- MLflow Configuration ---
# Point to the local MLflow server directory
MLFLOW_TRACKING_URI = "file:./mlruns"

# Fetch the latest models from the registry
def load_latest_models():
    """Loads the latest registered models from MLflow."""
    try:
        # Set the MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Load Regression Model
        reg_model_uri = "models:/FuturePricePredictor/latest"
        reg_model = mlflow.sklearn.load_model(reg_model_uri)
        
        # Load Classification Model
        cls_model_uri = "models:/InvestmentClassifier/latest"
        cls_model = mlflow.sklearn.load_model(cls_model_uri)
        
        return reg_model, cls_model
    except Exception as e:
        st.error(f"Error loading models from MLflow registry. Please ensure 'model_trainer.py' has been run successfully.")
        st.error(f"Detailed Error: {e}")
        return None, None

# --- Load Models and Features ---
reg_model, cls_model = load_latest_models()

# Load a template to get column names and feature options for the app
try:
    with open('feature_template.json', 'r') as f:
        feature_template = json.load(f)
        
    # Extract unique values for categorical features from the template
    X_train = pd.read_csv('X_train.csv')
    
    CAT_OPTIONS = {}
    for col in X_train.select_dtypes(include='object').columns:
        CAT_OPTIONS[col] = sorted(X_train[col].unique().tolist())
        
except FileNotFoundError:
    st.error("Feature template or X_train not found. Please run 'data_processor.py' first.")
    CAT_OPTIONS = {}

# --- Helper Functions ---

def create_input_df(data):
    """Converts Streamlit input dictionary into a DataFrame for prediction."""
    
    if feature_template:
        first_key = next(iter(feature_template)) 
        feature_cols = list(feature_template[first_key].keys())
    else:
        return pd.DataFrame() 

    # Create the DataFrame from the input data, ensuring correct order
    input_df = pd.DataFrame([data], columns=feature_cols)
            
    return input_df


# --- Streamlit Application ---

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏡 Real Estate Investment Advisor")
st.markdown("---")

if reg_model and cls_model:
    st.sidebar.header("Property Details Input")

    # --- Input Form ---
    with st.sidebar.form("property_input_form"):
        # Numerical Inputs
        size_sqft = st.number_input("Size in SqFt (500 - 5000)", min_value=500, max_value=5000, value=2500, step=100)
        bhk = st.number_input("BHK (Bedrooms, Hall, Kitchen)", min_value=1, max_value=10, value=3, step=1)
        age = st.number_input("Age of Property (Years)", min_value=0, max_value=50, value=15, step=1)
        floor = st.number_input("Floor Number", min_value=0, max_value=30, value=5, step=1)
        total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=50, value=20, step=1)
        nearby_schools = st.number_input("Nearby Schools (Count)", min_value=0, max_value=10, value=5, step=1)
        nearby_hospitals = st.number_input("Nearby Hospitals (Count)", min_value=0, max_value=10, value=4, step=1)
        
        # Categorical Inputs
        state = st.selectbox("State", options=CAT_OPTIONS.get('State', ['Unknown']))
        city = st.selectbox("City", options=CAT_OPTIONS.get('City', ['Unknown']))
        locality = st.selectbox("Locality", options=CAT_OPTIONS.get('Locality', ['Unknown']))
        property_type = st.selectbox("Property Type", options=CAT_OPTIONS.get('Property_Type', ['Unknown']))
        furnished_status = st.selectbox("Furnished Status", options=CAT_OPTIONS.get('Furnished_Status', ['Unknown']))
        public_transport = st.selectbox("Public Transport Access", options=CAT_OPTIONS.get('Public_Transport_Accessibility', ['Unknown']))
        parking_space = st.selectbox("Parking Space", options=CAT_OPTIONS.get('Parking_Space', ['Unknown']))
        security = st.selectbox("Security", options=CAT_OPTIONS.get('Security', ['Unknown']))
        amenities = st.selectbox("Amenities (Placeholder)", options=CAT_OPTIONS.get('Amenities', ['Unknown'])) 
        facing = st.selectbox("Facing Direction", options=CAT_OPTIONS.get('Facing', ['Unknown']))
        owner_type = st.selectbox("Owner Type", options=CAT_OPTIONS.get('Owner_Type', ['Unknown']))

        # Current Price 
        current_price_lakhs = st.number_input("Current Price (in Lakhs)", min_value=10.0, max_value=1000.0, value=250.0, step=10.0)

        submitted = st.form_submit_button("Get Investment Advice")

    # --- Prediction Logic ---
    if submitted:
        
        # 1. Prepare Input Data
        try:
            transport_score_map = {'Low': 1, 'Medium': 2, 'High': 3}
            transport_norm = transport_score_map.get(public_transport, 1) / 3 
            
            infrastructure_score = (
                0.4 * (nearby_schools / 10) + 
                0.3 * (nearby_hospitals / 10) + 
                0.3 * transport_norm
            )
        except:
            infrastructure_score = 0

        input_data = {
            'State': state,
            'City': city,
            'Locality': locality,
            'Property_Type': property_type,
            'BHK': bhk,
            'Size_in_SqFt': size_sqft,
            'Age_of_Property': age,
            'Furnished_Status': furnished_status,
            'Floor_No': floor,
            'Total_Floors': total_floors,
            'Nearby_Schools': nearby_schools,
            'Nearby_Hospitals': nearby_hospitals,
            'Public_Transport_Accessibility': public_transport,
            'Parking_Space': parking_space,
            'Security': security,
            'Amenities': amenities,
            'Facing': facing,
            'Owner_Type': owner_type,
            
            # Engineered Features
            'Price_per_BHK': current_price_lakhs / bhk,
            'Infrastructure_Score': infrastructure_score
        }
        
        input_df = create_input_df(input_data)

        # 2. Make Predictions
        try:
            # Classification Prediction
            cls_prob = cls_model.predict_proba(input_df)[:, 1][0]
            cls_result = cls_model.predict(input_df)[0]
            
            # Regression Prediction
            reg_price = reg_model.predict(input_df)[0]
            
            # 3. Display Results
            st.header("🎯 Investment Recommendation")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Price Forecast (5 Years)")
                st.metric(
                    label="Estimated Future Price", 
                    value=f"₹ {reg_price:,.2f} Lakhs",
                    delta=f"{((reg_price / current_price_lakhs) - 1) * 100:.2f}% Projected Growth"
                )
                
            with col2:
                st.subheader("Investment Classification")
                if cls_result == 1:
                    st.success(f"✅ GOOD INVESTMENT Potential")
                else:
                    st.warning(f"⚠️ MODERATE/LOW INVESTMENT Potential")
                
                # FIX 1: Cast cls_prob (NumPy float) to native float for Streamlit compatibility
                st.progress(float(cls_prob), text=f"Confidence Score: {cls_prob*100:.2f}%")
                st.caption("Score represents the model's confidence in the 'Good Investment' classification.")
            
            st.markdown("---")
            st.subheader("Property Data Summary")
            
            # FIX 2 & 3: Cast the 'Value' column to string to fix PyArrow error and update width argument
            summary_df = (
                pd.DataFrame([input_data])
                .T
                .rename(columns={0: "Value"})
                .astype({'Value': str})
            )
            st.dataframe(summary_df, width='stretch')

        except Exception as e:
            st.error(f"Prediction failed. Error: {e}")
            st.warning("Ensure all inputs are valid and that 'data_processor.py' and 'model_trainer.py' ran correctly.")

else:
    st.info("Models are still loading or failed to load. Please wait or check your console for errors.")