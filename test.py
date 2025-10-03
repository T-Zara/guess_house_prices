import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# Paths for saved files
MODEL_PATH = Path("house_model.pkl")
SCALER_PATH = Path("scaler.pkl")

# Check files exist
if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    st.error("Model or scaler file not found. Run the training script (train_save_model.py) first to create house_model.pkl and scaler.pkl.")
else:
    # Load model & scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    st.title("🏠 House Price Predictor (California features)")
    st.markdown("This model expects the 8 features from the California housing dataset: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude.")

    # Inputs that correspond to the California housing features
    MedInc = st.number_input("Median Income (tens of thousands USD)", min_value=0.0, max_value=50.0, value=3.0, step=0.1, format="%.2f")
    HouseAge = st.number_input("House Age (years)", min_value=0.0, max_value=100.0, value=20.0, step=1.0, format="%.1f")
    AveRooms = st.number_input("Average Rooms per Household", min_value=0.0, max_value=50.0, value=5.0, step=0.1, format="%.2f")
    AveBedrms = st.number_input("Average Bedrooms per Household", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.2f")
    Population = st.number_input("Population of Block Group", min_value=1, max_value=1000000, value=1000, step=1)
    AveOccup = st.number_input("Average Occupants per Household", min_value=0.0, max_value=50.0, value=3.0, step=0.1, format="%.2f")
    Latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=34.0, step=0.001, format="%.3f")
    Longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-118.0, step=0.001, format="%.3f")

    # Prediction
    if st.button("Predict Price"):
        try:
            # Create input array with shape (1, 8) — same order as training features
            X_input = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]], dtype=float)

            # Scale using the saved scaler (must have been fit on 8-feature data)
            X_scaled = scaler.transform(X_input)

            # Predict — model target for California dataset is in units of 100,000 USD
            pred = model.predict(X_scaled)[0]

            # Multiply by 100,000 to get dollars (common convention for this dataset)
            st.success(f"Estimated Median House Value: ${pred * 100000:,.2f}")
            st.caption("Note: California dataset target is in units of 100,000 USD.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
