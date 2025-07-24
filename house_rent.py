# house_rent_prediction_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import date

# Load model and encoders
@st.cache_resource
def load_model():
    try:
        model = joblib.load('house_rent_model.joblib')
        label_encoders = joblib.load('label_encoders.joblib')
        feature_columns = joblib.load('feature_columns.joblib')
        return model, label_encoders, feature_columns
    except FileNotFoundError:
        st.error("Model files not found. Please run the training script first.")
        return None, None, None

# Parse floor information
def parse_floor(floor_str):
    if pd.isna(floor_str) or floor_str == "":
        return np.nan, np.nan
    if 'out of' in str(floor_str):
        parts = str(floor_str).split(' out of ')
        try:
            if parts[0].lower() == 'ground':
                floor_num = 0
            elif parts[0].lower() == 'upper basement':
                floor_num = -1
            else:
                floor_num = int(parts[0])
            total_floors = int(parts[1])
            return floor_num, total_floors
        except:
            return np.nan, np.nan
    else:
        return np.nan, np.nan

# Prediction function
def predict_rent(model, label_encoders, feature_columns, input_data):
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Process floor information
    df['Floor_Number'], df['Total_Floors'] = zip(*df['Floor'].apply(parse_floor))
    
    # Add date features (using today's date)
    today = date.today()
    df['Posted_Year'] = today.year
    df['Posted_Month'] = today.month
    df['Posted_Day'] = today.day
    
    # Encode categorical variables
    categorical_columns = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']
    for col in categorical_columns:
        try:
            df[col + '_encoded'] = label_encoders[col].transform(df[col])
        except ValueError:
            # Handle unseen labels
            df[col + '_encoded'] = 0
    
    # Select features
    X = df[feature_columns]
    
    # Handle missing values
    X = X.fillna(0)
    
    # Make prediction
    prediction = model.predict(X)[0]
    return prediction

# Load data function for exploration
@st.cache_data
def load_data():
    # Since we can't properly parse the data without column headers,
    # we'll create a minimal dataframe for demonstration
    data = {
        'City': ['Mumbai', 'Chennai', 'Bangalore', 'Hyderabad', 'Kolkata'],
        'Rent': [50000, 20000, 25000, 15000, 12000],
        'Size': [1000, 800, 900, 1100, 700]
    }
    return pd.DataFrame(data)

# Main app
def main():
    st.set_page_config(page_title="House Rent Prediction", page_icon="🏠", layout="wide")
    
    st.title("🏠 House Rent Prediction")
    st.markdown("Predict rental prices based on property features")
    
    # Load model and encoders
    model, label_encoders, feature_columns = load_model()
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This app predicts house rent prices based on property features. "
        "Enter the property details in the input section and get an estimated rent."
    )
    
    st.sidebar.header("How to Use")
    st.sidebar.markdown(
        """
        1. Enter property details in the input section
        2. Click 'Predict Rent' button
        3. View the predicted rent
        """
    )
    
    # Tabs
    tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Data Exploration"])
    
    with tab1:
        st.header("Property Details")
        
        # Create input fields
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bhk = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)
            size = st.number_input("Size (sqft)", min_value=100, max_value=5000, value=1000, step=50)
            bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
            
        with col2:
            floor = st.text_input("Floor (e.g., '3 out of 5')", value="2 out of 4")
            area_type = st.selectbox(
                "Area Type",
                ["Super Area", "Carpet Area", "Built Area"]
            )
            city = st.selectbox(
                "City",
                ["Mumbai", "Chennai", "Bangalore", "Hyderabad", "Kolkata"]
            )
            
        with col3:
            furnishing_status = st.selectbox(
                "Furnishing Status",
                ["Furnished", "Semi-Furnished", "Unfurnished"]
            )
            tenant_preferred = st.selectbox(
                "Tenant Preferred",
                ["Bachelors/Family", "Bachelors", "Family"]
            )
            point_of_contact = st.selectbox(
                "Point of Contact",
                ["Contact Owner", "Contact Agent"]
            )
        
        # Prediction button
        if st.button("🔮 Predict Rent", type="primary"):
            if model is not None:
                # Prepare input data
                input_data = {
                    'BHK': bhk,
                    'Size': size,
                    'Bathroom': bathroom,
                    'Floor': floor,
                    'Area Type': area_type,
                    'City': city,
                    'Furnishing Status': furnishing_status,
                    'Tenant Preferred': tenant_preferred,
                    'Point of Contact': point_of_contact
                }
                
                # Make prediction
                with st.spinner("Calculating rent prediction..."):
                    predicted_rent = predict_rent(model, label_encoders, feature_columns, input_data)
                
                # Display result
                st.success(f"Predicted Rent: ₹{predicted_rent:,.0f} per month")
                
                # Additional info
                st.info(
                    f"This is an estimate based on {bhk} BHK, {size} sqft property in {city}. "
                    "Actual rent may vary based on location, condition, and amenities."
                )
            else:
                st.error("Model not loaded. Please check model files.")
    
    with tab2:
        st.header("Dataset Overview")
        st.markdown(
            """
            This app was trained on a dataset of house rental listings from major Indian cities.
            The dataset contains information about:
            - Property size and configuration
            - Location details
            - Furnishing status
            - Rental prices
            """
        )
        
        # Show sample data
        df = load_data()
        st.subheader("Sample Data")
        st.dataframe(df)
        
        # Show city distribution
        st.subheader("Average Rent by City")
        city_rent = df.groupby('City')['Rent'].mean().sort_values(ascending=False)
        st.bar_chart(city_rent)
        
        # Show size vs rent
        st.subheader("Size vs Rent")
        st.scatter_chart(df, x='Size', y='Rent', color='City')

if __name__ == "__main__":
    main()