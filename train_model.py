# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import re

# Create sample data since the original file is malformed
def create_sample_data():
    data = []
    cities = ["Mumbai", "Chennai", "Bangalore", "Hyderabad", "Kolkata"]
    area_types = ["Super Area", "Carpet Area", "Built Area"]
    furnishing = ["Furnished", "Semi-Furnished", "Unfurnished"]
    tenants = ["Bachelors/Family", "Bachelors", "Family"]
    contacts = ["Contact Owner", "Contact Agent"]
    
    for i in range(1000):
        city = np.random.choice(cities)
        bhk = np.random.randint(1, 5)
        size = np.random.randint(500, 3000)
        bathroom = np.random.randint(1, bhk+2)
        floor_num = np.random.randint(0, 20)
        total_floors = np.random.randint(floor_num, 30)
        area_type = np.random.choice(area_types)
        furnishing_status = np.random.choice(furnishing)
        tenant_preferred = np.random.choice(tenants)
        point_of_contact = np.random.choice(contacts)
        
        # Generate rent based on features (simplified model)
        base_rent = 1000
        rent = (base_rent * bhk + 
                size * 10 + 
                bathroom * 2000 + 
                floor_num * 500 +
                np.random.randint(-5000, 5000))
        
        # City multipliers
        city_multiplier = {"Mumbai": 2.0, "Bangalore": 1.5, "Hyderabad": 1.2, "Chennai": 1.0, "Kolkata": 0.8}
        rent = int(rent * city_multiplier[city])
        
        data.append({
            'Posted On': '2022-06-15',
            'BHK': bhk,
            'Rent': rent,
            'Size': size,
            'Floor': f"{floor_num} out of {total_floors}",
            'Area Type': area_type,
            'Area Locality': 'Sample Locality',
            'City': city,
            'Furnishing Status': furnishing_status,
            'Tenant Preferred': tenant_preferred,
            'Bathroom': bathroom,
            'Point of Contact': point_of_contact
        })
    
    return pd.DataFrame(data)

# Preprocess data
def preprocess_data(df):
    # Convert data types
    df['Posted On'] = pd.to_datetime(df['Posted On'])
    df['BHK'] = pd.to_numeric(df['BHK'], errors='coerce')
    df['Rent'] = pd.to_numeric(df['Rent'], errors='coerce')
    df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
    df['Bathroom'] = pd.to_numeric(df['Bathroom'], errors='coerce')

    # Extract floor information
    def parse_floor(floor_str):
        if pd.isna(floor_str):
            return pd.Series([np.nan, np.nan])
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
                return pd.Series([floor_num, total_floors])
            except:
                return pd.Series([np.nan, np.nan])
        else:
            return pd.Series([np.nan, np.nan])

    df[['Floor_Number', 'Total_Floors']] = df['Floor'].apply(parse_floor)

    # Extract date features
    df['Posted_Year'] = df['Posted On'].dt.year
    df['Posted_Month'] = df['Posted On'].dt.month
    df['Posted_Day'] = df['Posted On'].dt.day

    # Encode categorical variables
    categorical_columns = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']
    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders

# Train model
def train_model(df):
    # Select features for modeling
    feature_columns = [
        'BHK', 'Size', 'Bathroom', 'Floor_Number', 'Total_Floors',
        'Posted_Year', 'Posted_Month', 'Posted_Day'
    ] + [col + '_encoded' for col in ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']]

    X = df[feature_columns]
    y = df['Rent']

    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, feature_columns

# Main execution
if __name__ == "__main__":
    # Create and process data
    df = create_sample_data()
    df, label_encoders = preprocess_data(df)
    
    # Train model
    model, feature_columns = train_model(df)
    
    # Save model and encoders
    joblib.dump(model, 'house_rent_model.joblib')
    joblib.dump(label_encoders, 'label_encoders.joblib')
    joblib.dump(feature_columns, 'feature_columns.joblib')
    
    print("Model and encoders saved successfully!")