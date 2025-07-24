# House-rent-price-prediction-with-machine-learning
This project predicts house rent prices based on property features using machine learning algorithms. The application is built with Python, Scikit-learn, and Streamlit.
## Project Overview

This repository contains code for:
1. Training a machine learning model to predict house rent prices
2. A Streamlit web application for making rent predictions
3. Model persistence using joblib
## Dataset

The dataset contains rental listings from major Indian cities including:
- Bangalore
- Chennai
- Hyderabad
- Kolkata
- Mumbai
- Delhi

Features include:
- Property size and configuration (BHK, bathrooms)
- Location details (city, area type, floor)
- Furnishing status
- Tenant preferences
- Contact information
  ## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- streamlit
- joblib

Install dependencies with:
```bash
pip install -r requirements.txt

## Files
House_Rent_Dataset.csv - Rental listings data
train_model.py - Script to train and save the ML model
house_rent_prediction_app.py - Streamlit web application
requirements.txt - Python dependencies

## Usage
Train the model:
python train_model.py
This creates three joblib files:
house_rent_model.joblib - Trained Random Forest model
label_encoders.joblib - Encoders for categorical variables
feature_columns.joblib - List of features used in training
Run the Streamlit app:
streamlit run house_rent_prediction_app.py
Model Performance
The Random Forest Regressor achieves:

Mean Absolute Error: ~₹2,000-5,000
R-squared Score: ~0.75-0.85
Features
Predict rent based on property details
Data exploration dashboard
Responsive web interface
Real-time predictions
How to Use
Enter property details in the input section
Click "Predict Rent" button
View the predicted rent and additional information
# Project Structure
house-rent-prediction/
├── House_Rent_Dataset.csv
├── train_model.py
├── house_rent_prediction_app.py
├── requirements.txt
├── house_rent_model.joblib
├── label_encoders.joblib
├── feature_columns.joblib
└── README.md

And here's the requirements.txt file:

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
streamlit>=1.0.0
joblib>=1.1.0
