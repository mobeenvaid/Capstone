import streamlit as st
import pickle
import pandas as pd

# Load the trained SVC model from pickle file
with open('svc.pkl', 'rb') as f:
    svc = pickle.load(f)

# Define the Streamlit app
st.title("Insurance Claim Fraud Detection")

# Sidebar for user input
st.sidebar.header("User Input")

# Define input fields
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
insurance_type = st.sidebar.selectbox("Insurance Type", options=["Type1", "Type2", "Type3"])  # Update with actual types
claim_amount = st.sidebar.number_input("Claim Amount", min_value=0.0, value=100.0)
diagnosis_code = st.sidebar.number_input("Diagnosis Code", min_value=0, value=1)
treatment_code = st.sidebar.number_input("Treatment Code", min_value=0, value=1)
hospital_code = st.sidebar.number_input("Hospital Code", min_value=0, value=1)
doctor_fee = st.sidebar.number_input("Doctor Fee", min_value=0.0, value=50.0)
policy_type = st.sidebar.selectbox("Policy Type", options=["Policy1", "Policy2", "Policy3"])  # Update with actual types

# Create a DataFrame from the input values
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Insurance_Type': [insurance_type],
    'Claim_Amount': [claim_amount],
    'Diagnosis_Code': [diagnosis_code],
    'Treatment_Code': [treatment_code],
    'Hospital_Code': [hospital_code],
    'Doctor_Fee': [doctor_fee],
    'Policy_Type': [policy_type]
})

# Encode categorical variables
input_data_encoded = pd.get_dummies(input_data)

# Align the input data with the training data (i.e., ensure all columns are present)
feature_names = svc.feature_names_in_
for col in feature_names:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

input_data_encoded = input_data_encoded[feature_names]

# Make prediction
prediction = svc.predict(input_data_encoded)
prediction_proba = svc.predict_proba(input_data_encoded)

# Display results
st.write(f"Prediction: {'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent'}")
st.write(f"Prediction Probability: {prediction_proba[0]}")
