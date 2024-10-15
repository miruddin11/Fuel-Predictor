import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('fuel_efficiency_model.pkl')

# Streamlit app title
st.title('Maruti Suzuki Fuel Efficiency Predictor')

# Collect user inputs for prediction
engine_size = st.number_input('Engine Size (in liters)', min_value=0.0, max_value=5.0, step=0.1)
weight = st.number_input('Vehicle Weight (in kg)', min_value=500, max_value=3000, step=10)
horsepower = st.number_input('Horsepower', min_value=50, max_value=500, step=5)
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel'])

# Convert fuel type to numerical value
fuel_type_encoded = 1 if fuel_type == 'Petrol' else 0

# When the user clicks the button, make the prediction
if st.button('Predict Fuel Efficiency'):
    input_features = np.array([[engine_size, weight, horsepower, fuel_type_encoded]])
    prediction = model.predict(input_features)
    st.write(f'Estimated Fuel Efficiency: {prediction[0]:.2f} km/l')