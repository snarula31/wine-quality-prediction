# streamlit_app.py

import streamlit as st
import pandas as pd
from pipeline.prediction_pipeline import PredictionPipeline

# --- Caching the Prediction Pipeline ---
# This is a key performance optimization. 
# It loads the model and preprocessor only once when the app starts,
# and keeps them in memory for all subsequent user sessions.
@st.cache_resource
def load_pipeline():
    return PredictionPipeline()

pipeline = load_pipeline()

# --- Page Configuration ---
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# --- Application Title and Description ---
st.title("🍷 Wine Quality Prediction")
st.markdown("""
This application predicts the quality of a wine (`Good` vs. `Normal`) based on its physicochemical properties.
Please adjust the sliders on the left to input the wine's features and click 'Predict' to see the result.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Input Wine Features")

def user_input_features():
    """Creates sliders in the sidebar for user input."""
    # Note: These default values and ranges are illustrative. 
    # You should adjust them based on the actual range of your dataset for a better user experience.
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.0, 16.0, 7.4)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.1, 1.6, 0.7)
    citric_acid = st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.0)
    residual_sugar = st.sidebar.slider('Residual Sugar', 0.9, 16.0, 1.9)
    chlorides = st.sidebar.slider('Chlorides', 0.01, 0.65, 0.076)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 1, 72, 11)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', 6, 290, 34)
    density = st.sidebar.slider('Density', 0.990, 1.003, 0.996)
    pH = st.sidebar.slider('pH', 2.7, 4.1, 3.3)
    sulphates = st.sidebar.slider('Sulphates', 0.3, 2.0, 0.56)
    alcohol = st.sidebar.slider('Alcohol', 8.0, 17.0, 10.4)

    # Store inputs in a dictionary
    data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    
    # Convert dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# --- Main Panel for Displaying Inputs and Predictions ---
st.subheader("User Input Features")
st.write(input_df)

# --- Prediction Button and Output ---
if st.button("Predict Wine Quality"):
    try:
        prediction = pipeline.predict(input_df)
        
        # Display the result
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success("The wine is predicted to be of **Good Quality**! 🎉")
        else:
            st.warning("The wine is predicted to be of **Normal Quality**. 🙂")
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")