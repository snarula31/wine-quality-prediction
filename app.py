# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline

# --- 1. Define Data Schema ---
class WineData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# --- 2. Initialize the FastAPI App ---
app = FastAPI()

# --- 3. Load the Prediction Pipeline ---
# This loads the model and preprocessor into memory once when the app starts.
try:
    pipeline = PredictionPipeline()
except Exception as e:
    # If loading fails, the app won't start, and the error will be clear.
    print(f"Failed to load the prediction pipeline: {e}")
    pipeline = None

# --- 4. Define the Prediction Endpoint ---
@app.post("/predict")
def predict(data: WineData):
    if not pipeline:
        return {"error": "Prediction pipeline is not available."}
        
    try:
        # Convert the incoming Pydantic data object into a dictionary
        data_dict = data.dict()
        
        # Create a pandas DataFrame from the dictionary
        # The column order must match what the model was trained on.
        features = pd.DataFrame([data_dict])
        
        # Get a prediction
        prediction = pipeline.predict(features)

         # Rename the columns to match the training data (replace underscores with spaces)
        features.columns = [col.replace('_', ' ') for col in features.columns]
        
        # Interpret the prediction
        result = "Great Quality" if prediction == 1 else "Normal Quality"
        
        # Return the result as JSON
        return {"prediction": result}

    except Exception as e:
        # Return an error response if something goes wrong
        return {"error": str(e)}

# --- 5. Optional: Add a Root Endpoint ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Wine Quality Prediction API"}


# import pickle
# from flask import Flask, request, render_template
# import numpy as np
# import pandas as pd

# from sklearn.preprocessing import StandardScaler
# from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData

# application = Flask(__name__)

# app = application

# # Route for the home page
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predictdata', methods=['GET','POST'])
# def predict_datapoint():
#     if request.method == 'GET':
#         return render_template('home.html')
#     else:
#         data = CustomData(
#             fixed_acidity=request.form.get('fixed acidity'),
#             volatile_acidity=request.form.get('volatile acidity'),
#             citric_acid=request.form.get('citric acid'),
#             residual_sugar=request.form.get('residual sugar'),
#             chlorides=request.form.get('chlorides'),
#             free_sulfur_dioxide=float(request.form.get('free sulfur dioxide')),
#             total_sulfur_dioxide=float(request.form.get('total sulfur dioxide')),
#             density=float(request.form.get('density')),
#             pH=float(request.form.get('pH')),
#             sulphates=float(request.form.get('sulphates')),
#             alcohol=float(request.form.get('alcohol'))
#         )

#         pred_df = data.get_data_as_data_frame()
#         print(pred_df)

#         predict_pipeline = PredictionPipeline()
#         prediction = predict_pipeline.predict(pred_df)
#         results = "Great Quality" if prediction == 1 else "Normal Quality"
#         return render_template('home.html', results=results)


if __name__ == "__main__":
    app.run(host='0.0.0.0')