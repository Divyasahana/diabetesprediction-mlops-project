# src/predict.py
import mlflow.pyfunc
import pandas as pd
import os
from pathlib import Path

# Get path from environment variable or default
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    r"C:\Projects\DiabetesPrediction-mlops-project\mlruns\1\models\m-20a614b1b5244d369ca03809c249bb1a\artifacts"
)

# Convert Windows path to URI (file:///C:/...)
MODEL_URI = Path(MODEL_PATH).resolve().as_uri()

def load_model():
    try:
        print(f"Loading MLflow model from {MODEL_URI}")
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def make_prediction(model, features: dict):
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)
    return int(prediction[0])