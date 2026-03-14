import mlflow.pyfunc
import pandas as pd
from pathlib import Path
import os

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    r"C:\Projects\DiabetesPrediction-mlops-project\mlruns\1\models\m-20a614b1b5244d369ca03809c249bb1a\artifacts"
)

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
    try:
        input_df = pd.DataFrame([features])
        prediction = model.predict(input_df)
        
        # Convert probability to 0/1 if necessary
        result = int(prediction[0]) if prediction[0] in [0, 1] else int(prediction[0] > 0.5)
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None