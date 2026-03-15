from fastapi import FastAPI
import joblib
import numpy as np
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "Diabetes Prediction API running"}


@app.post("/v1/predict")
def predict(data: dict):

    features = [
        data["pregnancies"],
        data["glucose"],
        data["blood_pressure"],
        data["skin_thickness"],
        data["insulin"],
        data["bmi"],
        data["diabetes_pedigree_function"],
        data["age"],
    ]

    input_data = np.array(features).reshape(1, -1)

    prediction = model.predict(input_data)

    return {"prediction": int(prediction[0])}
