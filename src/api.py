# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import load_model, make_prediction

app = FastAPI(title="Diabetes Prediction API")

class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int

# Load model once at startup
model = load_model()
if model is None:
    print("Warning: Model did not load. Predictions will fail.")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/v1/predict")
def predict(data: DiabetesInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    features = data.dict()
    try:
        prediction = make_prediction(model, features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"prediction": prediction}