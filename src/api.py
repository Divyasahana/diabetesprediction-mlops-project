from fastapi import FastAPI
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

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/v1/predict")
def predict(data: DiabetesInput):
    model = load_model()   # 👈 load model here (lazy loading)

    features = [
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree_function,
        data.age
    ]

    prediction = make_prediction(model, features)

    return {"prediction": int(prediction)}
