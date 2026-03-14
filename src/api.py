from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import load_model, make_prediction

app = FastAPI(title="Diabetes Prediction API")

# Pydantic model with correct aliases
class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: int
    blood_pressure: int = Field(..., alias="blood_pressure")
    skin_thickness: int = Field(..., alias="skin_thickness")
    insulin: int
    bmi: float
    diabetes_pedigree_function: float = Field(..., alias="diabetes_pedigree_function")
    age: int

    class Config:
        allow_population_by_field_name = True

# Load MLflow model once at startup
model = load_model()
if model is None:
    print("Warning: Model did not load. Predictions will fail.")

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/v1/predict")
def predict(data: DiabetesInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Use by_alias=True so MLflow gets correct column names
    features = data.dict(by_alias=True)

    try:
        prediction = make_prediction(model, features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"prediction": prediction}