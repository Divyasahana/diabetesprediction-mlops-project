from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import load_model, make_prediction

app = FastAPI(title="Diabetes Prediction API")

model = load_model()

class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: int
    bloodpressure: int
    skinthickness: int
    insulin: int
    bmi: float
    diabetespedigreefunction: float
    age: int


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: DiabetesInput):
    input_data = [[
        data.pregnancies,
        data.glucose,
        data.bloodpressure,
        data.skinthickness,
        data.insulin,
        data.bmi,
        data.diabetespedigreefunction,
        data.age
    ]]

    prediction = make_prediction(model, input_data[0])

    return {"prediction": int(prediction)}
