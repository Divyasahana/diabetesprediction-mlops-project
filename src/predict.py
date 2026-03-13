import joblib
import os

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")


def load_model():
    return joblib.load(MODEL_PATH)


def make_prediction(model, features):
    return model.predict([features])[0]
