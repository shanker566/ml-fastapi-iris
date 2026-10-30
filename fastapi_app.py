
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import pickle
import numpy as np
from pathlib import Path

app = FastAPI(
    title="Iris ML Prediction API",
    description="Machine Learning API using FastAPI",
    version="1.0"
)

# Load trained model
model_path = Path(__file__).parent / "model.pkl"

with open(model_path, "rb") as file:
    saved_data = pickle.load(file)

model = saved_data["model"]
target_names = saved_data["target_names"]


# Input validation
class PredictionInput(BaseModel):
    features: list[float] = Field(min_length=4, max_length=4)

    @field_validator("features", mode="before")
    @classmethod
    def validate_features(cls, values):
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError("Exactly 4 features are required")

        for value in values:
            if isinstance(value, bool):
                raise ValueError("Boolean values are not allowed")

            if not isinstance(value, (int, float)):
                raise ValueError("Features must be numeric")

            if not np.isfinite(value):
                raise ValueError("Features must be finite numbers")

        return values


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "FastAPI ML Prediction API is running!"
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):

    try:
        features = np.array(data.features).reshape(1, -1)

        prediction = int(model.predict(features)[0])

        return {
            "prediction": prediction,
            "class_name": str(target_names[prediction]),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Run using Uvicorn from terminal
