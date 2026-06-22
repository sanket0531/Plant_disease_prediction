from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import joblib

model = joblib.load("crop_disease_model.pkl")

class CropInput(BaseModel):
    Crop_Type : str
    Soil_Type : str
    Temperature: int
    Irrigation_Method : str
    Disease_Description : str

app = FastAPI()

@app.get("/")
def home():
    return {"Message":"crop disease API is working"}

@app.post("/predict")
def predict(data: CropInput):
    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_disease": prediction
    }

