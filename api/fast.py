from fastapi import FastAPI
from pydantic import BaseModel
from shipvision_backend.preprocessing import *
from shipvision_backend.registry import *
from shipvision_backend.main import *
from shipvision_backend.params import *
app = FastAPI()
# Define the Pydantic model for request validation
class PredictRequest(BaseModel):
    X: list[int]

@app.get("/")
def index():
    return {"ok": True}

@app.post("/predict")
def predict(request: PredictRequest):
    # “”"
    # This endpoint receives a list of integers (pixels) and returns a prediction.
    # “”"
    # Call the prediction function



    y_pred = pred(request.X)


    return {"Prediction": int(y_pred)}
