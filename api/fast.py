from fastapi import FastAPI
from shipvision_backend.preprocessing import *
from shipvision_backend.registry import *
from shipvision_backend.main import *
from shipvision_backend.params import *

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}

@app.get("/predict")
def predict(X: list):

    """
    With this current version of predict, the input is a list of integers X containing the RGB pixel values of an image.
    Based on that, we call the pred function from main.py to load the most recent model, preprocess X, and predict whether X is a ship or not.
    """



    y_pred = pred(X)

    return {'Prediction': int(y_pred)}
