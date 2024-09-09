
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
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


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """
    This endpoint receives an image file from a web app and returns a prediction.
    """
    # Open the image using PIL
    image = Image.open(file.file)



    model= get_model_instance()

    print("Passed model loading")


    predictions ,coordinates= generate_scene_with_model_gray_scale(image,model)



    # Convert NumPy array of predictions to a list
    predictions_list = predictions.tolist()
    # Return both coordinates and predictions in the response
    return {
        'coordinates': coordinates,  # List of tuples (x, y)
        'predictions': predictions_list  # List of prediction values
    }



@app.post("/predict")
def predict(request: PredictRequest):
    # “”"
    # This endpoint receives a list of integers (pixels) and returns a prediction.
    # “”"
    # Call the prediction function

    y_pred = pred(request.X)



    return {"Prediction": int(y_pred)}
