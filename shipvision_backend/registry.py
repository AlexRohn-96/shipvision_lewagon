from tensorflow import keras
import glob
import os
import time
import pickle
from shipvision_backend.params import *
from colorama import Fore, Style
import logging
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get variables from .env
BUCKET_NAME = os.getenv('BUCKET_NAME')
model_instance = None  # This will hold the singleton model instance

# Set up logging to a file
logging.basicConfig(
    filename='shipvision_backend/app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_model_instance():
    """
    Loads the model from the cloud or local storage if it's not already loaded.
    Returns:
    --------
    model_instance : tensorflow.keras.Model
    The loaded model instance.
    """
    global model_instance
    if model_instance is None:
        logging.debug("Model instance is not loaded yet, loading now...")
        model_instance = load_model()
    else:
        logging.debug("Returning the already loaded model instance")
    return model_instance

def upload_to_gcs(local_path: str, bucket_name: str, destination_blob_name: str) -> None:
    """Uploads a file to Google Cloud Storage
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_path)

    print(f"✅ Model uploaded to GCS: {destination_blob_name}")
    logging.debug("starrting loading model")

def save_model(model: keras.Model = None) -> None:
    """
    Save trained model locally on the hard drive
    at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    """

    # Generate a timestamp for the filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")



    #construct the model save path
    model_path = os.path.expanduser(os.path.join(LOCAL_REGISTRY_PATH,'models',f"{timestamp}.h5"))
    # Save model locally
    model.save(model_path)

    print("✅ Model saved locally")

    #saving to the cloude
    gcs_model_path = os.path.join(BUCKET_NAME, f"{timestamp}.h5")

    # Upload the model to Google Cloud Storage
    upload_to_gcs(local_path=model_path, bucket_name=BUCKET_NAME, destination_blob_name=gcs_model_path)



def load_model() -> keras.Model:
    """
    Return a saved model:
    - Try loading from Google Cloud Storage (GCS)
    - If no model is found in GCS, load from the local file system (latest one in alphabetical order)

    Return None (but do not raise) if no model is found.
    """

    logging.debug("Starting model loading process...")

    # Try loading the model from Google Cloud Storage
    try:
        model = load_model_from_gcs()
        if model:
            print(Fore.GREEN + "✅ Model loaded from Google Cloud Storage" + Style.RESET_ALL)
            return model
    except Exception as e:
        logging.error(f"Failed to load model from GCS: {str(e)}")

    # Fallback to loading the model from local disk
    print(Fore.YELLOW + "⚠️ Loading model from local storage" + Style.RESET_ALL)
    model = load_model_from_local()

    if model:
        print("✅ Model loaded from local disk")
        return model
    else:
        print("❌ No model found in GCS or local storage")
        return None

def load_model_from_gcs() -> keras.Model:
    """
    Attempt to load the most recent model from Google Cloud Storage (GCS).
    Return the model if found, or None if no model is found in GCS.
    """

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    blobs = list(bucket.list_blobs(prefix=BUCKET_NAME))
    if not blobs:
        print("❌ No models found in GCS")
        return None

    # Sort blobs by name to get the latest model
    sorted_blobs = sorted(blobs, key=lambda x: x.name)
    latest_blob = sorted_blobs[-1]  # Get the latest model

    # Download the model to a temporary location
    model_file = f"/tmp/{latest_blob.name.split('/')[-1]}"
    latest_blob.download_to_filename(model_file)

    print(f"Trying to load model from GCS: {latest_blob.name}")
    return keras.models.load_model(model_file)

def load_model_from_local() -> keras.Model:
    """
    Attempt to load the most recent model from the local file system.
    Return the model if found, or None if no model is found locally.
    """
    local_model_directory = os.path.expanduser(os.path.join(LOCAL_REGISTRY_PATH, 'models'))

    logging.debug(f"Attempting to load model from local directory: {local_model_directory}")
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        print(f"❌ No models found in local directory: {local_model_directory}")
        return None

    # Sort models by name and load the most recent one
    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print(f"Trying to load model from local: {most_recent_model_path_on_disk}")
    return keras.models.load_model(most_recent_model_path_on_disk)
