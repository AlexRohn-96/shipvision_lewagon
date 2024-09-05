from tensorflow import keras
import glob
import os
import time
import pickle
from shipvision_backend.params import *
from colorama import Fore, Style




def save_model(model: keras.Model = None) -> None:
    """
    Save trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")



    # Save model locally
    model_path = os.path.expanduser(os.path.join(LOCAL_REGISTRY_PATH,'models',f"{timestamp}.h5"))
    model.save(model_path)

    print("✅ Model saved locally")






def load_model() -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)

    Return None (but do not Raise) if no model is found

    """
    # logging.basicConfig(filename='app.log', level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
    # logging.debug("starrting loading model")
    # Get the latest model version name by the timestamp on disk
    local_model_directory = os.path.expanduser(os.path.join(LOCAL_REGISTRY_PATH,'models'))
    print(local_model_directory)
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        model_path = "/models/20240904-161447.h5"
        model = keras.models.load_model(model_path)
        return model

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print(f"Trying to load model from: {most_recent_model_path_on_disk}")

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)


    latest_model = keras.models.load_model(most_recent_model_path_on_disk)

    print("✅ Model loaded from local disk")

    return latest_model
