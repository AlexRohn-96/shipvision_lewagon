from shipvision_backend.preprocessing import *
import json
import os
from shipvision_backend.params import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from shipvision_backend.modeling import *
from shipvision_backend.registry import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def transform_train():
    """ returns X_test_preproc, y_test with a shape (n_samples, 80, 80, 3) """
   #define the path to Json file
    json_path = os.path.expanduser(os.path.join(LOCAL_REGISTRY_PATH,'raw_data','shipsnet.json'))

    with open(json_path) as file:
        data= json.load(file)


    data_df = pd.DataFrame(data)



    #define the feature and target
    X= data['data']
    y=data_df['labels']



    #split the data into X_train, X_test, y_train, y_test
    X_train,X_test, y_train, y_test= train_test_split(X,
                                                      y,
                                                      random_state=43,
                                                    test_size=0.3,
                                                    stratify=y)

    #transform features into an array
    X_train_preproc= transform(X_train)

    X_test_preproc= transform(X_test)

    # Transform target into a dataframe




    # initializing the model
    model= initialize_model()

    #compiling the model
    model= compile_model(model)

    #training the model
    model, history = train_model(model,
        X_train_preproc,
        y_train,
        batch_size=32,
        patience=2,
        epochs=10,
        validation_data=None, # overrides validation_split
        validation_split=0.2)

    save_model(model)

    print('✅ Model train')

    return X_test_preproc, y_test




def evaluation(X_test_preproc, y_test):

    """ returns the accuracy of the model """
    #load the model
    model=load_model()

    #evaluate the model
    accuracy= evaluate_model (model,
                    X_test_preproc,
                    y_test,
                    )
    print('The accuracy is: ' , accuracy)

def pred( X_pred:list )-> int:
    """ Returns the prediction (ship or no ship) given a list of RGB pixels corresponding"""

#     image_array = np.array(X_pred)

# # Extract the R, G, and B channels
#     R_data = image_array[0:6400]
#     G_data = image_array[6400:2*6400]
#     B_data = image_array[2*6400:]

#     # Reshape each channel into an 80x80 array
#     R = R_data.reshape((80, 80))
#     G = G_data.reshape((80, 80))
#     B = B_data.reshape((80, 80))

#     # Stack the R, G, and B channels to form the image
#     image_rgb = np.stack((R, G, B), axis=-1)

#     print(image_rgb.shape)

#     # Plot the image
#     plt.imshow(image_rgb)
#     plt.axis('off')  # Optional: Hide axes
#     plt.show()


    X_pred_preproc= transform(X_pred)
    #load the model
    model= get_model_instance()

    #predict from an array and return 0 or 1
    y_pred = model.predict(X_pred_preproc)

    if y_pred[0] > 0.5:
        predicted_class = 1

    else:
        predicted_class = 0

    print('✅ Prediction :',  predicted_class)

    return predicted_class
