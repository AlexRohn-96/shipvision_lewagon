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
    """ returns X_test_preproc, y_test with a shape (n_samples, 80, 80, 3) or (n_samples, 80, 80, 1) """
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

    # Transform X_train and X_test into (80, 80, 3) array
    #X_train_preproc= transform(X_train)
    #X_test_preproc= transform(X_test)



    # Transform X_train and X_test into (n, 80, 80, 1) array
    X_train_preproc = rgb_to_grayscale(X_train)
    X_test_preproc = rgb_to_grayscale(X_test)





    # Initializing the model - V1
    #model= initialize_model_1()

    # Initializing the model - V2
    model= initialize_model_2()

    # Compiling the model - V1
    #model= compile_model_1(model)

    # Compiling the model - V2
    model= compile_model_2(model)


    #training the model
    model, history = train_model(model,
        X_train_preproc,
        y_train,
        batch_size=32,
        patience=2,
        epochs=15,
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
    accuracy, recall, precision = evaluate_model (model,
                    X_test_preproc,
                    y_test,
                    )
    print('The accuracy is: ' , accuracy)
    print('The recall is: ' , recall)
    print('The precision is: ' , precision)



def pred( X_pred:list )-> int:
    """ Returns the prediction (ship or no ship) given a list of RGB pixels corresponding"""

    # Transform to (1, 80, 80, 1) array
    X_pred_preproc= rgb_to_grayscale(X_pred)

    # Transform to (1, 80, 80, 3) array
    #X_pred_preproc = transform(X_pred)

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
