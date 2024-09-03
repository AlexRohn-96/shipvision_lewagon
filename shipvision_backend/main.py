from shipvision_backend.preprocessing import *
import json
import os
from shipvision_backend.params import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from shipvision_backend.modeling import *
from shipvision_backend.registry import *

def transform_train():
    """ returns X_test_preproc, y_test with a shape (n_samples, 80, 80, 3) """
   #define the path to Json file
    json_path = os.path.expanduser(os.path.join(LOCAL_REGISTRY_PATH,'raw_data','shipsnet.json'))

    with open(json_path) as file:
        data= json.load(file)

    #define the feature and target
    X= data['data']
    y=data['labels']

    #split the data into X_train, X_test, y_train, y_test
    X_train,X_test, y_train, y_test= train_test_split(X,
                                                      y,
                                                      random_state=43,
                                                    test_size=0.3,
                                                    stratify=y)

    #transform features into an array
    X_train_preproc= transform(X_train)

    X_test_preproc= transform(X_test)

    # Transform target into array
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)



    # initializing the model
    model= initialize_model(input_shape= (80,80,3))

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
    X_pred_preproc= transform(X_pred)
    #load the model
    model= load_model()

    #predict from an array and return 0 or 1
    y_pred= model.predict(X_pred_preproc)

    predicted_class = np.argmax(y_pred, axis=1)


    print('✅ Prediction :',  predicted_class[0])

    return predicted_class[0]
