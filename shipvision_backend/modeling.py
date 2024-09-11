import numpy as np
import time
from colorama import Fore, Style
from typing import Tuple
from shipvision_backend.params import *
from tensorflow.keras.metrics import Precision, Recall


# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def initialize_model_1():
    model = Sequential()


    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(8, (4,4), input_shape=(80, 80, 3), padding='same', activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(16, (3,3), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Third Convolution & MaxPooling
    model.add(layers.Conv2D(filters = 32, kernel_size = (3,3), activation="relu", padding = "same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = "same") )

    ### fourth Convolution & MaxPooling
    model.add(layers.Conv2D(filters = 64, kernel_size = (3,3), activation="relu", padding = "same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = "same") )

    ### Flattening
    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(10, activation='relu'))

    ### Last layer - Classification Layer with 2 outputs
    model.add(layers.Dense(1, activation='sigmoid'))

    print("✅ Model initialized")

    return model



def initialize_model_2():
    model = Sequential()
    model.add(layers.Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (80,80,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(layers.Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(layers.Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(layers.Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(layers.Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units = 128 , activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units = 1 , activation = 'sigmoid'))


    return model



def compile_model_1(model: Model) -> Model:
    """
    Compile the CNN
    """

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', Recall(), Precision()])

    print("✅ Model compiled")

    return model


def compile_model_2(model: Model) -> Model:
    """
    Compile the CNN
    """

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', Recall(), Precision()])

    print("✅ Model compiled")

    return model



def train_model(
        model: Model,
        X,
        y,
        batch_size=32,
        patience=2,
        epochs=20,
        validation_data=None, # overrides validation_split if provided
        validation_split=0.2
    ) -> Tuple[Model, dict]:

    """
    Fit the model and return a tuple (fitted_model, history)
    Returns:
    --------
    model : Model
        The trained Keras model.
    history : dict
        Training history, which includes loss and accuracy metrics for each epoch.
    """

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)
    # EarlyStopping callback to avoid overfitting
    es = EarlyStopping(
        patience=patience,
        restore_best_weights=True,
    )
    # Train the model
    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
    )
    # Logging the training completion with min val accuracy
    print(f"✅ Model trained on {len(X)} rows with min val accuracy: {round(np.min(history.history['val_accuracy']), 2)}")

    return model, history




def evaluate_model(
        model: Model,
        X_test,
        y_test_cat,
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    print the metrics: accuracy, precision and recall
    return A tuple containing the accuracy, recall,
    and precision metrics.
    """


    if model is None:
        print(f"\n❌ No model to evaluate")
        return None
    #evaluate the model
    metrics = model.evaluate(
        x=X_test,
        y=y_test_cat,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )
    # Extract relevant metrics
    loss = metrics["loss"]
    accuracy = metrics["accuracy"]
    recall = metrics["recall"]
    precision = metrics["precision"]

    #log and print the metrics
    print(f"✅ Model evaluated, accuracy: {round(accuracy, 2)}, recall: {round(recall, 2)}, precision: {round(precision, 2)}")

    return accuracy, recall, precision
