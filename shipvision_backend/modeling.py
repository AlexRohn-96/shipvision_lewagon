import numpy as np
import time
from colorama import Fore, Style
from typing import Tuple
from shipvision_backend.params import *


# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")





def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the CNN with random weights
    """
    model = Sequential()

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(8, (4,4), input_shape=input_shape, padding='same', activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(16, (3,3), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    ### Flattening
    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(10, activation='relu'))

    ### Last layer - Classification Layer with 2 outputs
    model.add(layers.Dense(2, activation='sigmoid'))

    print("✅ Model initialized")

    return model





def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the CNN
    """

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    print("✅ Model compiled")

    return model



def train_model(
        model: Model,
        X,
        y,
        batch_size=32,
        patience=2,
        epochs=10,
        validation_data=None, # overrides validation_split
        validation_split=0.2
    ) -> Tuple[Model, dict]:

    """
    Fit the model and return a tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
    )

    print(f"✅ Model trained on {len(X)} rows with min val accuracy: {round(np.min(history.history['val_accuracy']), 2)}")

    return model, history




def evaluate_model(
        model: Model,
        X_test,
        y_test_cat,
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """


    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X_test,
        y=y_test_cat,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, accuracy: {round(accuracy, 2)}")

    return metrics
