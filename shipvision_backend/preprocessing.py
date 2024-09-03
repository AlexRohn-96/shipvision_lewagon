import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

class image_reshape_normalize((BaseEstimator, TransformerMixin)):
    #create a class that inherits from BaseEstimator and TransformerMixin classes from scikit-learn.
    def __init__(self,normalize= True):# Initialize with a parameter to control normalization
        """
        Initialize the transformer with optional parameters.

        Parameters:
        - normalize (bool): Whether to normalize the image pixel values to the range [0, 1].
                            Default is True.
        """
        self.normalize=normalize
    def fit(self, X,y=None):
        """Fit method is required for the transformer but doesn't do anything in this case.
        Purpose:
        - This method is necessary to comply with scikit-learn's transformer interface.
        - Even if no fitting is required, this method must return self to allow chaining
          in pipelines.

        Parameters:
        - X (array-like): Input data (ignored in this method).
        - y (array-like): Target data (optional, ignored in this method).

        Returns:
        - self: Returns the instance itself, allowing method chaining."""
        return (self)

    def transform(self,X):

        """  Apply the transformation to the input data.
        return a A NumPy array of reshaped and optionally normalized images,
        with shape (n_samples, 80, 80, 3)"""
        reshaped_images = []

        for image in X:
            image_array = np.array(image) / 255.0  # Normalize the image

            R_data = image_array[0:6400].reshape((80, 80))
            G_data = image_array[6400:2*6400].reshape((80, 80))
            B_data = image_array[2*6400:].reshape((80, 80))

            image_rgb = np.stack((R_data, G_data, B_data), axis=-1)
            reshaped_images.append(image_rgb)

        return np.array(reshaped_images)
