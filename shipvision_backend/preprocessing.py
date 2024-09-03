import numpy as np


def transform(X,y=None):
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
