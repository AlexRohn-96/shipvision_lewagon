import numpy as np


def transform(X,y=None):
        """  Apply the transformation to the input data.
        return a A NumPy array of reshaped and optionally normalized images,
        with shape (n_samples, 80, 80, 3)
        Parameters:
    -----------
    X : list or NumPy array
        Input data consisting of flattened RGB images. Each image should be represented as
        a 1D array of 19200 elements, with 6400 elements for each color channel (R, G, B).

    y : None or array, optional
    Returns:
    --------
    np.ndarray
        A NumPy array of reshaped and normalized images with shape (n_samples, 80, 80, 3).
        The images are normalized to a [0, 1] range by dividing pixel values by 255.0.
    """
        reshaped_images = []



        element_1 = X[0]

        if isinstance(element_1, list):

            for image in X:
                image_array = np.array(image) / 255.0  # Normalize the image

                R_data = image_array[0:6400].reshape((80, 80))
                G_data = image_array[6400:2*6400].reshape((80, 80))
                B_data = image_array[2*6400:].reshape((80, 80))
                image_rgb = np.stack((R_data, G_data, B_data), axis=-1)
                reshaped_images.append(image_rgb)
            return np.array(reshaped_images)

        else:


            image_array = np.array(X) / 255.0  # Normalize the image

            R_data = image_array[0:6400].reshape((80, 80))
            G_data = image_array[6400:2*6400].reshape((80, 80))
            B_data = image_array[2*6400:].reshape((80, 80))
            image_rgb = np.stack((R_data, G_data, B_data), axis=-1)
            image_rgb = image_rgb.reshape((1, 80, 80, 3))



            return image_rgb



# Function to convert RGB to grayscale, reshape, and normalize
def rgb_to_grayscale(X):
    """  Apply the transformation to the input data.
        return a A NumPy array of reshaped and optionally normalized images,
        with shape (n_samples, 80, 80, 1)
        Parameters:
    -----------
    X : list or NumPy array
        The input data consisting of flattened RGB images. Each image is expected to be
        flattened into a 1D array with 3 consecutive sections corresponding to the red (R),
        green (G), and blue (B) color channels.

    Returns:
    --------
    np.ndarray
        A NumPy array of reshaped and normalized grayscale images with shape (n_samples, 80, 80, 1).
        The grayscale images are created using the luminosity method, where each pixel is
        computed as: 0.2989 * R + 0.5870 * G + 0.1140 * B.
        """

    X_gray = []

    element_1 = X[0]

    if isinstance(element_1, list):

        for img in X:
            # Split the flattened list into R, G, and B channels
            R = np.array(img[:6400])
            G = np.array(img[6400:12800])
            B = np.array(img[12800:])

            # Convert to grayscale using the luminosity method: 0.2989*R + 0.5870*G + 0.1140*B
            gray_img = 0.2989 * R + 0.5870 * G + 0.1140 * B


            # Reshape to (80, 80, 1)
            gray_img = gray_img.reshape(80, 80, 1)

            X_gray.append(gray_img)



    else:
         # Split the flattened list into R, G, and B channels
            R = np.array(X[:6400])
            G = np.array(X[6400:12800])
            B = np.array(X[12800:])

            # Convert to grayscale using the luminosity method: 0.2989*R + 0.5870*G + 0.1140*B
            gray_img = 0.2989 * R + 0.5870 * G + 0.1140 * B


            # Reshape to (80, 80, 1)
            gray_img = gray_img.reshape(80, 80, 1)

            X_gray.append(gray_img)

    return np.array(X_gray)
