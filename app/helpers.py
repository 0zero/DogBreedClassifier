import numpy as np
from keras.preprocessing import image
from PIL.JpegImagePlugin import JpegImageFile


def path_to_tensor(image_input: JpegImageFile) -> np.array:
    """
    Take image and turn it into a tensor for use by classification models.

    :param image_input: image to be turned into a tensor
    :return: tensor appropriate for use by classifier models
    """
    # loads RGB image as PIL.Image.Image type
    # img = image.load_img(image_path, target_size=(224, 224))
    image_input = image_input.convert("RGB")

    img = image_input.resize((224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
