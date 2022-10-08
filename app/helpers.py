import numpy as np
from keras.preprocessing import image


def path_to_tensor(image_path: str) -> np.array:
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(image_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def show_image(image_path: str) -> None:
    pass
