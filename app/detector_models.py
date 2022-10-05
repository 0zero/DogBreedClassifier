import cv2
import numpy as np

from cv2 import CascadeClassifier, imread, cvtColor
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing import image


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
    

class DogDetector:

    @property
    def model(self):
        return ResNet50(weights="imagenet")

    def is_dog(self, input_image):
        img = imagenet_utils.preprocess_input(path_to_tensor(input_image))
        prediction = np.argmax(self.model.predict(img))
        return ((prediction <= 268) & (prediction >= 151)) 


class HumanFaceDetector:

    @property
    def model(self):
        # TODO: fix path to haarcascade model
        return CascadeClassifier("haarcascade_frontalface_alt.xml")

    def is_human_face(self, input_image):
        img = imread(input_image)
        prediction = self.model.detectMultiScale(
            cvtColor(img, cv2.COLOR_BGR2GRAY)
        )
        return len(prediction) > 0

