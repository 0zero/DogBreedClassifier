import cv2
import json
import numpy as np

from cv2 import CascadeClassifier, cvtColor
from helpers import path_to_tensor
from keras.applications.resnet50 import ResNet50, preprocess_input as RNpreprocess
from keras.applications.xception import Xception, preprocess_input as Xpreprocess
from keras.models import load_model
from typing import List
from PIL.JpegImagePlugin import JpegImageFile


class DogDetector:
    @property
    def model(self) -> ResNet50:
        """
        ResNet50 classification model to be used to detect dogs

        :return: ResNet50 model
        """
        return ResNet50(weights="imagenet")

    def is_dog(self, input_image: JpegImageFile) -> bool:
        """
        Predcits whether input image is of a dog or not.

        :param input_image: input image to classify
        :return: True if image is predicted to be of a dog
        """
        img = RNpreprocess(path_to_tensor(input_image))
        prediction = np.argmax(self.model.predict(img))
        return (prediction <= 268) & (prediction >= 151)


class HumanFaceDetector:
    @property
    def model(self) -> CascadeClassifier:
        """
        Human face detector model

        :return: Human face classification model
        """
        return CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")

    def is_human_face(self, input_image: JpegImageFile) -> bool:
        """
        Predcits whether input image is of a human face or not.

        :param input_image: input image to classify
        :return: True if image is predicted to be of a human face
        """
        prediction = self.model.detectMultiScale(
            cvtColor(np.asarray(input_image), cv2.COLOR_BGR2GRAY)
        )
        return len(prediction) > 0


class DogBreedDetector:
    @property
    def original_model(self) -> Xception:
        """
        Xception classification model

        :return: Xception model
        """
        return Xception(weights="imagenet", include_top=False)

    @property
    def model(self) -> Xception:
        """
        Re-trained Xception model to be used to classify dog breeds

        :return: Xception model
        """
        return load_model("saved_models/dogBreedXception.h5")

    @property
    def dog_names(self) -> List[str]:
        """
        Reads a list of dog names from file

        :return: list of dog breed names
        """
        with open("data/dog_names.json", "r") as f:
            names = json.load(f)
        return names

    def predict_dog_breed(self, input_image: JpegImageFile) -> str:
        """
        Predict breed of dog in the input image

        :return: predicted dog breed
        """
        features = self.original_model.predict(Xpreprocess(path_to_tensor(input_image)))
        predicted_vector = self.model.predict(features)
        dog_breed = " ".join(
            self.dog_names[np.argmax(predicted_vector)].split(".")[-1].split("_")
        )
        return dog_breed
