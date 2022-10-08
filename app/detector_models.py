import cv2
import json
import numpy as np

from cv2 import CascadeClassifier, imread, cvtColor
from helpers import path_to_tensor
from keras.applications import ResNet50, Xception
from keras.applications import imagenet_utils
from keras.models import load_model
from typing import List


class DogDetector:
    @property
    def model(self) -> ResNet50:
        return ResNet50(weights="imagenet")

    def is_dog(self, input_image: str) -> bool:
        img = imagenet_utils.preprocess_input(path_to_tensor(input_image))
        prediction = np.argmax(self.model.predict(img))
        return (prediction <= 268) & (prediction >= 151)


class HumanFaceDetector:
    @property
    def model(self) -> CascadeClassifier:
        # TODO: fix path to haarcascade model
        return CascadeClassifier("haarcascade_frontalface_alt.xml")

    def is_human_face(self, input_image: str) -> bool:
        img = imread(input_image)
        prediction = self.model.detectMultiScale(cvtColor(img, cv2.COLOR_BGR2GRAY))
        return len(prediction) > 0


class DogBreedDetector:
    @property
    def original_model(self) -> Xception:
        return Xception(weights="imagenet", include_top=False)

    @property
    def model(self) -> Xception:
        return load_model("saved_models/dogBreedXception.h5")

    @property
    def dog_names(self) -> List[str]:
        with open("data/dog_names.json", "r") as f:
            names = json.load(f)
        return names

    def predict_dog_breed(self, input_image: str) -> str:
        features = self.original_model.predict(
            imagenet_utils.preprocess_input(path_to_tensor(input_image))
        )
        predicted_vector = self.model.predict(features)
        dog_breed = " ".join(
            self.dog_names[np.argmax(predicted_vector)].split(".")[-1].split("_")
        )
        return dog_breed
