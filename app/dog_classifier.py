from detector_models import DogBreedDetector, DogDetector, HumanFaceDetector
from helpers import show_image


class DogBreedClassifier:
    def __init__(
        self,
        dog_detector: DogDetector,
        face_detector: HumanFaceDetector,
        dog_breed_detector: DogBreedDetector,
    ) -> None:

        self.dog_detector_model: DogDetector = dog_detector
        self.face_detector_model: HumanFaceDetector = face_detector
        self.dog_breed_detector_model: DogBreedDetector = dog_breed_detector

    def classify_image(self, image_path: str) -> None:

        show_image(image_path)

        print("Evaluating image...")

        breed_prediction = self.dog_breed_detector_model.predict_dog_breed(image_path)

        if self.dog_detector_model.is_dog(image_path):
            msg = f"Looks like you've provided a picture of a {breed_prediction} dog."
        if self.face_detector_model.is_human_face(image_path):
            msg = (
                "Looks like you've provided a picture human that "
                f"looks like a {breed_prediction} dog."
            )
        else:
            msg = (
                "You haven't provided a dog or human picture but "
                f"whatever it is, it looks like a {breed_prediction} dog."
            )

        print(msg)
