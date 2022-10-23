import base64
import io
from pathlib import Path
from flask import Flask
from flask import render_template, request
from PIL import Image

from dog_classifier import DogBreedClassifier
from detector_models import DogDetector, DogBreedDetector, HumanFaceDetector

ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg"]

app = Flask(__name__)

# TODO: cache these models because something seems to take ages to get going

# load model
dog_detector = DogDetector()
face_detector = HumanFaceDetector()
dog_breed_detector = DogBreedDetector()
dog_classifier = DogBreedClassifier(
    dog_detector=dog_detector,
    face_detector=face_detector,
    dog_breed_detector=dog_breed_detector,
)


def check_image(req: request) -> bool:
    """
    Checks whether a request has files and whether they are
    of valid format

    :param req: request to validate
    :return: boolean showing whether files in request are valid
    """
    if "user-image" in req.files:
        if Path(req.files["user-image"].filename).suffix in ALLOWED_EXTENSIONS:
            return True
    return False


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.post("/getdoggy")
def getdoggy():
    """
    Get dog breed API

    :return: page and result to render
    """
    if check_image(request):
        user_image = request.files["user-image"].read()
        user_image = Image.open(io.BytesIO(user_image))

        prediction = dog_classifier.classify_image(user_image)
        display_image = io.BytesIO()
        user_image.save(display_image, "PNG", quality=100)
        display_image.seek(0)
        display_image = base64.b64encode(display_image.getvalue()).decode("ascii")

    return (
        render_template("index.html", data=[prediction, display_image])
        if prediction
        else render_template("index.html")
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
