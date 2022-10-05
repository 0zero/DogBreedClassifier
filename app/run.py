import json

from flask import Flask
from flask import render_template, request, jsonify


app = Flask(__name__)

# load data


# load model


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    pass


# web page that handles user query and displays model results
@app.route("/go")
def go():
    pass


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()