# DogBreedClassifier


Convolutional Neural Networks and Transfer Learning are used to classify dog breeds


### File descriptions
Below we have a brief description of the directories and the files they contain. 
- `app` <br>
  &nbsp;| - `template` <br>
  &nbsp;| --- `index.html`: main page of web app <br>
  &nbsp;| - `detector_models.py`: Detector model classes <br>
  &nbsp;| - `dog_classifier.py`: Dog breed classifier class <br>
  &nbsp;| - `helpers.py`: Helper functions <br>
  &nbsp;| - `run.py`: Flask file that runs app <br>
- `bin` <br>
  &nbsp;| - `disaster_categories.csv`: data to process <br>
  &nbsp;| - `disaster_messages.csv`: data to process <br>
  &nbsp;| - `process_data.py`: ETL script <br>
  &nbsp;| - `DisasterResponse.db`: database containing cleaned data <br>
- `data` <br>
  &nbsp;| - `train_classifier.py`: ML pipeline script <br>
  &nbsp;| - `classifier.pkl`: enhanced saved model with extra features <br>
  &nbsp;| - `classifierCV.pkl`: original saved model <br>
- `images` <br>
  &nbsp;| - `category_correlation.png`: plot of correlation between each category <br>
  &nbsp;| - `category_counts.png`: plot of counts in each category <br>
- `requirements` <br>
  &nbsp;| - `category_correlation.png`: plot of correlation between each category <br>
  &nbsp;| - `category_counts.png`: plot of counts in each category <br>
- `saved_models` <br>
  &nbsp;| - `category_correlation.png`: plot of correlation between each category <br>
  &nbsp;| - `category_counts.png`: plot of counts in each category <br>
- `test_images` <br>
  &nbsp;| - 7 `jpg/jpeg` files: image files used to testing final CNN model <br>
- `README.md`: readme file
- `.gitattributes`: contains files managed by git-lfs
- `.gitignore`: file/folders to ignore
- `environment_dogApp.yml`: anaconda python environment export
- `setup.cfg`: setup configs for flake8
- `dog_app.ipynb`: notebook used for exploration, model creation and training.


---

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.