***Note: Due to the nature of toxic comments please cosider this project as explicit.***

# Toxic Comment Test
## Python script to train TF model and a streamlit app to use the model.

Data is from kaggle, the *Toxic Comment Classification Challenge*
<br>
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/
<br>
Please downlad the data from this link and put `train.csv` into the `data` dir:
<br>
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data?select=train.csv.zip

To use pretrained model, please donload it from HuggingFace link:
<br>
https://huggingface.co/vluz/toxmodel/tree/main

Put `toxmodel.keras` and `vectorizer.pkl` into the `model` dir.

<hr>

Open a command prompt and `cd` to a new directory of your choosing:

Create a virtual environment with:
```
python -m venv "venv"
venv\Scripts\activate
```

To install do:
```
git clone https://github.com/vluz/ToxTest.git
cd ToxTest
pip install -r requirements.txt
```
Put `train.csv` into the `data` dir and/or     
Put `toxmodel.keras` and `vectorizer.pkl` into the `model` dir.     
<br>

To train do:<br>
```
python toxtrain.py
``` 

To test using existing model do:
```
stramlit run toxtest.py
```

To exit the virtual environment do:
```
venv\Scripts\deactivate
```

<hr>
