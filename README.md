***Note: Due to the nature of toxic comments please cosider this project as explicit.***

# Toxic Comment Test
## Python script to train a classification TensorFlow model, and a streamlit app to use the model.

Data is from kaggle, the *Toxic Comment Classification Challenge*
<br>
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/
<br>
Original data:
<br>
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data?select=train.csv.zip

<br>

**Demo running instance:** https://huggingface.co/spaces/vluz/Tox

<hr>

To use pretrained model, please donload `toxmodel.keras` and `vectorizer.pkl` from HuggingFace link:
<br>
https://huggingface.co/vluz/toxmodel30/tree/main/model     

<br>

To download the cleaned up data please go here:
<br>
https://huggingface.co/datasets/vluz/Tox/blob/main/alt_format/train.csv

<hr>

Open a command prompt and `cd` to a new directory of your choosing.

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
<br>

Put `train.csv` into the `data` dir      
*and/or*     
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

The helper script `dataclean.py` provides text cleaning for original data

The helper script `renderwordcloud.py` renders wordclouds for both the data as a whole, and toxic comments

<hr>
