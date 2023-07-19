# Toxic Comment Test
## Python script to train TF model and a streamlit app to use the model.

Data is from kaggle, the *Toxic Comment Classification Challenge*
<br>
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/
<br>
Please downlad the data from this link and put `train.csv` into the `data` dir.
<br>
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data?select=train.csv.zip

To use pretrained model, please donload it from HuggingFace here:
<br>
https://huggingface.co/

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

To train do:<br>
```
python toxtrain.py
``` 

To test do:
```
stramlit run toxtest.py
```

To exit the virtual environment do:
```
venv\Scripts\deactivate
```

<hr>
