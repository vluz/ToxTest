import os
import re
import pandas as pd
import warnings


def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\'\s]', ' ', text)
    text = re.sub(r'(\s)([iI][eE]|[eE][gG])(\s)', r' \2 ', text)
    text = " ".join(text.split())
    return text.lower()


warnings.simplefilter("ignore", category=Warning)
print("Read")
df = pd.read_csv(os.path.join("data", "train.csv"))
df = df.reset_index()
print("Convert")
for ind in df.index:
    df["comment_text"][ind]='\"'+clean_text(df["comment_text"][ind])+'\"'
    df["id"][ind]='\"'+df["id"][ind]+'\"'
    print("Line: "+ str(ind)+"/"+str(len(df.index)-1), end='\r')
del df["index"]
print("Write")
df.to_csv(os.path.join("data", "train_T.csv"), index=False, encoding='utf-8', header=True)
print("Done.")