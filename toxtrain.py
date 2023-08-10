import os
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras import mixed_precision


mixed_precision.set_global_policy('mixed_float16')
MAX_FEATURES = 200000
df = pd.read_csv(os.path.join("data", "train.csv"))
print("\n\nData loaded.\n")
X = df["comment_text"]
y = df[df.columns[2:]].values
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode="int")
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)
train = dataset.take(int(len(dataset)))
val = dataset.skip(int(len(dataset)*.8)).take(int(len(dataset)*.2))
model = Sequential()
model.add(Embedding(MAX_FEATURES+1, 32))
model.add(Bidirectional(LSTM(32, activation="tanh")))
model.add(Dense(128, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(6, activation="sigmoid"))
model.compile(loss="BinaryCrossentropy", optimizer="Adam")
print("\nModel:")
model.summary()
print("\nTraining...")
model.fit(train, epochs=30, validation_data=val)
print("\nSaving...")
pickle.dump({'config': vectorizer.get_config(),
             'weights': vectorizer.get_weights()}
            , open(os.path.join("model", "vectorizer.pkl"), "wb"))
with open(os.path.join("model", "vocab.txt"), 'w', encoding="utf-8") as f:
    for index, token in enumerate(vectorizer.get_vocabulary()):
        f.write('{}\t{}\n'.format(token, index))
model.save(os.path.join("model", "toxmodel.keras"))
print("\nAll Done.")
