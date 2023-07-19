import os
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(os.path.join("model", "toxmodel.keras"))
    return model


@st.cache_resource
def load_vectorizer():
    from_disk = pickle.load(open(os.path.join("model", "vectorizer.pkl"), "rb"))
    new_v = TextVectorization.from_config(from_disk['config'])
    new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"])) # Keras bug
    new_v.set_weights(from_disk['weights'])
    return new_v


@st.cache_resource
def load_vocab():
    vocab = {}
    with open('vocab.txt', 'r') as f:
        for line in f:
            token, index = line.strip().split('\t')
            vocab[token] = int(index)


st.title("Toxic Comment Test")
st.divider()
model = load_model()
vectorizer = load_vectorizer()
input_text = st.text_area("Comment:", "I love you man, but fuck you!", height=150)
if st.button("Test"):
    with st.spinner("Testing..."):
        inputv = vectorizer([input_text])
        output = model.predict(inputv)
        res = (output > 0.5)
    st.write(["toxic","severe toxic","obscene","threat","insult","identity hate"], res)


