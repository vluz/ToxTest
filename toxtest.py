import os
import re
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization


def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\'\s]', ' ', text)
    text = re.sub(r'(\s)([iI][eE]|[eE][gG])(\s)', r' \2 ', text)
    text = " ".join(text.split())
    return text.lower()


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(os.path.join("model", "toxmodel.keras"))
    return model


@st.cache_resource
def load_vectorizer():
    from_disk = pickle.load(open(os.path.join("model", "vectorizer.pkl"), "rb"))
    new_v = TextVectorization.from_config(from_disk['config'])
    new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"])) # fix for Keras bug
    new_v.set_weights(from_disk['weights'])
    return new_v


st.title("Toxic Comment Test")
st.divider()
model = load_model()
vectorizer = load_vectorizer()
default_prompt = "I love you man, but fuck you!"
input_text = st.text_area("Comment:", default_prompt, height=150).lower()
if st.button("Test"):
    if not input_text:
        st.write("⚠ Warning: Empty prompt.")
    elif len(input_text) < 15:
        st.write("⚠ Warning: Model is far less accurate with a small prompt.")
    if input_text == default_prompt:
        st.write("Expected results from default prompt are positive for 0 and 2")
    with st.spinner("Testing..."):
        clean_input_text = clean_text(input_text)
        inputv = vectorizer([clean_input_text])
        output = model.predict(inputv)
        res = (output > 0.5)
    st.write(["toxic","severe toxic","obscene","threat","insult","identity hate"], res)
    st.write(output)
