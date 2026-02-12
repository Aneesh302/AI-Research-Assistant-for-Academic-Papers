import streamlit as st
import re
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle
import numpy as np


st.title("Academic Paper Abstract Category Classification")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\$.*?\$", " ", text)  # remove latex math
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

model = tf.keras.models.load_model('/home/sunbeam/STUDY_NEW/PROJECT/src/Modelling/models/arxiv_bilstm_model.keras',compile=False)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder_category.pkl", "rb") as f:
    label_encoder = pickle.load(f)


MAX_LEN = 150

text = st.text_area("Enter abstract")

if st.button("Classify"):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    preds = model.predict(padded)

    class_idx = np.argmax(preds, axis=1)[0]
    category = label_encoder.inverse_transform([class_idx])[0]
    confidence = preds[0][class_idx]
    st.success(f"Predicted Category: **{category}**")
    st.write(f"Confidence: {confidence:.2%}")
