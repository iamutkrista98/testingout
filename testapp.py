import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import gdown
import os
from PIL import Image

# ---------------- CONFIG ----------------
GOOGLE_DRIVE_FILE_ID = "1yn35ZX_h8wiyfsnqSvTmdkIUrA5J5DYK"  # your original h5 file id
MODEL_PATH = "Plant_Village_Detection_Model.h5"
MAPPING_XLSX = "leaf_disease_responses.xlsx"
IMG_SIZE = (224, 224)

# ---------------- DOWNLOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    # ✅ Keras 3 can still load .h5
    model = keras.models.load_model(MODEL_PATH, compile=False)
    return model

# ---------------- LOAD RESPONSES ----------------
@st.cache_data
def load_responses():
    if os.path.exists(MAPPING_XLSX):
        df = pd.read_excel(MAPPING_XLSX)
        mapping = dict(zip(df["Label"], df["Treatment"]))
    else:
        mapping = {}
    return mapping

# ---------------- PREPROCESS IMAGE ----------------
def preprocess_image(image):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# ---------------- STREAMLIT APP ----------------
st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload a plant leaf image to detect disease, confidence, and suggested treatment.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    with st.spinner("Loading model..."):
        model = load_model()
        responses = load_responses()

    # Prediction
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    predicted_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]) * 100)

    predicted_label = str(predicted_idx)

    st.subheader("Prediction Results")
    st.write(f"**Predicted Disease:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    treatment = responses.get(predicted_label, "No treatment information available.")
    st.subheader("Treatment Recommendation")
    st.write(treatment)
