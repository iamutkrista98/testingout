import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import gdown
import os
import zipfile
from PIL import Image

# ---------------- CONFIG ----------------
GOOGLE_DRIVE_FILE_ID = "1IgXN5-6nzWmI0XTTGqxcNpYaBgN8o89W"
MODEL_ZIP_PATH = "plant_disease_model.zip"
MODEL_DIR = "plant_disease_model"
MAPPING_XLSX = "leaf_disease_responses.xlsx"
IMG_SIZE = (224, 224)  # adjust if your model used a different input size

# ---------------- DOWNLOAD & EXTRACT MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        # Download from Google Drive
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_ZIP_PATH, quiet=False)

        # Extract zip
        with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(".")

    # ✅ Use TFSMLayer instead of load_model (Keras 3 fix)
    model = keras.layers.TFSMLayer(MODEL_DIR, call_endpoint="serving_default")
    return model

# ---------------- LOAD TREATMENT RESPONSES ----------------
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
    if len(img_array.shape) == 2:  # grayscale
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

    # Preprocess & Predict
    img_array = preprocess_image(image)

    # ✅ Call TFSMLayer directly
    preds_dict = model(img_array)
    preds = preds_dict["predictions"].numpy()  # "predictions" usually the output key

    predicted_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]) * 100)

    # Label mapping
    predicted_label = str(predicted_idx)

    # Show results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Disease:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Treatment suggestion
    treatment = responses.get(predicted_label, "No treatment information available.")
    st.subheader("Treatment Recommendation")
    st.write(treatment)
