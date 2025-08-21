import streamlit as st
import keras
import numpy as np
import pandas as pd
from PIL import Image
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "Plant_Village_Detection_Model.h5"
MAPPING_XLSX = "leaf_disease_responses.xlsx"
IMG_SIZE = (224, 224)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH, compile=False)

# ---------------- LOAD LABELS & RESPONSES ----------------
@st.cache_data
def load_mappings():
    try:
        df = pd.read_excel(MAPPING_XLSX)
        label_map = dict(zip(df["class_index"], df["disease_name"]))
        treatment_map = dict(zip(df["disease_name"], df["response_message"]))
        return label_map, treatment_map
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return {}, {}

# ---------------- PREPROCESS IMAGE ----------------
def preprocess_image(image):
    try:
        img = image.convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        return None

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload a leaf image to detect disease, confidence, and treatment advice.")

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    with st.spinner("🔄 Loading model and mappings..."):
        model = load_model()
        label_map, treatment_map = load_mappings()

    img_array = preprocess_image(image)
    if img_array is not None:
        preds = model.predict(img_array)
        predicted_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]) * 100)

        predicted_disease = label_map.get(predicted_idx, f"Unknown class {predicted_idx}")
        treatment = treatment_map.get(predicted_disease, "No treatment information available.")

        st.subheader("🔍 Prediction Results")
        st.markdown(f"**🦠 Disease:** `{predicted_disease}`")
        st.markdown(f"**📊 Confidence:** `{confidence:.2f}%`")

        st.subheader("💊 Treatment Recommendation")
        st.info(treatment)
    else:
        st.error("❌ Could not process the image. Please try a different one.")
else:
    st.warning("Please upload a leaf image to begin diagnosis.")
