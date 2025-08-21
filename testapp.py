import streamlit as st
import keras
import numpy as np
import pandas as pd
from PIL import Image
import gdown
import os

# ---------------- CONFIG ----------------
KERAS_MODEL_FILE_ID = "1jGCwRGwO2bTbvZQS_3yodaeyLjdPx5JQ"  # Your model file ID
XLSX_FILE_ID = "1dJbbLx348xTBiOCh4ywW-qAcfNhqbrVO"         # Your Excel file ID

MODEL_PATH = "Plant_Village_Detection_Model.keras"
MAPPING_XLSX = "leaf_disease_responses.xlsx"
IMG_SIZE = (224, 224)

# ---------------- DOWNLOAD FILES ----------------
def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    download_from_drive(KERAS_MODEL_FILE_ID, MODEL_PATH)
    return keras.models.load_model(MODEL_PATH)

# ---------------- LOAD LABELS & RESPONSES ----------------
@st.cache_data
def load_mappings():
    download_from_drive(XLSX_FILE_ID, MAPPING_XLSX)
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
        img = image.resize(IMG_SIZE).convert("RGB")  # Ensure RGB and correct size
        img_array = np.asarray(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
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
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    with st.spinner("🔄 Loading model and mappings..."):
        model = load_model()
        label_map, treatment_map = load_mappings()

    img_array = preprocess_image(image)
    if img_array is not None:
        if img_array.shape[-1] != 3:
            st.error(f"Expected 3 channels (RGB), but got shape {img_array.shape}")
        else:
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
