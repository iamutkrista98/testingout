import streamlit as st
import keras
import numpy as np
import pandas as pd
from PIL import Image
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "clean_model1.keras"
MAPPING_XLSX = "leaf_disease_responses.xlsx"
IMG_SIZE = (224, 224)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        return None
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
        st.error(f"❌ Error loading Excel: {e}")
        return {}, {}

# ---------------- PREPROCESS IMAGE ----------------
def preprocess_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.asarray(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")
st.title("🌿 Plant Leaf Disease Detection")
st.markdown("Upload a plant leaf image to detect disease and get treatment advice.")

uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    with st.spinner("🔄 Loading model and treatment data..."):
        model = load_model()
        label_map, treatment_map = load_mappings()

    if model:
        img_array = preprocess_image(image)

        # Confirm shape before prediction
        if img_array.shape != (1, 224, 224, 3):
            st.error(f"❌ Image shape mismatch: {img_array.shape}")
        else:
            preds = model.predict(img_array)
            predicted_idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]) * 100)

            predicted_disease = label_map.get(predicted_idx, f"Unknown class {predicted_idx}")
            treatment = treatment_map.get(predicted_disease, "No treatment information available.")

            # Show results
            st.subheader("🧪 Prediction Results")
            st.write(f"**Predicted Disease:** {predicted_disease}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            st.subheader("💊 Treatment Recommendation")
            st.write(treatment)
