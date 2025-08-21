import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import keras

# ---------------- CONFIG ----------------
MODEL_PATH = "clean_model1.keras"
MAPPING_XLSX = "leaf_disease_responses.xlsx"
IMG_SIZE = (224, 224)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ---------------- LOAD LABELS & RESPONSES ----------------
@st.cache_data
def load_mappings():
    df = pd.read_excel(MAPPING_XLSX)
    label_map = dict(zip(df["class_index"], df["disease_name"]))
    treatment_map = dict(zip(df["disease_name"], df["response_message"]))
    return label_map, treatment_map

label_map, treatment_map = load_mappings()

# ---------------- PREPROCESS IMAGE ----------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Leaf Disease Detector", layout="centered")
st.title("🌿 Plant Leaf Disease Classifier")
st.markdown("Upload a leaf image to detect disease and get treatment advice.")

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("Analyzing image..."):
        img_array, display_img = preprocess_image(uploaded_file)
        preds = model.predict(img_array)
        predicted_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]) * 100)

        predicted_disease = label_map.get(predicted_idx, f"Unknown class {predicted_idx}")
        treatment = treatment_map.get(predicted_disease, "No treatment information available.")

    # ---------------- DISPLAY RESULTS ----------------
    st.image(display_img, caption="Uploaded Leaf", use_column_width=True)
    st.subheader("🔍 Prediction Results")
    st.markdown(f"**Disease:** {predicted_disease}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    st.subheader("💊 Treatment Recommendation")
    st.markdown(treatment)

    st.success("Diagnosis complete. Follow the treatment plan above.")
else:
    st.info("Please upload a leaf image to begin.")
