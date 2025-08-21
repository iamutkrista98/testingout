import streamlit as st
import tensorflow as tf
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
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

# ---------------- LOAD TREATMENT RESPONSES ----------------
@st.cache_data
def load_responses():
    if not os.path.exists(MAPPING_XLSX):
        st.warning(f"⚠️ Mapping file not found: {MAPPING_XLSX}")
        return {}
    df = pd.read_excel(MAPPING_XLSX)
    return dict(zip(df["Label"], df["Treatment"]))

# ---------------- PREPROCESS IMAGE ----------------
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure 3 channels
    image = image.resize(IMG_SIZE)  # Resize to 224x224
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Debug output
    st.write(f"✅ Final image shape: {img_array.shape}")  # Should be (1, 224, 224, 3)
    st.write(f"✅ Final image dtype: {img_array.dtype}")  # Should be float32

    return img_array

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")
st.title("🌿 Plant Leaf Disease Detection")
st.markdown("Upload a plant leaf image to detect disease, confidence, and suggested treatment.")

uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    with st.spinner("🔄 Loading model and treatment data..."):
        model = load_model()
        responses = load_responses()

    if model:
        img_array = preprocess_image(image)

        # Confirm shape before prediction
        if img_array.shape != (1, 224, 224, 3):
            st.error(f"❌ Image shape mismatch: {img_array.shape}")
        else:
            preds = model.predict(img_array)
            predicted_idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]) * 100)

            # Handle class labels
            predicted_label = str(predicted_idx)

            # Show results
            st.subheader("🧪 Prediction Results")
            st.write(f"**Predicted Disease:** {predicted_label}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            # Treatment suggestion
            treatment = responses.get(predicted_label, "No treatment information available.")
            st.subheader("💊 Treatment Recommendation")
            st.write(treatment)
