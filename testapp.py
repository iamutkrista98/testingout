import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import keras
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Leaf Disease Detector", layout="centered")
st.title("🌿 Plant Leaf Disease Classifier")
st.markdown("Upload a leaf image to detect disease and get treatment advice.")

# ---------------- FILE PATHS ----------------
MODEL_PATH = "clean_model1.keras"
EXCEL_PATH = "leaf_disease_responses.xlsx"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH, compile=False)

# ---------------- LOAD LABELS & RESPONSES ----------------
@st.cache_data
def load_mappings():
    try:
        df = pd.read_excel(EXCEL_PATH)
        label_map = dict(zip(df["class_index"], df["disease_name"]))
        full_info_map = {
            row["class_index"]: {
                "disease_name": row["disease_name"],
                "response_message": row["response_message"]
            }
            for _, row in df.iterrows()
        }
        return label_map, full_info_map
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return {}, {}

# ---------------- PREPROCESS IMAGE ----------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def preprocess_image_grayscale(uploaded_file):
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((225, 225))
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# ---------------- UI INPUT ----------------
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

# ---------------- MAIN LOGIC ----------------
if uploaded_file:
    try:
        with st.spinner("🔄 Loading model and data..."):
            model = load_model()
            label_map, full_info_map = load_mappings()

        if not model or not label_map:
            st.error("❌ Failed to load model or mapping data.")
        else:
            try:
                img_array, display_img = preprocess_image(uploaded_file)
                preds = model.predict(img_array)
            except Exception as e:
                st.warning(f"RGB preprocessing failed: {e}. Trying grayscale...")
                img_array, display_img = preprocess_image_grayscale(uploaded_file)
                preds = model.predict(img_array)

            predicted_idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]) * 100)

            disease_info = full_info_map.get(predicted_idx, {
                "disease_name": f"Unknown class {predicted_idx}",
                "response_message": "No treatment information available."
            })

            # ---------------- DISPLAY RESULTS ----------------
            st.image(display_img, caption="📷 Uploaded Leaf", use_column_width=True)

            st.subheader("🔍 Prediction Results")
            st.markdown(f"**Disease Name:** `{disease_info['disease_name']}`")
            st.markdown(f"**Confidence:** `{confidence:.2f}%`")

            st.subheader("💊 Treatment Recommendation")
            st.markdown(disease_info["response_message"])

            st.success("✅ Diagnosis complete. Follow the treatment plan above.")

    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}")
else:
    st.info("Please upload a leaf image to begin.")
