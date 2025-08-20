# streamlit_leaf_disease_app.py

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
import gdown
import os, zipfile

st.set_page_config(page_title="🌿 Plant Leaf Disease Classifier", layout="centered")

# ---------------- CONFIG ----------------
# Your Google Drive File ID (from "share link")
# Example link: https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
GOOGLE_DRIVE_FILE_ID = "1IgXN5-6nzWmI0XTTGqxcNpYaBgN8o89W"

# Local paths
MODEL_H5_PATH = "plant_disease_model.h5"
MODEL_FOLDER_PATH = "plant_disease_model"  # for SavedModel format
MAPPING_XLSX = "leaf_disease_responses.xlsx"
IMG_SIZE = (224, 224)

# ---------------- HELPERS ----------------
def download_from_drive(file_id, output_path):
    """Download file from Google Drive if not already cached."""
    if not os.path.exists(output_path):
        st.info("📥 Downloading model from Google Drive (first run only)...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return output_path

def extract_zip(zip_path, extract_to="."):
    """Unzip model folder if provided as a .zip."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

@st.cache_data
def load_mapping(path: str):
    df = pd.read_excel(path)
    if "class_index" in df.columns:
        df = df.sort_values("class_index")
    return df.reset_index(drop=True)

@st.cache_resource
def load_model():
    """
    Try loading as:
    1. TensorFlow SavedModel (zipped folder)
    2. Keras .h5 (with compile=False)
    """
    # Case A: zipped SavedModel (future-proof)
    zip_path = "model.zip"
    if GOOGLE_DRIVE_FILE_ID and GOOGLE_DRIVE_FILE_ID.endswith("zip"):
        download_from_drive(GOOGLE_DRIVE_FILE_ID, zip_path)
        extract_zip(zip_path, ".")
        return tf.keras.models.load_model(MODEL_FOLDER_PATH)

    # Case B: plain .h5
    local_model_path = download_from_drive(GOOGLE_DRIVE_FILE_ID, MODEL_H5_PATH)
    return tf.keras.models.load_model(local_model_path, compile=False)

def preprocess_image(img: Image.Image, target_size=IMG_SIZE):
    img = img.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(model, arr, class_names):
    preds = model.predict(arr)
    probs = tf.nn.softmax(preds, axis=1).numpy()
    top_idx = int(np.argmax(probs[0]))
    top_conf = float(probs[0][top_idx])
    top3_idx = np.argsort(-probs[0])[:3]
    top3 = [(class_names[i], float(probs[0][i])) for i in top3_idx]
    return top_idx, top_conf, top3

# ---------------- UI ----------------
st.title("🌿 Plant Leaf Disease Classifier")
st.write("Upload a plant leaf photo to classify the disease, view confidence, and receive a treatment recommendation.")

# Load mapping
try:
    mapping_df = load_mapping(MAPPING_XLSX)
    class_names = mapping_df["label"].tolist()
except Exception as e:
    st.error(f"❌ Failed to load mapping file: {e}")
    st.stop()

# Load model (from Drive)
try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.info("Tip: If using a zipped SavedModel, upload it as a .zip to Drive and update GOOGLE_DRIVE_FILE_ID.")
    st.stop()

# Upload image
uploaded = st.file_uploader("📷 Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    arr = preprocess_image(image)
    with st.spinner("🔍 Running inference..."):
        top_idx, top_conf, top3 = predict_image(model, arr, class_names)

    pred_label = class_names[top_idx]

    st.subheader("Prediction")
    st.write(f"**Label:** {pred_label}")
    st.write(f"**Confidence:** {top_conf:.2%}")

    with st.expander("Top-3 predictions"):
        for name, p in top3:
            st.write(f"- {name} — {p:.2%}")

    # Treatment lookup
    row = mapping_df.iloc[top_idx]
    severity = row.get("severity", "low")
    message = row.get("response_message", "")

    if severity in ("critical", "high"):
        st.error(message)
    elif severity == "medium":
        st.warning(message)
    else:
        st.success(message)

    with st.expander("Details"):
        st.json({
            "category": row.get("category", ""),
            "severity": severity,
            "plant": row.get("plant", ""),
            "disease_name": row.get("disease_name", ""),
            "action_code": row.get("action_code", "")
        })

st.caption("⚙️ Model is downloaded once from Google Drive and cached.")
