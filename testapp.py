# streamlit_leaf_disease_app.py

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
import gdown
import os

st.set_page_config(page_title="🌿 Plant Leaf Disease Classifier", layout="centered")

# ---------------- CONFIG ----------------
# Google Drive File ID of your .h5 model
# Example: https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
GOOGLE_DRIVE_FILE_ID = "1Ttkd63o8AMkSU5WToJd_aNUm3O9H5lny"
MODEL_LOCAL_PATH = "plant_disease_model.h5"
MAPPING_XLSX = "leaf_disease_responses.xlsx"
IMG_SIZE = (224, 224)  # match training pipeline

# ---------------- HELPERS ----------------
def download_model_from_drive(file_id, output_path):
    """Download the model from Google Drive if not already cached."""
    if not os.path.exists(output_path):
        st.info("Downloading model from Google Drive... (first time only)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return output_path

@st.cache_data
def load_mapping(path: str):
    df = pd.read_excel(path)
    if "class_index" in df.columns:
        df = df.sort_values("class_index")
    return df.reset_index(drop=True)

@st.cache_resource
def load_model():
    local_model_path = download_model_from_drive(GOOGLE_DRIVE_FILE_ID, MODEL_LOCAL_PATH)
    model = tf.keras.models.load_model(local_model_path, compile=False)
    return model

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
    st.error(f"Failed to load mapping file: {e}")
    st.stop()

# Load model (download if needed)
try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Upload image
uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    arr = preprocess_image(image)
    with st.spinner("Running inference..."):
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

st.caption("⚙️ Model automatically downloaded from Google Drive (cached for session).")
