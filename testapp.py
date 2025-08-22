import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from keras.layers import TFSMLayer
import requests
import io
import tempfile
import os
import zipfile

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Leaf Disease Detector", layout="centered")
st.title("🌿 Plant Leaf Disease Classifier")
st.markdown("Upload a leaf image to detect disease and get treatment advice.")

# ---------------- GOOGLE DRIVE HELPERS ----------------
def download_from_drive(drive_url):
    try:
        file_id = drive_url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(download_url)
        response.raise_for_status()
        return io.BytesIO(response.content)
    except Exception as e:
        st.error(f"Failed to download from Google Drive: {e}")
        return None

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_from_drive(drive_url):
    model_bytes = download_from_drive(drive_url)
    if model_bytes:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "model.zip")
            with open(zip_path, "wb") as f:
                f.write(model_bytes.read())

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)

            model = TFSMLayer(tmp_dir, call_endpoint="serving_default")
            return model
    return None

# ---------------- LOAD LABELS & RESPONSES ----------------
@st.cache_data
def load_mappings_from_drive(drive_url):
    excel_bytes = download_from_drive(drive_url)
    if excel_bytes:
        df = pd.read_excel(excel_bytes)
        label_map = dict(zip(df["class_index"], df["disease_name"]))
        full_info_map = {
            row["class_index"]: {
                "disease_name": row["disease_name"],
                "response_message": row["response_message"]
            }
            for _, row in df.iterrows()
        }
        return label_map, full_info_map
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

# ---------------- UI INPUTS ----------------
model_url = st.text_input("🔗 Paste public Google Drive link to zipped SavedModel", 
    value="https://drive.google.com/file/d/1enLxaLvyPpJL1yuwDByVMmZMiAFQ6iGz/view")

excel_url = st.text_input("🔗 Paste public Google Drive link to Excel file (.xlsx)", 
    value="https://drive.google.com/file/d/1dJbbLx348xTBiOCh4ywW-qAcfNhqbrVO/view")

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

# ---------------- MAIN LOGIC ----------------
if model_url and excel_url and uploaded_file:
    try:
        with st.spinner("🔄 Loading model and data..."):
            model = load_model_from_drive(model_url)
            label_map, full_info_map = load_mappings_from_drive(excel_url)

        if not model or not label_map:
            st.error("❌ Failed to load model or mapping data.")
        else:
            try:
                img_array, display_img = preprocess_image(uploaded_file)
                preds = model(img_array)
            except Exception as e:
                st.warning(f"RGB preprocessing failed: {e}. Trying grayscale...")
                img_array, display_img = preprocess_image_grayscale(uploaded_file)
                preds = model(img_array)

            predicted_idx = int(np.argmax(preds.numpy()[0]))
            confidence = float(np.max(preds.numpy()[0]) * 100)

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
    st.info("Please upload a leaf image and provide both Google Drive links to begin.")
