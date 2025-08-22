import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import keras
import matplotlib.pyplot as plt
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

# ---------------- PREDICTION VISUALIZATION ----------------
def show_prediction_chart(img_path, model, class_dict):
    labels = list(class_dict.values())
    img = Image.open(img_path).convert("RGB")
    resized_img = img.resize((224, 224))
    img_array = np.asarray(resized_img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    probs = list(predictions[0])

    fig = plt.figure(figsize=(10, 12))
    plt.subplot(2, 1, 1)
    plt.imshow(resized_img)
    plt.axis('off')
    plt.title("Uploaded Leaf Image")

    plt.subplot(2, 1, 2)
    bars = plt.barh(labels, probs, color='mediumseagreen')
    plt.xlabel('Prediction Confidence', fontsize=15)
    ax = plt.gca()
    ax.bar_label(bars, fmt='%.2f')
    plt.tight_layout()

    st.pyplot(fig)

    top_idx = int(np.argmax(probs))
    top_label = labels[top_idx]
    confidence = float(probs[top_idx] * 100)
    return top_label, confidence

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
            predicted_label, confidence = show_prediction_chart(uploaded_file, model, label_map)

            # Lookup treatment info
            treatment = full_info_map.get(
                list(label_map.keys())[list(label_map.values()).index(predicted_label)],
                {"response_message": "No treatment information available."}
            )["response_message"]

            # ---------------- DISPLAY RESULTS ----------------
            st.subheader("🔍 Prediction Results")
            st.markdown(f"**Disease Name:** `{predicted_label}`")
            st.markdown(f"**Confidence:** `{confidence:.2f}%`")

            st.subheader("💊 Treatment Recommendation")
            st.markdown(treatment)

            st.success("✅ Diagnosis complete. Follow the treatment plan above.")

    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}")
else:
    st.info("Please upload a leaf image to begin.")
