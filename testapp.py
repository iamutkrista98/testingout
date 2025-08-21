import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd

# --- Constants ---
MODEL_PATH = "clean_model1.keras"
EXCEL_PATH = "leaf_disease_responses.xlsx"
IMG_SIZE = (225, 225)

# --- Load Model ---
@st.cache_resource
def load_plant_model():
    return load_model(MODEL_PATH)

model = load_plant_model()

# --- Load Mapping ---
@st.cache_data
def load_mapping():
    df = pd.read_excel(EXCEL_PATH)
    return df

mapping_df = load_mapping()

# --- Preprocess Image ---
def preprocess_image(image):
    try:
        img = image.convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        return None

# --- Predict Disease ---
def predict_disease(img_array):
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    return predicted_class, confidence

# --- Map to Treatment ---
def get_treatment_info(class_id):
    row = mapping_df[mapping_df["ClassID"] == class_id]
    if not row.empty:
        disease = row.iloc[0]["Disease"]
        treatment = row.iloc[0]["Treatment"]
        return disease, treatment
    else:
        return "Unknown", "No treatment info available"

# --- Streamlit UI ---
st.title("🌿 Plant Leaf Disease Classifier")
st.markdown("Upload a leaf image to detect the disease and get treatment advice.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    if img_array is not None:
        class_id, confidence = predict_disease(img_array)
        disease, treatment = get_treatment_info(class_id)

        st.subheader("🧪 Prediction Result")
        st.write(f"**Disease:** {disease}")
        st.write(f"**Confidence:** {confidence:.2%}")

        st.subheader("💊 Suggested Treatment")
        st.write(treatment)
