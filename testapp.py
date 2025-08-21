import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import keras

# ---------------- CONFIG ----------------
MODEL_PATH = "clean_model1.keras"
MAPPING_XLSX = "leaf_disease_responses.xlsx"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    
    # Detect the model's expected input shape
    try:
        input_shape = model.layers[0].input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        expected_channels = input_shape[-1]
        expected_size = input_shape[1:3]  # Height and width
        return model, expected_channels, expected_size
    except:
        # Default to RGB if we can't detect
        return model, 3, (224, 224)

# ---------------- LOAD LABELS & RESPONSES ----------------
@st.cache_data
def load_mappings():
    try:
        df = pd.read_excel(MAPPING_XLSX)
        label_map = dict(zip(df["class_index"], df["disease_name"]))
        treatment_map = dict(zip(df["disease_name"], df["response_message"]))
        return label_map, treatment_map
    except Exception as e:
        st.error(f"Error loading Excel: {e}")
        return {}, {}

# ---------------- PREPROCESS IMAGE ----------------
def preprocess_image(uploaded_file, expected_channels, expected_size):
    img = Image.open(uploaded_file)
    
    # Convert to appropriate color mode based on model expectations
    if expected_channels == 1:
        img = img.convert('L')  # Grayscale
    else:
        img = img.convert('RGB')  # RGB
    
    # Resize to expected dimensions
    img = img.resize(expected_size)
    
    # Convert to numpy array
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    
    # Handle channel dimension
    if expected_channels == 1 and len(img_array.shape) == 2:
        # Add channel dimension for grayscale
        img_array = np.expand_dims(img_array, axis=-1)
    elif expected_channels == 3 and len(img_array.shape) == 2:
        # Convert grayscale to RGB
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif expected_channels == 3 and img_array.shape[2] == 4:
        # Remove alpha channel
        img_array = img_array[:, :, :3]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Leaf Disease Detector", layout="centered")
st.title("🌿 Plant Leaf Disease Classifier")
st.markdown("Upload a leaf image to detect disease and get treatment advice.")

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        with st.spinner("Loading model and processing image..."):
            # Load model and detect its requirements
            model, expected_channels, expected_size = load_model()
            label_map, treatment_map = load_mappings()
            
            # Show model requirements for debugging
            st.write(f"Model expects: {expected_channels} channel(s), size: {expected_size}")
            
            # Preprocess and predict
            img_array, display_img = preprocess_image(uploaded_file, expected_channels, expected_size)
            st.write(f"Processed image shape: {img_array.shape}")
            
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

    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}")
        st.info("Please make sure you're uploading a valid image file.")
else:
    st.info("Please upload a leaf image to begin.")
