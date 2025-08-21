import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import keras
import cv2

# ---------------- CONFIG ----------------
MODEL_PATH = "clean_model1.keras"
MAPPING_XLSX = "leaf_disease_responses.xlsx"

# ---------------- LOAD LABELS & RESPONSES ----------------
@st.cache_data
def load_mappings():
    df = pd.read_excel(MAPPING_XLSX)
    label_map = dict(zip(df["class_index"], df["disease_name"]))
    treatment_map = dict(zip(df["disease_name"], df["response_message"]))
    return label_map, treatment_map

label_map, treatment_map = load_mappings()

# ---------------- PREPROCESS IMAGE ----------------
def preprocess_image(uploaded_file, expected_channels=3):
    # Read the file
    image = Image.open(uploaded_file)
    
    # Determine if we need grayscale or RGB based on expected channels
    if expected_channels == 1:
        image = image.convert('L')  # Convert to grayscale
        img_size = (225, 225)  # Grayscale model expects 225x225
    else:
        image = image.convert('RGB')  # Convert to RGB
        img_size = (224, 224)  # RGB model expects 224x224
    
    # Resize to expected dimensions
    image = image.resize(img_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
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
    
    # Normalize
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, image, expected_channels

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Leaf Disease Detector", layout="centered")
st.title("🌿 Plant Leaf Disease Classifier")
st.markdown("Upload a leaf image to detect disease and get treatment advice.")

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        with st.spinner("Analyzing image..."):
            # ✅ Load model only when needed
            @st.cache_resource
            def load_model():
                model = keras.models.load_model(MODEL_PATH, compile=False)
                
                # Try to determine input shape from model
                try:
                    input_shape = model.layers[0].input_shape
                    if isinstance(input_shape, list):
                        input_shape = input_shape[0]
                    expected_channels = input_shape[-1]
                    return model, expected_channels
                except:
                    # If we can't determine, default to RGB (3 channels)
                    return model, 3
            
            model, expected_channels = load_model()
            st.write(f"Model expects input with {expected_channels} channel(s)")

            # Preprocess and predict
            img_array, display_img, used_channels = preprocess_image(uploaded_file, expected_channels)
            
            # Debug: Check the shape of the image array
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
        st.error("Please make sure you're uploading a valid image file.")
        # For debugging
        import traceback
        st.error(traceback.format_exc())
else:
    st.info("Please upload a leaf image to begin.")
