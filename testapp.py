import streamlit as st
import keras
import tempfile
import gdown
import os
import numpy as np
from PIL import Image

# Google Drive file ID for your .keras model
FILE_ID = "1PsWiPaVBUP-T0X-r3B2uYlaiC3PLGMt6"

# Optional: Map predicted class index to disease name
class_names = {
    0: "Healthy",
    1: "Powdery Mildew",
    2: "Leaf Spot",
    3: "Rust",
    4: "Blight"
    # Add more if needed
}

@st.cache_resource
def load_model():
    # Create a temporary directory and file path
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "clean_model1.keras")

    # Download the model using gdown
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", model_path, quiet=False)

    # Load the model using Keras
    model = keras.models.load_model(model_path, compile=False)
    return model

# Load the model once
model = load_model()

# Streamlit UI
st.title("🌿 Plant Leaf Disease Classifier")
st.markdown("Upload a leaf image and let the model identify the disease.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # Adjust if your model expects a different size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))

    # Display result
    disease_name = class_names.get(predicted_class, f"Class {predicted_class}")
    st.success(f"🧠 Predicted Disease: **{disease_name}**")
