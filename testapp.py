import streamlit as st
import keras
import tempfile
import requests
import os

# Google Drive file ID
FILE_ID = "1jGCwRGwO2bTbvZQS_3yodaeyLjdPx5JQ"

@st.cache_resource
def load_model():
    # Create a temporary file to store the downloaded model
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "clean_model1.keras")

    # Download from Google Drive using export URL
    download_url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    response = requests.get(download_url)
    with open(model_path, "wb") as f:
        f.write(response.content)

    # Load the model using Keras 3.x compatible API
    model = keras.models.load_model(model_path, compile=False)
    return model

# Load model once
model = load_model()

# Streamlit UI
st.title("🌿 Plant Disease Classifier")
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    from PIL import Image
    import numpy as np

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # Adjust to your model's expected input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.success(f"Predicted class: {predicted_class}")
