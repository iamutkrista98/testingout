def run_leaf_disease_classifier():
    import streamlit as st
    import numpy as np
    import pandas as pd
    from PIL import Image
    import keras
    import requests
    import tempfile
    import os

    # 🌿 App Header
    st.markdown("<h1 style='text-align: center;'>🌿 Leaf Disease Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload or capture a leaf image to detect disease and get treatment advice.</p>", unsafe_allow_html=True)

    # 🔗 Remote model and local Excel mapping
    MODEL_URL = "https://huggingface.co/iamutkrista98/testing/resolve/main/testmodel.h5"
    EXCEL_PATH = "leaf_disease_responses.xlsx"

    # 📦 Load model from Hugging Face (cached for performance)
    @st.cache_resource
    def load_model_from_huggingface(url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            model = keras.models.load_model(tmp_path, compile=False)
            os.remove(tmp_path)
            return model
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            return None

    # 📊 Load class-to-label and treatment mappings from Excel
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
            st.error(f"❌ Error loading Excel file: {e}")
            return {}, {}

    # 🖼️ Preprocess uploaded image for model input
    def preprocess_image(uploaded_file):
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.asarray(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img

    # 📤 Image input options
    st.markdown("### 📸 Choose Image Source")
    input_method = st.radio("Select input method:", ["Upload from file", "Capture from camera"])
    st.caption("📌 Tip: Make sure the leaf is well-lit and centered in the frame for best results.")

    uploaded_file = None
    if input_method == "Upload from file":
        uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])
    elif input_method == "Capture from camera":
        uploaded_file = st.camera_input("📷 Take a photo of the leaf")

    # 🔍 Main classification logic
    if uploaded_file:
        try:
            with st.spinner("🔄 Loading model and data..."):
                model = load_model_from_huggingface(MODEL_URL)
                label_map, full_info_map = load_mappings()

            if not model or not label_map:
                st.error("❌ Failed to load model or mapping data.")
                return

            img_array, display_img = preprocess_image(uploaded_file)

            # ✅ Validate input shape
            if img_array.shape != (1, 224, 224, 3):
                st.error(f"❌ Invalid input shape: {img_array.shape}. Expected (1, 224, 224, 3).")
                return

            predictions = model.predict(img_array)
            top_idx = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]) * 100)
            predicted_label = label_map.get(top_idx, f"Unknown class {top_idx}")
            treatment = full_info_map.get(top_idx, {"response_message": "No treatment information available."})["response_message"]

            # 📸 Display uploaded image
            st.image(display_img, caption="📷 Uploaded Leaf", use_container_width=True)

            # 🧠 Prediction result
            st.markdown("---")
            st.markdown(f"<h3 style='color:#32CD32;'>🪴 Disease Detected: <strong>{predicted_label}</strong></h3>", unsafe_allow_html=True)

            # 📊 Confidence bar
            st.markdown("📊 **Confidence Level**")
            st.progress(confidence / 100)
            st.markdown(f"<p style='font-size:18px;'>Confidence: <strong>{confidence:.2f}%</strong></p>", unsafe_allow_html=True)

            # 💊 Treatment advice
            st.markdown("💊 **Treatment Recommendation**")
            st.markdown(f"""
                <div style='
                    background-color: #1e1e1e;
                    border-left: 6px solid #32CD32;
                    padding: 15px;
                    border-radius: 8px;
                    color: #f0f0f0;
                    font-size: 16px;
                    line-height: 1.6;
                '>
                    {treatment}
                </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.success("✅ Diagnosis complete. Follow the treatment plan above.")

        except Exception as e:
            st.error(f"⚠️ Error during processing: {e}")
    else:
        st.info("Please upload or capture a leaf image to begin.")


# Run the app
if __name__ == "__main__":
    run_leaf_disease_classifier()
