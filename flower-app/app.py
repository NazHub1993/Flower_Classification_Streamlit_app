import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from keras.applications.mobilenet_v2 import preprocess_input

# Page Configuration
st.set_page_config(page_title="Flower AI", page_icon="🌸", layout="wide")


#Loading the model and set the class names...................................
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("flower-app/flower_model.h5")


model = load_model()
CLASS_NAMES = ['Dandelion', 'Daisy', 'Tulips', 'Sunflowers', 'Roses']
#.............................................................................

# Placeholder URLs (No disk space used!)
IMG_PLACEHOLDER = "https://placehold.co/600x400?text=Your+Flower+Image+Will+Appear+Here"

# --- SIDEBAR ---
with st.sidebar:
    st.header("Input Method")
    mode = st.radio("Choose how to provide image:", [
                    "Upload File", "Paste Image URL"])
    st.divider()
    st.write("Current Disk Space: 🌕 Critical (Using Cloud URLs)")

# --- MAIN INTERFACE ---
st.title("🌸 Flower Classification App")

# Image Logic
image = None
if mode == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload a flower image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
else:
    url = st.text_input("Paste Image Address (URL):")
    if url:
        try:
            response = requests.get(url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("Invalid URL. Make sure it's a direct image link.")

st.divider()

# --- LAYOUT: PHOTO SEC & ANALYSIS SEC ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Image Section")
    if image:
        st.image(image, caption="Uploaded Flower", use_container_width=True)
    else:
        st.image(IMG_PLACEHOLDER, use_container_width=True)

with col2:
    st.subheader("📊 Analysis Section")

    # Check if an image is provided to enable the button
    if image:
        run_button = st.button("🚀 Run Prediction")

        if run_button:
            with st.spinner("Classifying..."):
                # Preprocess
                img = image.resize((224, 224))
                img_array = np.array(img)
                img_array = preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                # Prediction
                preds = model.predict(img_array)
                pred_index = np.argmax(preds)
                pred_label = CLASS_NAMES[pred_index]
                confidence = float(np.max(preds))

                # Display Results
                st.success(f"**Prediction: {pred_label}**")
                st.metric(label="Confidence Score",
                          value=f"{confidence * 100:.2f}%")
                st.progress(confidence)
        else:
            st.info("Click 'Run Prediction' to analyze the image.")
    else:
        # Before image is uploaded
        st.write(
            "Analysis results will appear here once you provide an image and click 'Run'.")
