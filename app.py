import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- 1. Load the Saved Model ---
@st.cache_resource
def load_my_model():
    """Loads the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model('dental_cavity_detector.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# --- 2. Define Helper Functions ---
def preprocess_image(image_file):
    """Loads and preprocesses an image for model prediction."""
    img = Image.open(image_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# --- 3. Create the Streamlit UI ---
st.title("🦷 Dental Cavity Detection")
st.write("Upload an image of a tooth, and this app will predict whether a cavity is present.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Analyzing...")

    if model is not None:
        try:
            processed_image = preprocess_image(uploaded_file)
            prediction = model.predict(processed_image)
            
            # --- DIAGNOSTIC SECTION ---
            # This will help us understand the model's output.
            st.write(f"The model's raw output (prediction score) is: **{prediction[0][0]:.4f}**")
            st.write("""
            **Instructions:**
            1. Upload an image that **CLEARLY HAS a cavity**. Note the score.
            2. Upload an image that is **CLEARLY HEALTHY**. Note the score.
            """)

            confidence_score = prediction[0][0]

            # --- FINAL LOGIC: CHOOSE ONE BLOCK ---

            # BLOCK 1: Use this if a CAVITY image gives a LOW score (less than 0.5)
            # if confidence_score < 0.5:
            #     st.error("Prediction: Cavity Detected")
            #     st.write(f"Confidence Score for Cavity: {1 - confidence_score:.2f}")
            # else:
            #     st.success("Prediction: No Cavity Detected")
            #     st.write(f"Confidence Score for No Cavity: {confidence_score:.2f}")

            # BLOCK 2: Use this if a CAVITY image gives a HIGH score (greater than 0.5)
            if confidence_score > 0.5:
                st.error("Prediction: Cavity Detected")
                st.write(f"Confidence Score for Cavity: {confidence_score:.2f}")
            else:
                st.success("Prediction: No Cavity Detected")
                st.write(f"Confidence Score for No Cavity: {1 - confidence_score:.2f}")


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

