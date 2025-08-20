import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image # Used for image handling

# --- 1. Load the Saved Model ---
# Use st.cache_resource to load the model only once and cache it.
# This prevents the model from being reloaded every time the user interacts with the app.
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
    """
    Loads and preprocesses an image for model prediction.
    """
    # Open the image file
    img = Image.open(image_file)
    
    # Convert image to RGB (some images might be RGBA or grayscale)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Resize to the target size your model expects (e.g., 150x150)
    img = img.resize((150, 150))
    
    # Convert image to a numpy array
    img_array = np.array(img)
    
    # Expand dimensions to create a batch of 1
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the pixel values to be between 0 and 1
    img_array = img_array / 255.0
    
    return img_array


# --- 3. Create the Streamlit UI ---

# Set the title of the app
st.title("🦷 Dental Cavity Detection")

# Add a description
st.write("Upload an image of a tooth, and this app will predict whether a cavity is present.")

# Create a file uploader widget
# This allows users to upload an image file.
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    st.write("") # Add a little space
    st.write("Analyzing...")

    # Ensure the model is loaded before making a prediction
    if model is not None:
        try:
            # Preprocess the image
            processed_image = preprocess_image(uploaded_file)
            
            # Make a prediction
            prediction = model.predict(processed_image)
            
            # --- REVERTED LOGIC ---
            # A low prediction score (closer to 0) indicates a 'cavity'.
            # A high prediction score (closer to 1) indicates 'no_cavity'.
            
            confidence_score = prediction[0][0]
            
            if confidence_score < 0.5: # Low score means cavity
                st.error("Prediction: Cavity Detected")
                st.write(f"Confidence Score for Cavity: {1 - confidence_score:.2f}")
            else: # High score means no cavity
                st.success("Prediction: No Cavity Detected")
                st.write(f"Confidence Score for No Cavity: {confidence_score:.2f}")


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
