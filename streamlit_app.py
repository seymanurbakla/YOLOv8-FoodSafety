import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load the model
MODEL_PATH = "best.pt"  # Path to your trained model
model = YOLO(MODEL_PATH)

# Title and description with improved visualization
st.title("YOLOv8 OBJECT DETECTION - FOOD SAFETY")

st.markdown("""
**WELCOME TO THE FOOD SAFETY OBJECT DETECTION APPLICATION!**

This application is designed to ensure food safety by detecting objects in food production environments. 

### The model can detect the following 7 classes:

- **GLOVES**
- **HAIRNET**
- **NO GLOVES**
- **NO HAIRNET**
- **NOT SMOKING**
- **SMOKING**
- **RAT**


Upload an image and let the model perform object detection to ensure a safe environment.
""")

# Image upload area
uploaded_file = st.file_uploader("CHOOSE AN IMAGE...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    
    # Convert to RGB if the image has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image = np.array(image)  # Convert to numpy array
    
    # Model prediction
    with st.spinner("RUNNING MODEL..."):
        results = model(image)  # Make predictions with the model

    # Get and display the processed image
    annotated_image = results[0].plot()  # Apply predictions to the image

    # Display processed image on Streamlit
    st.image(annotated_image, caption="PROCESSED IMAGE WITH DETECTIONS", use_column_width=True)

    # Display the detected objects
    st.write("DETECTED OBJECTS:")
    for item in results[0].boxes:
        class_id = int(item.cls[0])  # Class ID
        confidence = item.conf[0]  # Confidence score
        st.write(f"CLASS: {model.names[class_id].upper()} - CONFIDENCE: {confidence:.2f}")
