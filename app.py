import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2

# --- PAGE CONFIG ---
st.set_page_config(page_title="Coastal Plant AI Detector", page_icon="🌿")

st.title("🌿 Coastal Plant Species Detector")
st.write("Upload an image to detect **Mangroves, Seagrass, Seaweed, and Corals** using our trained YOLO model.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Model Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # We use 'ultralytics' for YOLOv8/v11. 
    # If you used YOLOv5, Claude can adjust this slightly.
    from ultralytics import YOLO
    model = YOLO("best.pt") 
    return model

model = load_model()

# --- IMAGE UPLOADER ---
uploaded_file = st.file_uploader("Choose a coastal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Run Detection
    st.write("### Detecting...")
    results = model.predict(image, conf=confidence_threshold)
    
    # Plot results on the image
    res_plotted = results[0].plot()
    
    # Display Result
    st.image(res_plotted, caption="Detection Result", use_column_width=True)
    
    # Show summary of what was found
    st.write("### Detection Summary:")
    labels = results[0].names
    counts = {}
    for c in results[0].boxes.cls:
        label = labels[int(c)]
        counts[label] = counts.get(label, 0) + 1
    
    if counts:
        for plant, count in counts.items():
            st.success(f"Detected **{count} {plant}**")
    else:
        st.warning("No coastal plants detected. Try lowering the confidence threshold.")