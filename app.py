import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

# --- PAGE CONFIG ---
st.set_page_config(page_title="Coastal Plant AI Detector", page_icon="🌿")

st.title("🌿 Coastal Plant Species Detector")
st.write("Upload an image to detect **Mangroves, Seagrass, Seaweed, and Corals**.")

# --- SIDEBAR ---
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.45)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # This loads your trained best.pt
    return YOLO("best.pt")

model = load_model()

# --- UPLOADER ---
uploaded_file = st.file_uploader("Choose a coastal image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Run Detection
    results = model.predict(image, conf=confidence_threshold)
    
    # The 'plot()' function from Ultralytics uses its own internal logic
    # so we don't need to import cv2 here!
    res_plotted = results[0].plot()
    
    # Show the result
    st.image(res_plotted, caption="Detection Results", use_container_width=True)
    
    # Summary of detections
    st.write("### Found:")
    names = results[0].names
    counts = {}
    for c in results[0].boxes.cls:
        label = names[int(c)]
        counts[label] = counts.get(label, 0) + 1
    
    if counts:
        for plant, count in counts.items():
            st.info(f"{plant}: {count}")
    else:
        st.warning("No coastal plants detected at this confidence level.")
