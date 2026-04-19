import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Coastal AI Detector", page_icon="🌿")

st.title("🌿 Coastal Plant Species Detector")
st.write("Detecting: Mangroves, Seagrass, Seaweed, and Corals")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- UPLOADER ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    
    # Run Detection
    results = model.predict(img, conf=0.4)
    
    # --- DRAWING WITHOUT CV2 ---
    # We use matplotlib to show the results instead of cv2.plot()
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    # Draw boxes manually from results
    for box in results[0].boxes:
        # Get coordinates
        b = box.xyxy[0].to('cpu').detach().numpy() 
        c = box.cls
        # Draw a simple rectangle
        rect = plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], 
                             fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(b[0], b[1], f"{model.names[int(c)]}", 
                color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')
    st.pyplot(fig)
    
    # Summary list
    st.write("### Detection Summary:")
    for c in results[0].boxes.cls:
        st.success(f"Found: {model.names[int(c)]}")
