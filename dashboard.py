import os
import streamlit as st
import pandas as pd
from datetime import datetime
import cv2
import folium
from streamlit_folium import st_folium

# Cloud-safe YOLO setup
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
os.environ["ULTRALYTICS_SETTINGS"] = "False"

st.set_page_config(page_title="Forest Fire Detection System", layout="wide")
st.title("🌲🔥 Forest Fire Detection System")

# Import YOLO after Streamlit is ready
from ultralytics import YOLO

# Load model
@st.cache_resource
def load_model():
    model = YOLO("fire.pt")
    model.to("cpu")
    return model

model = load_model()

# Upload video
uploaded_video = st.file_uploader("Upload drone video", type=["mp4", "avi"])
if uploaded_video is None:
    st.warning("Please upload a drone video to start.")
    st.stop()

with open("temp_video.mp4", "wb") as f:
    f.write(uploaded_video.read())

cap = cv2.VideoCapture("temp_video.mp4")

# Satellite risk (safe fallback)
try:
    from satellite_risk import get_fire_risk
    lat, lon = 30.1, 79.2
    risk = get_fire_risk(lat, lon)
except:
    risk = "Unknown"

st.metric("Satellite Fire Risk", risk)

# Map
st.subheader("Fire Location Map")
m = folium.Map(location=[30.1, 79.2], zoom_start=7)
folium.Marker([30.1, 79.2], tooltip="Fire Risk Zone", icon=folium.Icon(color="red")).add_to(m)
st_folium(m, width=700, height=400)

frame_placeholder = st.empty()
log_placeholder = st.empty()

log = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)
    annotated = results[0].plot()

    if len(results[0].boxes) > 0:
        time = datetime.now().strftime("%H:%M:%S")
        log.append({"Time": time, "Alert": f"Fire detected — Risk {risk}"})

    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

    if log:
        df = pd.DataFrame(log)
        log_placeholder.table(df)

cap.release()
