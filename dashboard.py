import streamlit as st
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import cv2
from satellite_risk import get_fire_risk
import folium
from streamlit_folium import st_folium


st.set_page_config(page_title="Forest Fire Detection System", layout="wide")

st.title("Forest Fire Detection System")

# Load model
model = YOLO("fire.pt")

# Simulated satellite risk
lat, lon = 28.6, 77.2
risk = get_fire_risk(lat, lon)

st.metric("Satellite Fire Risk", risk)
st.subheader("Fire Location Map")

lat, lon = 30.1, 79.2  # sample forest coordinates

m = folium.Map(location=[lat, lon], zoom_start=7)
folium.Marker([lat, lon], tooltip="Fire Risk Zone", icon=folium.Icon(color="red")).add_to(m)

st_folium(m, width=700, height=400)


video_file = "fire.mp4"
cap = cv2.VideoCapture(video_file)

log = []

frame_placeholder = st.empty()
log_placeholder = st.empty()

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

    if len(log) > 0:
        df = pd.DataFrame(log)
        log_placeholder.table(df)

cap.release()

"""python -m streamlit run dashboard.py"""
