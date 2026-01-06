from ultralytics import YOLO
import cv2
from satellite_risk import get_fire_risk

# Load model
model = YOLO("fire.pt")

# Simulated satellite risk
lat, lon = 28.6, 77.2
risk = get_fire_risk(lat, lon)

cap = cv2.VideoCapture("fire.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)
    annotated = results[0].plot()

    # Show satellite risk on screen
    cv2.putText(annotated, f"Satellite Risk: {risk}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if len(results[0].boxes) > 0:
        if risk == "High":
            text = "🔥 HIGH RISK FIRE"
            color = (0, 0, 255)
        elif risk == "Medium":
            text = "⚠️ MEDIUM RISK FIRE"
            color = (0, 165, 255)
        else:
            text = "Fire detected (Low Risk Area)"
            color = (0, 255, 0)

        cv2.putText(annotated, text, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Forest Fire Detection System", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
