import cv2
import torch
import time
from ultralytics import YOLO

model_det = YOLO("yolov8n.pt")
model_pose = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    results_det = model_det(frame, classes=[0], conf=0.5, verbose=False)
    results_pose = model_pose(frame, conf=0.5, verbose=False)

    annotated_frame = results_det[0].plot()
    annotated_frame = results_pose[0].plot()

    end_time = time.time()
    fps = 1 / (end_time - start_time)

    num_persons = len(results_det[0].boxes) if results_det else 0
    acc_text = f"Persons: {num_persons} | FPS: {fps:.2f}"
    cv2.putText(annotated_frame, acc_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated_frame, timestamp, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLOv8n + Pose", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()