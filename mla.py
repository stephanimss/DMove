import cv2
from  ultralytics import YOLO
import os
from datetime import datetime, timedelta
import time

model = YOLO("yolov8n.pt")

save_dir = "snapshots"
os.makedirs(save_dir, exist_ok=True)

last_snap_time = datetime.min
snap_interval = timedelta(minutes=2)
count_snap = 0
max_snaps = 5

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret: 
        break
    results = model(frame)
    detected_person = False
     
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == "person":
                detected_person =True
                now = datetime.now()

                datetime_string = now.strftime("%Y-%m-%d %H:%M:%S")
                text = F"{label} {datetime_string}"
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        if detected_person and (now - last_snap_time >= snap_interval):
            while count_snap < max_snaps:
                now = datetime.now()
                filename = now.strftime(f"snapshots_%Y-%m-%d_%H:%M:%S_{count_snap + 1}.png")
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, frame)
                count_snap += 1
                print(F"snapshots terambil ({count_snap} / {max_snaps}): {filepath}")
                time.sleep(0.5)
            count_snap = 0
            last_snap_time = now

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0XFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()