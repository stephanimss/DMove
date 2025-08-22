import cv2
import time
import numpy as np
import gpiozero
from ultralytics import YOLO

model_path = "yolo11n_ncnn_model"   
cam_source = "usb0"                
resW, resH = 1280, 720              #1920×1080
min_thresh = 0.5
gpio_pin = 14

led = gpiozero.LED(gpio_pin)

model = YOLO(model_path, task="detect")
labels = model.names

if "usb" in cam_source:
    cam_type = "usb"
    cam_idx = int(cam_source[3:])
    cam = cv2.VideoCapture(cam_idx)
    cam.set(3, resW)
    cam.set(4, resH)
elif "picamera" in cam_source:
    from picamera2 import Picamera2
    cam_type = "picamera"
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cam.start()
else:
    print("Cam source invalid!")
    exit()

consecutive_detections = 0
gpio_state = 0

fps_buffer = []
fps_avg_len = 50
avg_fps = 0

while True:
    t_start = time.perf_counter()

    if cam_type == "usb":
        ret, frame = cam.read()
        if not ret: break
    else:
        frame_bgra = cam.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)

    results = model.track(frame, verbose=False)
    detections = results[0].boxes

    person_detected = False
    for det in detections:
        classid = int(det.cls.item())
        conf = det.conf.item()

        if conf > min_thresh:
            classname = labels[classid]
            if classname == 'person':
                person_detected = True

            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
            label = f'{classname}: {int(conf*100)}%'
            cv2.putText(frame, label, (xmin, ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    if person_detected:
        consecutive_detections = min(8, consecutive_detections + 1)
    else:
        consecutive_detections = max(0, consecutive_detections - 1)

    if consecutive_detections >= 8 and gpio_state == 0:
        gpio_state = 1
        led.on()
    elif consecutive_detections <= 0 and gpio_state == 1:
        gpio_state = 0
        led.off()

    if gpio_state == 0:
        cv2.putText(frame, "Light OFF", (20,60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)
    else:
        cv2.putText(frame, "Light ON", (20,60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)

    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)
    fps_buffer.append(frame_rate_calc)
    if len(fps_buffer) > fps_avg_len:
        fps_buffer.pop(0)
    avg_frame_rate = np.mean(fps_buffer)

    cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (20,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.imshow("YOLOv8n", frame)

    key = cv2.waitKey(5)
    if key == ord('q') or key == ord('Q'):
        break

if cam_type == 'usb': cam.release()
cv2.destroyAllWindows()