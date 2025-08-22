import os
import sys
import time
import datetime
import cv2
import numpy as np
from ultralytics import YOLO

# Path model NCNN hasil export (export dilakukan sekali saja di PC)
model_path = 'yolov8n_ncnn_model'
cam_source = 'usb0'         # 'usb0' untuk USB cam, 'picamera0' untuk PiCam
min_thresh = 0.5            # Ambang batas confidence
resW, resH = 1280, 720      # Resolusi kamera lebih rendah = FPS lebih tinggi
bbox_color_person = (0, 0, 255)

# Pastikan model ada
if not os.path.exists(model_path):
    print("ERROR: Model path tidak ditemukan!")
    sys.exit()

# Load model NCNN
try:
    model = YOLO(model_path, task='detect')
    labels = model.names
    print(f"Model YOLOv8n NCNN berhasil dimuat. Kelas: {labels}")
except Exception as e:
    print(f"ERROR saat memuat model: {e}")
    sys.exit()

# Inisialisasi kamera
cam = None
cam_type = None
try:
    if 'usb' in cam_source:
        cam_type = 'usb'
        cam_idx = int(cam_source[3:])
        cam = cv2.VideoCapture(cam_idx)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
        if not cam.isOpened():
            raise IOError("Kamera USB tidak terbuka.")
        print(f"Kamera USB {cam_idx} siap.")

    elif 'picamera' in cam_source:
        from picamera2 import Picamera2
        cam_type = 'picamera'
        cam = Picamera2()
        cam.configure(cam.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
        cam.start()
        print(f"Picamera siap dengan resolusi {resW}x{resH}.")

    else:
        print("cam_source tidak valid!")
        sys.exit()

except Exception as e:
    print(f"ERROR saat inisialisasi kamera: {e}")
    sys.exit()

# Variabel FPS
frame_rate_buffer = []
fps_avg_len = 30

print("\nMulai deteksi. Tekan 'q' untuk keluar.")

# Loop utama
while True:
    t_start = time.perf_counter()
    current_time = datetime.datetime.now()

    # Ambil frame
    if cam_type == 'usb':
        ret, frame = cam.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break
    else:
        frame_bgra = cam.capture_array()
        frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

    # Deteksi YOLO (stream=True untuk percepat)
    results = model(frame, verbose=False, stream=False)
    detections = results[0].boxes

    human_detected = False
    detection_info_list = []

    for det in detections:
        xmin, ymin, xmax, ymax = map(int, det.xyxy[0])
        conf = det.conf.item()
        classidx = int(det.cls.item())
        classname = labels[classidx]

        if conf > min_thresh and classname == "person":
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), bbox_color_person, 2)
            label = f"{classname}: {int(conf*100)}%"
            cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color_person, 1)
            human_detected = True
            detection_info_list.append(f"{classname} ({int(conf*100)}%)")

    # Status
    status_text = "Status: Terdeteksi" if human_detected else "Status: Tidak Terdeteksi"
    cv2.putText(frame, status_text, (20, resH - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,255,0) if human_detected else (0,0,255), 2)

    # Timestamp
    timestamp_text = current_time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Timestamp: {timestamp_text}", (20, resH - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Akurasi
    y_offset = 90
    for info in detection_info_list:
        cv2.putText(frame, f"Akurasi: {info}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        y_offset += 30

    # FPS
    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)
    cv2.putText(frame, f"FPS: {avg_frame_rate:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    # Tampilkan
    cv2.imshow("YOLOv8n NCNN - Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
if cam_type == 'usb':
    cam.release()
else:
    cam.stop()
cv2.destroyAllWindows()
print("Program selesai.")
