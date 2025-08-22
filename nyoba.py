import os
import sys
import time
import datetime
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='ncnn')

# User-defined parameters
model_path = 'yolov8n_ncnn_model'   # Path ke folder model YOLOv8n NCNN
cam_source = 'usb0'                 # 'usb0' untuk USB camera, 'picamera0' untuk Picamera
min_thresh = 0.5                    # Ambang confidence minimum
resW, resH = 1280, 720              # Resolusi kamera

# Area fokus deteksi manusia (dapat diabaikan jika tidak diperlukan)
pbox_xmin, pbox_ymin = 540, 160
pbox_xmax, pbox_ymax = 760, 450

# Warna bounding box (Tableau 10)
bbox_color_person = (0, 0, 255)

# Load YOLOv8n NCNN model
if not os.path.exists(model_path):
    print("ERROR: Model path tidak ditemukan!")
    sys.exit()

try:
    model = YOLO(model_path, task='detect')
    labels = model.names
    print(f"Model YOLOv8n NCNN berhasil dimuat. Kelas: {labels}")
except Exception as e:
    print(f"ERROR saat memuat model: {e}")
    sys.exit()

# Inisialisasi Kamera
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
        print("cam_source tidak valid! Gunakan 'usb0' atau 'picamera0'.")
        sys.exit()

except Exception as e:
    print(f"ERROR saat inisialisasi kamera: {e}")
    sys.exit()

# Variabel FPS
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 50

print("\nMulai deteksi. Tekan 'q' untuk keluar, 'p' untuk simpan screenshot.")

# Loop Utama
while True:
    t_start = time.perf_counter()
    current_time = datetime.datetime.now()

    # Ambil frame
    if cam_type == 'usb':
        ret, frame = cam.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break
    elif cam_type == 'picamera':
        frame_bgra = cam.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)

    # Jalankan deteksi YOLO
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Status deteksi manusia
    human_detected = False
    detection_info_list = []

    # Loop hasil deteksi
    for det in detections:
        xmin, ymin, xmax, ymax = map(int, det.xyxy[0].cpu().numpy())
        conf = det.conf.item()
        classidx = int(det.cls.item())
        classname = labels[classidx]

        if conf > min_thresh:
            if classname == "person":
                color = bbox_color_person
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f"{classname}: {int(conf*100)}%"
                cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                human_detected = True
                detection_info_list.append(f"{classname} ({int(conf*100)}%)")

    # Tampilkan status
    status_text = "Status: Terdeteksi" if human_detected else "Status: Tidak Terdeteksi"
    cv2.putText(frame, status_text, (20, resH - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,255,0) if human_detected else (0,0,255), 2)

    # Timestamp
    timestamp_text = current_time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Timestamp: {timestamp_text}", (20, resH - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Akurasi tiap deteksi manusia
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

    # Tampilkan hasil
    cv2.imshow("YOLOv8n NCNN - Human Detection", frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        filename = f"screenshot_{current_time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot tersimpan: {filename}")

# Cleanup
if cam_type == 'usb' and cam is not None:
    cam.release()
elif cam_type == 'picamera' and cam is not None:
    cam.stop()
cv2.destroyAllWindows()
print("Program selesai.")