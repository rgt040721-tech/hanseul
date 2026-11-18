from ultralytics import YOLO
from picamera2 import Picamera2
import cv2, time, json
import numpy as np

model = YOLO('best.tflite')
with open('zone.json') as f:
    tent = np.array(json.load(f)['tent'], dtype=np.int32)

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
cam.start()
time.sleep(2)
frame = cam.capture_array()
cam.stop()

r = model(frame, classes=[0], iou=0.8, verbose=False)[0]
print("detected:", len(r.boxes))

cv2.imwrite("snap_raw.jpg", frame)

img = r.plot()
cv2.polylines(img, [tent], True, (0, 255, 255), 3)
for x, y in tent:
    cv2.putText(img, f"({x},{y})", (int(x) + 5, int(y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
for x in range(0, 641, 80):
    cv2.line(img, (x, 0), (x, 480), (200, 200, 200), 1)
    cv2.putText(img, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
for y in range(0, 481, 80):
    cv2.line(img, (0, y), (640, y), (200, 200, 200), 1)
    cv2.putText(img, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
cv2.imwrite("snap_boxes.jpg", img)
print("saved snap_raw.jpg, snap_boxes.jpg")
