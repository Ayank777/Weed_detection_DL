import os
os.environ["OMP_NUM_THREADS"] = "4"

from ultralytics import YOLO
import cv2
import time

model = YOLO("models/best_openvino_model")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

frame_count = 0

print("OpenVINO weed detection started")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 2 != 0:
        continue

    start = time.time()

    results = model(frame, imgsz=416, conf=0.50)

    annotated_frame = results[0].plot()

    fps = 1/(time.time()-start)

    cv2.putText(annotated_frame,
                f"FPS:{int(fps)}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                2)

    cv2.imshow("Weed Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()