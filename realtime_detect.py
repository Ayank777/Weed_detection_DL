import cv2
import time
from ultralytics import YOLO

# -------------------------------
# Load trained YOLOv8 model
# -------------------------------
model = YOLO("runs/detect/train/weights/best.pt")

# -------------------------------
# Open webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

print("✅ Webcam started successfully")
print("Press 'Q' to exit")

# -------------------------------
# Real-time detection loop
# -------------------------------
while True:
    start_time = time.time()  # Start FPS timer

    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Draw bounding boxes on frame
    annotated_frame = results[0].plot()

    # -------------------------------
    # Calculate FPS
    # -------------------------------
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Display FPS on screen
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show output window
    cv2.imshow("Weed Detection - YOLOv8", annotated_frame)

    # Exit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()