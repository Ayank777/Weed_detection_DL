import cv2
import threading
import time
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Create main window
root = Tk()
root.title("Weed Detection System - YOLOv8")
root.geometry("1000x700")
root.configure(bg="#1e1e1e")

# Global variables
cap = None
running = False

# -------------------------------
# Function: Upload Image
# -------------------------------
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    frame = cv2.imread(file_path)
    process_frame(frame)

# -------------------------------
# Function: Start Webcam
# -------------------------------
def start_camera():
    global cap, running
    cap = cv2.VideoCapture(0)
    running = True
    threading.Thread(target=update_camera, daemon=True).start()

# -------------------------------
# Function: Stop Webcam
# -------------------------------
def stop_camera():
    global running
    running = False
    if cap:
        cap.release()

# -------------------------------
# Process Frame
# -------------------------------
def process_frame(frame):
    conf_value = confidence_slider.get()

    results = model(frame, conf=conf_value, verbose=False)
    annotated_frame = results[0].plot()

    # Count detections
    detection_count = len(results[0].boxes)
    count_label.config(text=f"Detections: {detection_count}")

    show_frame(annotated_frame)

# -------------------------------
# Show Frame in GUI
# -------------------------------
def show_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.resize((800, 500))
    img_tk = ImageTk.PhotoImage(img)

    video_label.img_tk = img_tk
    video_label.config(image=img_tk)

# -------------------------------
# Update Webcam Feed
# -------------------------------
def update_camera():
    global running
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        process_frame(frame)
        fps = 1 / (time.time() - start_time)

        fps_label.config(text=f"FPS: {fps:.2f}")

# -------------------------------
# UI Components
# -------------------------------

title_label = Label(root, text="Weed Detection using YOLOv8",
                    font=("Arial", 22, "bold"),
                    bg="#1e1e1e", fg="white")
title_label.pack(pady=10)

video_label = Label(root, bg="black")
video_label.pack(pady=10)

button_frame = Frame(root, bg="#1e1e1e")
button_frame.pack(pady=10)

upload_btn = Button(button_frame, text="Upload Image",
                    command=upload_image,
                    width=15, bg="#4CAF50", fg="white", font=("Arial", 12))
upload_btn.grid(row=0, column=0, padx=10)

start_btn = Button(button_frame, text="Start Camera",
                   command=start_camera,
                   width=15, bg="#2196F3", fg="white", font=("Arial", 12))
start_btn.grid(row=0, column=1, padx=10)

stop_btn = Button(button_frame, text="Stop Camera",
                  command=stop_camera,
                  width=15, bg="#f44336", fg="white", font=("Arial", 12))
stop_btn.grid(row=0, column=2, padx=10)

confidence_slider = Scale(root, from_=0.1, to=1.0,
                          resolution=0.05,
                          orient=HORIZONTAL,
                          length=400,
                          label="Confidence Threshold",
                          bg="#1e1e1e", fg="white",
                          highlightbackground="#1e1e1e")
confidence_slider.set(0.5)
confidence_slider.pack(pady=10)

count_label = Label(root, text="Detections: 0",
                    font=("Arial", 14),
                    bg="#1e1e1e", fg="white")
count_label.pack()

fps_label = Label(root, text="FPS: 0",
                  font=("Arial", 14),
                  bg="#1e1e1e", fg="white")
fps_label.pack()

root.mainloop()