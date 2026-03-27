import os
os.environ["OMP_NUM_THREADS"] = "4"

import cv2
import threading
import time
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

# -------------------------------
# Load OpenVINO Model
# -------------------------------
try:
    model = YOLO("best_openvino_model")
    MODEL_STATUS = "OpenVINO Model Loaded"
except:
    MODEL_STATUS = "Model Load Failed"

# -------------------------------
# GUI Setup
# -------------------------------
root = Tk()
root.title("Weed Detection (OpenVINO Optimized)")
root.geometry("1100x750")
root.configure(bg="#202124")

cap = None
running = False
frame_count = 0

# -------------------------------
# Show Frame
# -------------------------------
def show_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.resize((850, 520))
    img_tk = ImageTk.PhotoImage(img)

    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

# -------------------------------
# Process Image (UPLOAD)
# -------------------------------
def process_image(frame):

    conf = confidence_slider.get()

    results = model(frame, imgsz=416, conf=conf, verbose=False)
    annotated = results[0].plot()

    count = len(results[0].boxes)
    count_label.config(text=f"Weeds: {count}")

    show_frame(annotated)

    status_bar.config(text="Image Processed")

# -------------------------------
# Upload Image
# -------------------------------
def upload_image():

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if not file_path:
        return

    frame = cv2.imread(file_path)

    if frame is None:
        status_bar.config(text="Error loading image")
        return

    process_image(frame)

# -------------------------------
# Camera Loop
# -------------------------------
def camera_loop():
    global frame_count

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip alternate frames
        if frame_count % 2 != 0:
            continue

        start = time.time()

        conf = confidence_slider.get()
        results = model(frame, imgsz=416, conf=conf, verbose=False)

        annotated = results[0].plot()

        count = len(results[0].boxes)
        count_label.config(text=f"Weeds: {count}")

        fps = 1 / (time.time() - start)
        fps_label.config(text=f"FPS: {int(fps)}")

        show_frame(annotated)

# -------------------------------
# Start Camera
# -------------------------------
def start_camera():
    global cap, running

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    running = True
    status_bar.config(text="Camera Running")

    threading.Thread(target=camera_loop, daemon=True).start()

# -------------------------------
# Stop Camera
# -------------------------------
def stop_camera():
    global running

    running = False
    if cap:
        cap.release()

    status_bar.config(text="Camera Stopped")

# -------------------------------
# UI Components
# -------------------------------

title = Label(root,
              text="Weed Detection System (OpenVINO)",
              font=("Arial", 22, "bold"),
              bg="#202124",
              fg="white")
title.pack(pady=10)

video_label = Label(root, bg="black")
video_label.pack(pady=10)

button_frame = Frame(root, bg="#202124")
button_frame.pack(pady=10)

upload_btn = Button(button_frame,
                    text="Upload Image",
                    command=upload_image,
                    width=15,
                    bg="#4CAF50",
                    fg="white",
                    font=("Arial",12))
upload_btn.grid(row=0,column=0,padx=10)

start_btn = Button(button_frame,
                   text="Start Camera",
                   command=start_camera,
                   width=15,
                   bg="#2196F3",
                   fg="white",
                   font=("Arial",12))
start_btn.grid(row=0,column=1,padx=10)

stop_btn = Button(button_frame,
                  text="Stop Camera",
                  command=stop_camera,
                  width=15,
                  bg="#f44336",
                  fg="white",
                  font=("Arial",12))
stop_btn.grid(row=0,column=2,padx=10)

# Confidence slider
confidence_slider = Scale(root,
                          from_=0.1,
                          to=1.0,
                          resolution=0.05,
                          orient=HORIZONTAL,
                          length=400,
                          label="Confidence Threshold",
                          bg="#202124",
                          fg="white",
                          highlightbackground="#202124")
confidence_slider.set(0.5)
confidence_slider.pack(pady=10)

# Info labels
info_frame = Frame(root, bg="#202124")
info_frame.pack()

count_label = Label(info_frame,
                    text="Weeds: 0",
                    font=("Arial",14),
                    bg="#202124",
                    fg="white")
count_label.grid(row=0,column=0,padx=20)

fps_label = Label(info_frame,
                  text="FPS: 0",
                  font=("Arial",14),
                  bg="#202124",
                  fg="white")
fps_label.grid(row=0,column=1,padx=20)

# Status bar
status_bar = Label(root,
                   text=MODEL_STATUS,
                   bd=1,
                   relief=SUNKEN,
                   anchor=W,
                   bg="#303134",
                   fg="white")
status_bar.pack(side=BOTTOM, fill=X)

root.mainloop()