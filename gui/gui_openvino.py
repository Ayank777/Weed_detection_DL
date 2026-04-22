import os
os.environ["OMP_NUM_THREADS"] = "4"

import cv2
import threading
import time
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# -------------------------------
# Load OpenVINO Model (SAFE LOAD)
# -------------------------------
model = None

try:
    model = YOLO("models/best_openvino_model", task="detect")
    MODEL_STATUS = "✅ OpenVINO Model Loaded"
except Exception as e:
    MODEL_STATUS = f"❌ Model Load Failed"

# -------------------------------
# GUI Setup
# -------------------------------
root = Tk()
root.title("Weed Detection (OpenVINO Optimized)")
root.geometry("1100x750")

cap = None
running = False
frame_count = 0
current_frame = None

# -------------------------------
# COLORS
# -------------------------------
BG_COLOR = "#1e1e1e"
PANEL_COLOR = "#2a2a2a"
ACCENT = "#00adb5"
TEXT_COLOR = "#eeeeee"

root.configure(bg=BG_COLOR)

# -------------------------------
# Show Frame
# -------------------------------
def show_frame(frame):
    global current_frame

    current_frame = frame.copy()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.resize((850, 520))
    img_tk = ImageTk.PhotoImage(img)

    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

# -------------------------------
# Process Image
# -------------------------------
def process_image(frame):

    if model is None:
        messagebox.showerror("Error", "Model not loaded!")
        return

    conf = confidence_slider.get()

    start = time.time()
    results = model(frame, imgsz=640, conf=conf, verbose=False)
    annotated = results[0].plot()
    fps = 1 / (time.time() - start)

    count = len(results[0].boxes)

    count_label.config(text=f"Weeds: {count}")
    fps_label.config(text=f"FPS: {int(fps)}")

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
fps_avg = 0
prev_boxes = []

def camera_loop():
    global frame_count, fps_avg, prev_boxes

    if model is None:
        status_bar.config(text="Model not loaded!")
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()

        # Improve quality
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

        # Resize for model
        frame_resized = cv2.resize(frame, (640, 640))

        conf = confidence_slider.get()

        results = model(frame_resized, imgsz=640, conf=conf, verbose=False)

        # Stabilize detections
        if len(results[0].boxes) > 0:
            prev_boxes = results[0].boxes
        else:
            results[0].boxes = prev_boxes

        annotated = results[0].plot()

        count = len(results[0].boxes)

        # Smooth FPS
        current_fps = 1 / (time.time() - start)
        fps_avg = (fps_avg * 0.9) + (current_fps * 0.1)

        count_label.config(text=f"Weeds: {count}")
        fps_label.config(text=f"FPS: {int(fps_avg)}")

        annotated = cv2.resize(annotated, (850, 520))
        show_frame(annotated)

        time.sleep(0.03)
# -------------------------------
# Start Camera
# -------------------------------
def start_camera():
    global cap, running

    if model is None:
        messagebox.showerror("Error", "Model not loaded!")
        return

    cap = cv2.VideoCapture(1)
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
# Save Image
# -------------------------------
def save_image():

    if current_frame is None:
        messagebox.showwarning("Warning", "No result to save!")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
    )

    if not file_path:
        return

    cv2.imwrite(file_path, current_frame)
    status_bar.config(text="Image Saved Successfully")

# -------------------------------
# UI COMPONENTS
# -------------------------------

# Title
title = Label(root,
              text="🌿 Weed Detection System",
              font=("Segoe UI", 24, "bold"),
              bg=BG_COLOR,
              fg=ACCENT)
title.pack(pady=10)

# Main Frame
main_frame = Frame(root, bg=BG_COLOR)
main_frame.pack(fill=BOTH, expand=True)

# Left Panel
left_panel = Frame(main_frame, bg=PANEL_COLOR, width=250)
left_panel.pack(side=LEFT, fill=Y, padx=10, pady=10)

# Right Panel
right_panel = Frame(main_frame, bg=BG_COLOR)
right_panel.pack(side=RIGHT, expand=True, fill=BOTH)

video_label = Label(right_panel, bg="black")
video_label.pack(padx=10, pady=10)

# Button Style
def styled_button(parent, text, command):
    return Button(parent,
                  text=text,
                  command=command,
                  width=18,
                  height=2,
                  bg=ACCENT,
                  fg="black",
                  font=("Segoe UI", 10, "bold"),
                  bd=0,
                  activebackground="#00cfd6")

# Controls Title
Label(left_panel,
      text="Controls",
      bg=PANEL_COLOR,
      fg=TEXT_COLOR,
      font=("Segoe UI", 14, "bold")).pack(pady=10)

# Buttons
styled_button(left_panel, "📁 Upload Image", upload_image).pack(pady=8)
styled_button(left_panel, "📷 Start Camera", start_camera).pack(pady=8)
styled_button(left_panel, "⛔ Stop Camera", stop_camera).pack(pady=8)
styled_button(left_panel, "💾 Save Result", save_image).pack(pady=8)

# Confidence Slider
Label(left_panel,
      text="Confidence",
      bg=PANEL_COLOR,
      fg=TEXT_COLOR,
      font=("Segoe UI", 12)).pack(pady=15)

confidence_slider = Scale(left_panel,
                          from_=0.1,
                          to=1.0,
                          resolution=0.05,
                          orient=HORIZONTAL,
                          bg=PANEL_COLOR,
                          fg=TEXT_COLOR,
                          highlightbackground=PANEL_COLOR,
                          length=200)
confidence_slider.set(0.5)
confidence_slider.pack()

# Info Bar
info_frame = Frame(root, bg=PANEL_COLOR)
info_frame.pack(fill=X)

count_label = Label(info_frame,
                    text="Weeds: 0",
                    font=("Segoe UI", 12),
                    bg=PANEL_COLOR,
                    fg=TEXT_COLOR)
count_label.pack(side=LEFT, padx=20, pady=5)

fps_label = Label(info_frame,
                  text="FPS: 0",
                  font=("Segoe UI", 12),
                  bg=PANEL_COLOR,
                  fg=TEXT_COLOR)
fps_label.pack(side=LEFT, padx=20)

status_bar = Label(info_frame,
                   text=MODEL_STATUS,
                   font=("Segoe UI", 10),
                   bg=PANEL_COLOR,
                   fg=ACCENT)
status_bar.pack(side=RIGHT, padx=20)

# Run App
root.mainloop()