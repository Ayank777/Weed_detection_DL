import cv2
import threading
import time
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# -------------------------------
# Load YOLO Model
# -------------------------------
try:
    model = YOLO("models/best.pt")
    MODEL_STATUS = "Model Loaded Successfully"
except:
    MODEL_STATUS = "Model Load Failed"

# -------------------------------
# GUI Window
# -------------------------------
root = Tk()
root.title("Weed Detection System (YOLOv8)")
root.geometry("1100x750")
root.configure(bg="#202124")

# Global variables
cap = None
running = False
current_frame = None

# -------------------------------
# Process Frame
# -------------------------------
def process_frame(frame):
    global current_frame

    conf = confidence_slider.get()

    results = model(frame, conf=conf, verbose=False)
    annotated = results[0].plot()

    # Detection count
    count = len(results[0].boxes)
    detection_label.config(text=f"Weeds Detected: {count}")

    current_frame = annotated
    show_frame(annotated)

# -------------------------------
# Display Frame
# -------------------------------
def show_frame(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.resize((850, 520))
    img_tk = ImageTk.PhotoImage(img)

    video_panel.imgtk = img_tk
    video_panel.configure(image=img_tk)

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
    process_frame(frame)

    status_bar.config(text="Image loaded and processed")

# -------------------------------
# Start Camera
# -------------------------------
def start_camera():
    global cap, running

    cap = cv2.VideoCapture(0)
    running = True

    status_bar.config(text="Camera Started")

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
# Camera Loop
# -------------------------------
def camera_loop():

    while running:
        ret, frame = cap.read()

        if not ret:
            break

        start = time.time()

        process_frame(frame)

        fps = 1/(time.time()-start)
        fps_label.config(text=f"FPS: {fps:.2f}")

# -------------------------------
# Save Result
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
    status_bar.config(text="Result image saved")

# -------------------------------
# UI COMPONENTS
# -------------------------------

title = Label(root,
              text="Weed Detection System using YOLOv8",
              font=("Arial", 22, "bold"),
              bg="#202124",
              fg="white")
title.pack(pady=10)

# Video display
video_panel = Label(root, bg="black")
video_panel.pack(pady=10)

# Control buttons
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

save_btn = Button(button_frame,
                  text="Save Result",
                  command=save_image,
                  width=15,
                  bg="#FF9800",
                  fg="white",
                  font=("Arial",12))
save_btn.grid(row=0,column=3,padx=10)

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

detection_label = Label(info_frame,
                        text="Weeds Detected: 0",
                        font=("Arial",14),
                        bg="#202124",
                        fg="white")
detection_label.grid(row=0,column=0,padx=20)

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