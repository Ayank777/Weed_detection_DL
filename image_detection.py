import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog

# Hide root tkinter window
Tk().withdraw()

# Open file dialog
image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    print("❌ No file selected.")
    exit()

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Read image
frame = cv2.imread(image_path)

# Run detection
results = model(frame, verbose=False)

# Draw results
annotated_frame = results[0].plot()

# Show output
cv2.imshow("Weed Detection Result", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()