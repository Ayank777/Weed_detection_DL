import json
import os

def convert_coco(json_path, label_dir):
    with open(json_path, 'r') as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        bbox = ann["bbox"]
        category_id = ann["category_id"] - 1  # YOLO starts at 0

        img = images[image_id]
        w, h = img["width"], img["height"]

        x, y, bw, bh = bbox
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        bw /= w
        bh /= h

        label_file = os.path.join(
            label_dir,
            img["file_name"].replace(".jpg", ".txt")
        )

        with open(label_file, "a") as f:
            f.write(f"{category_id} {x_center} {y_center} {bw} {bh}\n")

# -------- TRAIN --------
convert_coco(
    "../data/train/_annotations.coco.json",
    "../data/labels/train"
)

# -------- VALIDATION --------
convert_coco(
    "../data/val/_annotations.coco.json",
    "../data/labels/val"
)

print("COCO to YOLO conversion completed!")