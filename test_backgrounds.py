from ultralytics import YOLO
from pathlib import Path

model = YOLO(r"runs/detect/runs/detect/hyacinth_bg_v2/weights/best.pt")
files = [Path(x.strip()) for x in open("background_test_list.txt", encoding="utf-8") if x.strip()]

detected = 0
total_boxes = 0

for i, path in enumerate(files, 1):
    result = model.predict(source=str(path), conf=0.30, verbose=False)[0]
    count = len(result.boxes)
    total_boxes += count
    detected += count > 0
    print(f"{i}/{len(files)}: {path.name} -> {count} detections")

print("\nTOTAL_IMAGES:", len(files))
print("IMAGES_WITH_DETECTIONS:", detected)
print("IMAGES_WITHOUT_DETECTIONS:", len(files) - detected)
print("TOTAL_DETECTION_BOXES:", total_boxes)
