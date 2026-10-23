from ultralytics import YOLO
from pathlib import Path
import csv

model = YOLO(r"runs/detect/runs/detect/hyacinth_bg_v2/weights/best.pt")
files = [Path(x.strip()) for x in open("background_test_list.txt", encoding="utf-8") if x.strip()]

out_file = Path("background_detection_report.csv")

with out_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "detections", "confidence_scores", "review_status"])

    detected_images = 0
    total_boxes = 0

    for i, path in enumerate(files, 1):
        result = model.predict(source=str(path), conf=0.50, verbose=False)[0]
        scores = [round(float(box.conf[0]), 3) for box in result.boxes]
        count = len(scores)

        total_boxes += count
        if count:
            detected_images += 1

        writer.writerow([
            path.name,
            count,
            "; ".join(map(str, scores)),
            "REVIEW" if count else "NO_DETECTION"
        ])
        print(f"{i}/{len(files)}: {path.name} -> {count} detections, confidence={scores}")

print("\nImages tested:", len(files))
print("Images with detections:", detected_images)
print("Images without detections:", len(files) - detected_images)
print("Total detection boxes:", total_boxes)
print("CSV report:", out_file.resolve())
