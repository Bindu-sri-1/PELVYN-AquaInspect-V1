from ultralytics import YOLO
from pathlib import Path

model = YOLO(r"runs/detect/runs/detect/hyacinth_bg_v2/weights/best.pt")
files = [Path(x.strip()) for x in open("background_test_list.txt", encoding="utf-8") if x.strip()]
out = Path("false_positive_review")
out.mkdir(exist_ok=True)

for path in files:
    result = model.predict(source=str(path), conf=0.30, verbose=False)[0]
    if len(result.boxes):
        result.save(filename=str(out / path.name))

print("Review images saved to:", out.resolve())
