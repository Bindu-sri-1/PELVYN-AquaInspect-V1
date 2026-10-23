import cv2
from pathlib import Path

SAVE_DIR = Path("plant_collection")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera open avvaledu")

count = len(list(SAVE_DIR.glob("plant_*.jpg")))

print("Plant image collection started")
print("SPACE = save image | Q = quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"Saved: {count}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Water Hyacinth Image Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            path = SAVE_DIR / f"plant_{count:03d}.jpg"
            if cv2.imwrite(str(path), frame):
                count += 1
                print("Saved:", path)

        elif key == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

print("Total plant images:", count)
