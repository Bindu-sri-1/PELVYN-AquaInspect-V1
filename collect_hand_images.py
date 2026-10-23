import cv2
from pathlib import Path

SAVE_DIR = Path("dataset_hyacinth_bg_v1/train/images/hand_negative")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera open avvaledu")

count = 0
print("Hand collection started.")
print("SPACE = save image | Q = quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"Saved: {count}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Hand Negative Image Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            path = SAVE_DIR / f"hand_{count:03d}.jpg"
            if cv2.imwrite(str(path), frame):
                count += 1
                print("Saved:", path)

        elif key == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

print(f"Total images saved: {count}")
