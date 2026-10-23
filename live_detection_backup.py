import cv2
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
CONFIDENCE = 0.50

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera open avvaledu. Camera index or permissions check cheyyi.")

print("Live detection started. Quit cheyyadaniki Q press cheyyi.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera frame read avvaledu.")
            break

        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            iou=0.45,
            max_det=5,
            verbose=False
        )

        annotated_frame = results[0].plot()
        cv2.imshow("PELVYN AquaInspect - Live Water Hyacinth Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

