import cv2
from ultralytics import YOLO

print("Loading models...")
HYACINTH_MODEL = YOLO("runs/detect/runs/detect/hyacinth_bg_v2/weights/best.pt")
PERSON_MODEL = YOLO("yolo11n.pt")

CONFIDENCE = 0.30
PERSON_CONFIDENCE = 0.30

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Camera open avvaledu.")

print("Live detection started. Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Water hyacinth detection - every frame
        plant_result = HYACINTH_MODEL.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=640,
            iou=0.45,
            max_det=5,
            verbose=False
        )[0]

        output = plant_result.plot()
        plant_count = len(plant_result.boxes)
        print("Plant detections:", plant_count, "| Confidence:", [round(float(b.conf[0]), 3) for b in plant_result.boxes] if plant_count else [])

        # Person detection - same frame
        person_result = PERSON_MODEL.predict(
            source=frame,
            classes=[0],
            conf=PERSON_CONFIDENCE,
            imgsz=640,
            verbose=False
        )[0]

        for box in person_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                output, "person", (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        cv2.imshow("PELVYN AquaInspect - Plant + Person Detection", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()









