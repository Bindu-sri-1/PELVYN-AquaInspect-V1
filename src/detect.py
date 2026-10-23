from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path("models/best.pt")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Place your trained water hyacinth model at this path."
        )
    return YOLO(str(MODEL_PATH))


def detect_image(image_path: str, confidence: float = 0.25):
    model = load_model()
    results = model.predict(
        source=image_path,
        conf=confidence,
        save=True
    )
    return results


if __name__ == "__main__":
    print("PELVYN AquaInspect - AI Detection Module")
    print(f"Expected trained model: {MODEL_PATH}")
    print("Add your trained best.pt model before running detection.")
