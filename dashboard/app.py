from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best.pt"

st.set_page_config(
    page_title="PELVYN AquaInspect Mini V1",
    page_icon="??",
    layout="wide",
)

st.title("PELVYN AquaInspect Mini V1")
st.subheader("AI-Based Water Hyacinth Detection")
st.write("Upload a water hyacinth image to test the trained AI model.")


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

confidence = st.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.05,
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    with st.spinner("Running AI detection..."):
        model = load_model()
        result = model.predict(
            source=image,
            conf=confidence,
            iou=0.45,
            max_det=5,
            verbose=False,
        )[0]
    annotated_image = result.plot()[:, :, ::-1]
    count = len(result.boxes)

    left, right = st.columns(2)

    with left:
        st.image(image, caption="Original Image", use_container_width=True)

    with right:
        st.image(
            annotated_image,
            caption="Detection Result",
            use_container_width=True,
        )

    st.metric("Detected Objects", count)

    for index, box in enumerate(result.boxes, start=1):
        score = float(box.conf[0])
        st.write(
            f"Object {index}: water_hyacinth | Confidence: {score:.1%}"
        )
    if count == 0:
        st.warning("No water hyacinth detected in this image.")


