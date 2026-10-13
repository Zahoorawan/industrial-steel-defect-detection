
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "industrial_defect"
    / "yolo26n_neu_det"
    / "weights"
    / "best.pt"
)

st.set_page_config(
    page_title="Steel Defect Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("Steel Surface Defect Detection")
st.write(
    "Upload a steel surface image to detect and locate possible defects."
)

st.caption(
    "Model: YOLO26n | Classes: 6 steel surface defect categories"
)


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a steel surface image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

confidence = st.slider(
    "Detection confidence threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("Detect Defects", type="primary"):
        with st.spinner("Analyzing the image..."):
            model = load_model()
            results = model.predict(
                source=image,
                imgsz=640,
                conf=confidence,
                device=0,
                verbose=False,
            )

        result = results[0]
        annotated_image = result.plot()

        # Ultralytics returns a BGR image; Streamlit displays RGB.
        annotated_image = annotated_image[:, :, ::-1]

        with right:
            st.subheader("Detection Results")
            st.image(annotated_image, use_container_width=True)

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            st.warning(
                "No defects were detected above this confidence threshold."
            )
        else:
            st.subheader("Detected Defects")

            detections = []
            for box in boxes:
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                score = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    {
                        "Defect": class_name,
                        "Confidence": f"{score:.1%}",
                        "Box (x1, y1, x2, y2)": (
                            f"({x1:.0f}, {y1:.0f}, "
                            f"{x2:.0f}, {y2:.0f})"
                        ),
                    }
                )

            st.dataframe(
                detections,
                use_container_width=True,
                hide_index=True,
            )

            st.success(f"Detected {len(detections)} possible defect(s).")

st.divider()
st.caption(
    "Educational portfolio project. Predictions can be incorrect "
    "and should not be treated as proof of defect-free steel."
)
