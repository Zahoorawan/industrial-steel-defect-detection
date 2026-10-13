
from pathlib import Path
from ultralytics import YOLO

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Trained model
MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "industrial_defect"
    / "yolo26n_neu_det"
    / "weights"
    / "best.pt"
)

# Folder containing your test images
IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "neu-det"
    / "images"
    / "test"
)

# Folder for saved predictions
OUTPUT_DIR = PROJECT_ROOT / "runs" / "custom_predictions"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image folder not found: {IMAGE_DIR}"
        )

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(IMAGE_DIR),
        imgsz=640,
        conf=0.25,
        device=0,
        save=True,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True
    )

    print(f"\nProcessed {len(results)} images.")
    print(f"Predictions saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
