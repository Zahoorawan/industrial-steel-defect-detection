
from pathlib import Path
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parent.parent

    data_yaml = project_root / "data" / "raw" / "neu-det" / "data.yaml"

    best_model_path = (
        project_root
        / "runs"
        / "industrial_defect"
        / "yolo26n_neu_det"
        / "weights"
        / "best.pt"
    )

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {best_model_path}"
        )

    model = YOLO(str(best_model_path))

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        batch=4,
        device=0 if __import__("torch").cuda.is_available() else "cpu",
        workers=0
    )

    print("Test mAP50:", metrics.box.map50)
    print("Test mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
