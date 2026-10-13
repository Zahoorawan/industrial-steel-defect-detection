
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    # 1. Check GPU availability
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("GPU not available. Using CPU.")
        device = "cpu"

    # 2. Locate the dataset configuration
    project_root = Path(__file__).resolve().parent.parent
    data_yaml = project_root / "data" / "raw" / "neu-det" / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data_yaml}"
        )

    print("Dataset configuration:", data_yaml)

    # 3. Load the pretrained YOLO26n model
    model = YOLO("yolo26n.pt")

    # 4. Train the model
    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=4,
        device=device,
        workers=0,
        project=str(project_root / "runs" / "industrial_defect"),
        name="yolo26n_neu_det",
        exist_ok=True
    )

    print("\nTraining completed!")
    print("Training results:", results)

    # 5. Check for the best trained model
    best_model_path = (
        project_root
        / "runs"
        / "industrial_defect"
        / "yolo26n_neu_det"
        / "weights"
        / "best.pt"
    )

    if best_model_path.exists():
        print("Best model saved at:", best_model_path)
    else:
        print("Best model checkpoint was not found.")


if __name__ == "__main__":
    main()
