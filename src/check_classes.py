
from collections import Counter
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
labels_dir = project_root / "data/raw/neu-det/labels/train"

class_names = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}

counts = Counter()

for label_file in labels_dir.glob("*.txt"):
    for line in label_file.read_text().splitlines():
        parts = line.split()

        if parts:
            class_id = int(parts[0])
            counts[class_id] += 1

print("Training instances per class:")

for class_id, name in class_names.items():
    print(f"{name:20s}: {counts[class_id]}")
