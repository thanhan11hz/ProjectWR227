import sys
from pathlib import Path
import json
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "ultralytics"))
sys.path.insert(0, str(ROOT))

WEIGHTS = ROOT / "outputs/baseline+augmentation+se/weights/best.pt"

DATA_ROOT = ROOT / "dataset"
IMAGES_TEST = DATA_ROOT / "images/test"
LABELS_TEST = DATA_ROOT / "labels/test"

OUTPUT = ROOT / "outputs/metrics"
OUTPUT.mkdir(parents=True, exist_ok=True)


def create_dataset_yaml(img_dir, yaml_path):

    data = {
        "path": str(DATA_ROOT),
        "train": "",
        "val": "",
        "test": str(img_dir.relative_to(DATA_ROOT)),
        "names": {
            0: "license_plate"
        }
    }

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f)


def evaluate():

    model = YOLO(WEIGHTS)

    results_all = {}

    subfolders = [f for f in IMAGES_TEST.iterdir() if f.is_dir()]

    for folder in subfolders:

        print(f"\nEvaluating corruption: {folder.name}")

        yaml_path = ROOT / "configs/temp_test.yaml"

        create_dataset_yaml(folder, yaml_path)

        metrics = model.val(
            data=yaml_path,
            split="test",
            save_json=True,
            plots=False
        )

        result = {
            "mAP50": float(metrics.box.map50),
            "mAP75": float(metrics.box.map75),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "fitness": float(metrics.fitness),
            "per_class_mAP": [float(x) for x in metrics.box.maps]
        }

        results_all[folder.name] = result

        print(result)

    with open(OUTPUT / "results_by_corruption.json", "w") as f:
        json.dump(results_all, f, indent=4)


if __name__ == "__main__":
    evaluate()