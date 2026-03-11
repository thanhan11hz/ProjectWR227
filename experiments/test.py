import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project_root
sys.path.append(str(ROOT))

import json
from ultralytics import YOLO

WEIGHTS = ROOT + "outputs/exp_baseline/weights/best.pt"
DATASET = ROOT + "configs/dataset.yaml"


def evaluate():

    model = YOLO(WEIGHTS)

    metrics = model.val(
        data=DATASET,
        split="test",
        save_json=True
    )

    results = {
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "mAP75": metrics.box.map75,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr
    }

    with open(ROOT+"outputs/metrics/results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(results)


if __name__ == "__main__":
    evaluate()