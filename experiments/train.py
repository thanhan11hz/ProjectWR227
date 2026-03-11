import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project_root
sys.path.append(str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace
import yaml
from pathlib import Path

def create_dataset_yaml(root_dir, save_path="configs/dataset.yaml"):
    data = {
        "path": root_dir,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "license_plate"}
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(data, f)

    return save_path

hyp = IterableSimpleNamespace(
    mosaic=0.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste = 0,

    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    flipud=0.0,
    fliplr=0.5,
    degrees = 5,
    translate = 0.05,
    scale = 0.1,
)

DATA_ROOT = ROOT / "data/"
MODEL_WEIGHTS = "yolov8s.pt"   # pretrained weights
MODEL_CFG = ROOT / "ultralytics/cfg/models/v8/yolov8.yaml"

EPOCHS = 100
BATCH = 16
IMG_SIZE = 640

EXP_NAME = "baseline"
SAVE_DIR = ROOT / "outputs"

def train():

    dataset_yaml = create_dataset_yaml(DATA_ROOT)

    model = YOLO(MODEL_CFG)

    model.load(MODEL_WEIGHTS)   # finetune từ pretrained

    results = model.train(
        data=dataset_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=SAVE_DIR,
        name=EXP_NAME,
        pretrained=True,
        save=True,
        exist_ok=True,
        hyp=hyp,
    )

    print("Training completed")

if __name__ == "__main__":
    train()