import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "ultralytics"))
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
import yaml
from pathlib import Path

def create_dataset_yaml(root_dir, save_path="configs/dataset.yaml"):

    data = {
        "path": str(root_dir),   # ✅ convert sang string
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "license_plate"}
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.safe_dump(data, f)   # dùng safe_dump luôn

    return save_path

DATA_ROOT = ROOT / "data/"
MODEL_WEIGHTS = "yolov8s.pt"   # pretrained weights
MODEL_CFG = ROOT / "ultralytics/ultralytics/cfg/models/v8/yolov8s.yaml"

EPOCHS = 35
BATCH = 32
IMG_SIZE = 640

EXP_NAME = "baseline+augmentation"
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
        workers=8,
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

    print("Training completed")

if __name__ == "__main__":
    train()