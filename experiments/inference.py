import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "ultralytics"))
sys.path.insert(0, str(ROOT))

import cv2
import torch
import numpy as np
from ultralytics import YOLO


WEIGHTS = ROOT / "outputs/baseline/weights/best.pt"

IMAGE_LIST = [
    "test/normal/026724137931-89_95-240&487_489&600-474&584_240&600_258&497_492&481-0_0_14_8_32_26_33-158-24.jpg",
    "test/db/0044-1_1-291&497_392&534-392&532_291&534_291&499_392&497-0_0_6_17_31_33_31-11-5.jpg",
    "test/blur/0090-2_1-344&537_484&591-481&591_344&585_347&537_484&543-0_0_32_10_24_27_32-74-2.jpg",
    "test/fn/0019-1_1-340&500_404&526-404&524_340&526_340&502_404&500-0_0_11_26_25_28_17-66-3.jpg",
    "test/tilt/0357-40_36-298&272_438&485-431&485_298&371_305&272_438&386-0_0_3_27_24_25_23-79-184.jpg",
]

OUTPUT_DIR = ROOT / "outputs/attention_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- YOLO-style letterbox ----------
def letterbox(img, new_shape=640, color=(114,114,114)):

    shape = img.shape[:2]

    r = min(new_shape / shape[0], new_shape / shape[1])

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]

    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return img


# ---------- Eigen-CAM ----------
def generate_eigen_cam(model, img_tensor, target_layer):

    activations = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    hook = target_layer.register_forward_hook(forward_hook)

    with torch.no_grad():
        _ = model(img_tensor)

    feature = activations[0]  # shape: [1, C, H, W]

    feature = feature.squeeze(0)

    C, H, W = feature.shape

    reshaped = feature.view(C, -1).cpu().numpy()

    # PCA
    reshaped = reshaped - reshaped.mean(axis=1, keepdims=True)

    U, S, Vt = np.linalg.svd(reshaped, full_matrices=False)

    cam = Vt[0].reshape(H, W)

    cam = np.maximum(cam, 0)

    hook.remove()

    return cam


# ---------- PROCESS IMAGE ----------
def process_image(model, image_path):

    img0 = cv2.imread(str(image_path))

    img = letterbox(img0, 640)

    h, w = img.shape[:2]

    device = next(model.model.parameters()).device

    tensor = (
        torch.from_numpy(img)
        .permute(2,0,1)
        .unsqueeze(0)
        .float() / 255
    ).to(device)

    # Hook layer (backbone/neck)
    target_layer = model.model.model[12]

    cam = generate_eigen_cam(model.model, tensor, target_layer)

    cam = cv2.resize(cam, (w, h))

    cam -= cam.min()
    cam /= (cam.max() + 1e-6)

    cam = cam ** 2

    heatmap = cv2.applyColorMap(
        (cam * 255).astype("uint8"),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    name = Path(image_path).stem

    cv2.imwrite(str(OUTPUT_DIR / f"{name}_cam.jpg"), overlay)


# ---------- MAIN ----------
def inference():

    model = YOLO(WEIGHTS)

    model.model.eval()

    for image_path in IMAGE_LIST:

        print(f"Processing {image_path}")

        image_path = ROOT / ("data/images/" + image_path)

        model.predict(
            image_path,
            save=True,
            project=ROOT / "outputs/predictions"
        )

        process_image(model, image_path)

    print("All images processed")


if __name__ == "__main__":
    inference()