import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "ultralytics"))
sys.path.insert(0, str(ROOT))

import cv2
import torch
import numpy as np
from ultralytics import YOLO


WEIGHTS = ROOT / "outputs/baseline+augmentation/weights/best.pt"

IMAGE_LIST = [
    "test/normal/026724137931-89_95-240&487_489&600-474&584_240&600_258&497_492&481-0_0_14_8_32_26_33-158-24.jpg",
    "test/db/0044-1_1-291&497_392&534-392&532_291&534_291&499_392&497-0_0_6_17_31_33_31-11-5.jpg",
    "test/blur/0090-2_1-344&537_484&591-481&591_344&585_347&537_484&543-0_0_32_10_24_27_32-74-2.jpg"
    "test/fn/0019-1_1-340&500_404&526-404&524_340&526_340&502_404&500-0_0_11_26_25_28_17-66-3.jpg"
    "test/tilt/0357-40_36-298&272_438&485-431&485_298&371_305&272_438&386-0_0_3_27_24_25_23-79-184.jpg"
]

OUTPUT_DIR = ROOT / "outputs/attention_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_attention_map(model, img_tensor, target_layer):

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)

    score = output[0].max()
    score.backward()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)

    cam = (weights * act).sum(dim=1)
    cam = torch.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()

    h1.remove()
    h2.remove()

    return cam


def process_image(model, image_path):

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255

    # target layer (có thể thay đổi)
    target_layer = model.model.model[9]

    cam = generate_attention_map(model.model, tensor, target_layer)

    cam = cv2.resize(cam, (w, h))

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)

    heatmap = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    name = Path(image_path).stem

    cv2.imwrite(str(OUTPUT_DIR / f"{name}_cam.jpg"), overlay)


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