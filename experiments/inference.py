import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project_root
sys.path.append(str(ROOT))

import cv2
import torch
from ultralytics import YOLO
import numpy as np

IMAGE_PATH = "test.jpg"
WEIGHTS = ROOT / "outputs/baseline/weights/best.pt"


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

    weights = grad.mean(dim=(2,3), keepdim=True)

    cam = (weights * act).sum(dim=1)
    cam = torch.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()

    cam = cv2.resize(cam, (640,640))

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)

    h1.remove()
    h2.remove()

    return cam

def inference():

    model = YOLO(WEIGHTS)

    results = model.predict(
        IMAGE_PATH,
        save=True,
        project=ROOT / "outputs/predictions"
    )

    img = cv2.imread(IMAGE_PATH)

    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()/255

    target_layer = model.model.model[9]  # ví dụ stage P4

    cam = generate_attention_map(model.model, tensor, target_layer)

    heatmap = cv2.applyColorMap((cam*255).astype("uint8"), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img,0.5,heatmap,0.5,0)

    cv2.imwrite(ROOT / "outputs/attention_maps/cam.jpg", overlay)

    print("Inference done")


if __name__ == "__main__":
    inference()