import base64
import io
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    for _name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("No Conv2d layer found in model")


class GradCAMService:
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = (
            target_layer if target_layer else find_last_conv_layer(model)
        )

    def generate(
        self,
        image_bytes: bytes,
        input_tensor: torch.Tensor,
        class_idx: int,
    ) -> str:
        self.model.eval()
        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, _inp, output):
            activations.append(output)

        handle_forward = self.target_layer.register_forward_hook(forward_hook)
        handle_backward = self.target_layer.register_full_backward_hook(backward_hook)

        output = self.model(input_tensor)
        self.model.zero_grad()

        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        handle_forward.remove()
        handle_backward.remove()

        if not activations or not gradients:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(img.resize((224, 224)))
            _, buffer = cv2.imencode(".png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buffer).decode("utf-8")

        gradient = gradients[0].cpu().data.numpy()[0]
        activation = activations[0].cpu().data.numpy()[0]

        weights = np.mean(gradient, axis=(1, 2))
        cam = np.zeros(activation.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activation[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (224, 224))

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img.resize((224, 224))) / 255.0

        heatmap_raw = np.uint8(255 * cam)
        heatmap_colored = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap_colored) / 255
        heatmap = np.flip(heatmap, axis=2)

        cam_image = heatmap * 0.4 + np.float32(img_np) * 0.6
        cam_image = np.clip(cam_image, 0, 1)

        cam_uint8 = np.uint8(255 * cam_image)
        _, buffer = cv2.imencode(".png", cam_uint8)
        return base64.b64encode(buffer).decode("utf-8")

    def generate_with_regions(
        self,
        image_bytes: bytes,
        input_tensor: torch.Tensor,
        class_idx: int,
    ) -> tuple[str, list[str]]:
        """Generate GradCAM heatmap and extract anatomical regions."""
        gradcam_base64 = self.generate(image_bytes, input_tensor, class_idx)
        cam = self._compute_cam(input_tensor, class_idx)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        regions = self.extract_regions(cam, img)
        return gradcam_base64, regions

    def _compute_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Compute raw CAM values for region extraction."""
        self.model.eval()
        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, _inp, output):
            activations.append(output)

        handle_forward = self.target_layer.register_forward_hook(forward_hook)
        handle_backward = self.target_layer.register_full_backward_hook(backward_hook)

        output = self.model(input_tensor)
        self.model.zero_grad()

        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        handle_forward.remove()
        handle_backward.remove()

        if not activations or not gradients:
            return np.zeros((224, 224), dtype=np.float32)

        gradient = gradients[0].cpu().data.numpy()[0]
        activation = activations[0].cpu().data.numpy()[0]

        weights = np.mean(gradient, axis=(1, 2))
        cam = np.zeros(activation.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activation[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (224, 224))
        return cam

    def extract_regions(self, cam: np.ndarray, original_image: Image.Image) -> list[str]:
        """Extract detailed anatomical regions from CAM activation.

        Returns detailed regions like:
        ["fovea_centralis", "superior_temporal_arcade", "inferior_nasal_periphery",
         "optic_disk_temporal", "macula_center", "hard_exudates_region"]
        """
        h, w = cam.shape

        threshold = np.percentile(cam, 90)
        hot_spots = (cam > threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            hot_spots,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return self._get_default_regions(cam)

        regions = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            area = cv2.contourArea(cnt)
            intensity = float(cam[cy, cx]) if 0 <= cy < h and 0 <= cx < w else 0.0

            region = self._map_to_anatomical_region(cx, cy, w, h, area, intensity)
            regions.append(region)

        regions = list(set(regions))

        if not regions:
            return self._get_default_regions(cam)

        return sorted(regions)

    def _map_to_anatomical_region(
        self,
        cx: int,
        cy: int,
        w: int,
        h: int,
        area: float,
        intensity: float,
    ) -> str:
        """Map pixel coordinates to detailed anatomical region.

        Fundus image anatomy mapping (central fovea at image center):
        - Center 30%: fovea_centralis, macula_center, perifovea
        - Superior half: superior_temporal_arcade, superior_macula
        - Inferior half: inferior_temporal_arcade, inferior_macula
        - Nasal (left for right eye, right for left eye): optic_disk_nasal, nasal_periphery
        - Temporal: temporal_arcade, temporal_periphery
        """
        nx = (cx / w) * 2 - 1
        ny = (cy / h) * 2 - 1

        dist_from_center = (nx**2 + ny**2) ** 0.5

        is_nasal = nx < -0.15
        is_temporal = nx > 0.15
        is_central = dist_from_center < 0.3
        is_superior = ny < -0.1
        is_inferior = ny > 0.1
        is_peripheral = dist_from_center > 0.5

        if is_central:
            if dist_from_center < 0.15:
                return "fovea_centralis"
            elif dist_from_center < 0.25:
                if is_superior:
                    return "superior_macula"
                elif is_inferior:
                    return "inferior_macula"
                else:
                    return "macula_center"
            else:
                return "perifovea"

        if is_nasal:
            if abs(ny) < 0.3 and dist_from_center < 0.5:
                return "optic_disk_nasal"
            elif is_peripheral:
                if is_superior:
                    return "superior_nasal_periphery"
                elif is_inferior:
                    return "inferior_nasal_periphery"
                else:
                    return "nasal_periphery"
            else:
                return "nasal_mid_periphery"

        if is_temporal:
            if is_peripheral:
                if is_superior:
                    return "superior_temporal_periphery"
                elif is_inferior:
                    return "inferior_temporal_periphery"
                else:
                    return "temporal_periphery"
            else:
                if is_superior:
                    return "superior_temporal_arcade"
                elif is_inferior:
                    return "inferior_temporal_arcade"
                else:
                    return "temporal_arcade"

        if is_peripheral:
            if is_superior:
                return "superior_periphery"
            elif is_inferior:
                return "inferior_periphery"
            else:
                return "mid_periphery"

        if is_superior:
            return "superior_arcade"
        elif is_inferior:
            return "inferior_arcade"
        else:
            return "posterior_pole"

    def _get_default_regions(self, cam: np.ndarray) -> list[str]:
        """Get default regions based on activation peak location."""
        h, w = cam.shape

        max_idx = np.unravel_index(np.argmax(cam), cam.shape)
        cy, cx = max_idx

        main_region = self._map_to_anatomical_region(cx, cy, w, h, 0, 1.0)

        return [main_region]
