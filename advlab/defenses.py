from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor


def apply_defense(x, name: str = "none", jpeg_quality: int = 75, smoothing_kernel: int = 3):
    if name == "none":
        return x
    if name == "jpeg":
        return jpeg_compression(x, quality=jpeg_quality)
    if name == "smoothing":
        return spatial_smoothing(x, kernel_size=smoothing_kernel)
    raise ValueError(f"Unsupported defense: {name}")


def jpeg_compression(x, quality: int = 75):
    outputs = []
    for image in x.detach().cpu():
        pil_image = to_pil_image(image.clamp(0.0, 1.0))
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        outputs.append(to_tensor(Image.open(buffer).convert("RGB")))
    return torch.stack(outputs, dim=0).to(x.device)


def spatial_smoothing(x, kernel_size: int = 3):
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        raise ValueError("smoothing_kernel must be odd")
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
