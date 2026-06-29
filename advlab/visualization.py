from pathlib import Path

import torch
from torchvision.utils import save_image


def save_adversarial_examples(
    x,
    x_adv,
    y,
    clean_preds,
    adv_preds,
    class_names,
    output_dir,
    attack_name: str,
    start_index: int = 0,
    max_images: int = 8,
) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(max_images, x.size(0))

    for i in range(count):
        diff = (x_adv[i] - x[i]).abs()
        max_diff = diff.max().clamp_min(1e-12)
        diff_vis = diff / max_diff
        grid = torch.stack([x[i].cpu(), x_adv[i].cpu(), diff_vis.cpu()], dim=0)

        true_name = _safe_name(class_names[int(y[i])])
        clean_name = _safe_name(class_names[int(clean_preds[i])])
        adv_name = _safe_name(class_names[int(adv_preds[i])])
        filename = (
            f"{attack_name}_{start_index + i:04d}_"
            f"true-{true_name}_clean-{clean_name}_adv-{adv_name}.png"
        )
        save_image(grid, output_dir / filename, nrow=3)

    return count


def _safe_name(value: str) -> str:
    return value.replace("/", "-").replace(" ", "_")
