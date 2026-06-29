import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def init_class_stats(class_names: List[str]) -> Dict[str, Dict[str, int]]:
    return {
        name: {"total": 0, "clean_correct": 0, "adv_correct": 0, "successful": 0}
        for name in class_names
    }


def update_class_stats(stats, y, clean_preds, adv_preds, class_names: List[str]) -> None:
    for label, clean_pred, adv_pred in zip(y.cpu(), clean_preds.cpu(), adv_preds.cpu()):
        class_name = class_names[int(label)]
        stats[class_name]["total"] += 1
        clean_ok = int(clean_pred) == int(label)
        adv_ok = int(adv_pred) == int(label)
        stats[class_name]["clean_correct"] += int(clean_ok)
        stats[class_name]["adv_correct"] += int(adv_ok)
        stats[class_name]["successful"] += int(clean_ok and not adv_ok)


def summarize_class_stats(stats, attack_name: str, params: Optional[dict] = None):
    rows = []
    params = params or {}
    for class_name, values in stats.items():
        total = values["total"]
        clean_correct = values["clean_correct"]
        row = {
            "attack": attack_name,
            "class": class_name,
            "num_samples": total,
            "clean_acc": _safe_div(values["clean_correct"], total),
            "adv_acc": _safe_div(values["adv_correct"], total),
            "attack_success_rate": _safe_div(values["successful"], clean_correct),
        }
        row.update(params)
        rows.append(row)
    return rows


def perturbation_norms(x, x_adv):
    delta = (x_adv - x).detach().view(x.size(0), -1)
    l2 = delta.pow(2).sum(dim=1).sqrt()
    linf = delta.abs().max(dim=1).values
    return l2, linf


def image_difference_metrics(x, x_adv, data_range: float = 1.0, ssim_kernel_size: int = 11):
    """Return per-image MSE, MAE, PSNR, and SSIM for tensors in NCHW format."""
    x = x.detach().float()
    x_adv = x_adv.detach().float()
    delta = (x_adv - x).view(x.size(0), -1)

    mse = delta.pow(2).mean(dim=1)
    mae = delta.abs().mean(dim=1)
    psnr = _psnr_from_mse(mse, data_range)
    ssim = _ssim(x, x_adv, data_range=data_range, kernel_size=ssim_kernel_size)
    return mse, mae, psnr, ssim


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.perf_counter() - self.start


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _psnr_from_mse(mse, data_range: float):
    max_value = torch.as_tensor(data_range, dtype=mse.dtype, device=mse.device)
    psnr = 10.0 * torch.log10(max_value.pow(2) / mse.clamp_min(torch.finfo(mse.dtype).eps))
    return torch.where(mse == 0, torch.full_like(psnr, float("inf")), psnr)


def _ssim(x, y, data_range: float = 1.0, kernel_size: int = 11):
    if x.shape != y.shape:
        raise ValueError(f"SSIM expects tensors with the same shape, got {x.shape} and {y.shape}.")
    if x.dim() != 4:
        raise ValueError(f"SSIM expects NCHW tensors, got {x.dim()} dimensions.")

    _, _, height, width = x.shape
    kernel_size = min(kernel_size, height, width)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size < 3:
        return _global_ssim(x, y, data_range)

    padding = kernel_size // 2
    mu_x = F.avg_pool2d(x, kernel_size, stride=1, padding=padding, count_include_pad=False)
    mu_y = F.avg_pool2d(y, kernel_size, stride=1, padding=padding, count_include_pad=False)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x = F.avg_pool2d(x * x, kernel_size, stride=1, padding=padding, count_include_pad=False) - mu_x_sq
    sigma_y = F.avg_pool2d(y * y, kernel_size, stride=1, padding=padding, count_include_pad=False) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, kernel_size, stride=1, padding=padding, count_include_pad=False) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / denominator.clamp_min(torch.finfo(x.dtype).eps)
    return ssim_map.view(x.size(0), -1).mean(dim=1).clamp(-1.0, 1.0)


def _global_ssim(x, y, data_range: float):
    x_flat = x.view(x.size(0), x.size(1), -1)
    y_flat = y.view(y.size(0), y.size(1), -1)

    mu_x = x_flat.mean(dim=2)
    mu_y = y_flat.mean(dim=2)
    sigma_x = ((x_flat - mu_x.unsqueeze(2)) ** 2).mean(dim=2)
    sigma_y = ((y_flat - mu_y.unsqueeze(2)) ** 2).mean(dim=2)
    sigma_xy = ((x_flat - mu_x.unsqueeze(2)) * (y_flat - mu_y.unsqueeze(2))).mean(dim=2)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    ssim = numerator / denominator.clamp_min(torch.finfo(x.dtype).eps)
    return ssim.mean(dim=1).clamp(-1.0, 1.0)
