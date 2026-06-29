import torch

from advlab.attacks import cw_l2_attack, fgsm_attack, pgd_attack
from advlab.defenses import apply_defense
from advlab.metrics import image_difference_metrics, perturbation_norms
from advlab.models import SimpleCNN


def test_attacks_keep_shape_and_range():
    model = SimpleCNN().eval()
    x = torch.rand(4, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 3])

    x_fgsm = fgsm_attack(model, x, y, eps=0.1)
    x_pgd = pgd_attack(model, x, y, eps=0.1, alpha=0.01, steps=3)
    x_cw = cw_l2_attack(model, x, y, c=0.1, steps=2, lr=0.01)

    assert x_fgsm.shape == x.shape
    assert x_pgd.shape == x.shape
    assert x_cw.shape == x.shape
    assert x_fgsm.min() >= 0 and x_fgsm.max() <= 1
    assert x_pgd.min() >= 0 and x_pgd.max() <= 1
    assert x_cw.min() >= 0 and x_cw.max() <= 1


def test_metrics_and_defense_keep_expected_shapes():
    x = torch.rand(2, 3, 16, 16)
    x_adv = (x + 0.05).clamp(0, 1)

    l2, linf = perturbation_norms(x, x_adv)
    mse, mae, psnr, ssim = image_difference_metrics(x, x_adv)
    smoothed = apply_defense(x, name="smoothing", smoothing_kernel=3)

    assert l2.shape == (2,)
    assert linf.shape == (2,)
    assert mse.shape == (2,)
    assert mae.shape == (2,)
    assert psnr.shape == (2,)
    assert ssim.shape == (2,)
    assert torch.all(mse >= 0)
    assert torch.all(mae >= 0)
    assert torch.all(ssim <= 1)
    assert smoothed.shape == x.shape
    assert smoothed.min() >= 0 and smoothed.max() <= 1


def test_image_difference_metrics_identical_images():
    x = torch.rand(2, 3, 16, 16)

    mse, mae, psnr, ssim = image_difference_metrics(x, x)

    assert torch.allclose(mse, torch.zeros_like(mse))
    assert torch.allclose(mae, torch.zeros_like(mae))
    assert torch.isinf(psnr).all()
    assert torch.allclose(ssim, torch.ones_like(ssim))
