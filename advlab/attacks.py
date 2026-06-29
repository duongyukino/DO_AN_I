import torch
import torch.nn.functional as F


def fgsm_attack(model, x, y, eps: float):
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_sign = x_adv.grad.sign()
    x_adv = x_adv + eps * grad_sign
    return x_adv.clamp(0.0, 1.0).detach()


def pgd_attack(model, x, y, eps: float, alpha: float, steps: int):
    x_orig = x.detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
            x_adv = (x_orig + delta).clamp(0.0, 1.0)

    return x_adv.detach()


def cw_l2_attack(
    model,
    x,
    y,
    c: float = 1.0,
    kappa: float = 0.0,
    steps: int = 100,
    lr: float = 0.01,
):
    """Untargeted Carlini-Wagner style L2 attack.

    This compact version optimizes in tanh-space and keeps the best
    successful adversarial image found for each sample.
    """
    x_orig = x.detach()
    x_scaled = x_orig * 2.0 - 1.0
    x_scaled = x_scaled.clamp(-0.999999, 0.999999)
    w = (0.5 * torch.log((1.0 + x_scaled) / (1.0 - x_scaled))).detach()
    w.requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)

    best_adv = x_orig.clone()
    best_l2 = torch.full((x_orig.size(0),), float("inf"), device=x_orig.device)

    for _ in range(steps):
        adv = 0.5 * (torch.tanh(w) + 1.0)
        logits = model(adv)
        num_classes = logits.size(1)
        one_hot = F.one_hot(y, num_classes=num_classes).float()

        correct_logit = (one_hot * logits).sum(dim=1)
        other_logit = ((1.0 - one_hot) * logits - one_hot * 1e4).max(dim=1).values
        attack_loss = torch.clamp(correct_logit - other_logit + kappa, min=0.0)
        l2_loss = (adv - x_orig).view(x_orig.size(0), -1).pow(2).sum(dim=1)
        loss = (l2_loss + c * attack_loss).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            improved = (preds != y) & (l2_loss < best_l2)
            best_l2[improved] = l2_loss[improved]
            best_adv[improved] = adv.detach()[improved]

    final_adv = (0.5 * (torch.tanh(w) + 1.0)).detach()
    never_succeeded = torch.isinf(best_l2)
    best_adv[never_succeeded] = final_adv[never_succeeded]
    return best_adv.detach()
