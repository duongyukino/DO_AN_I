import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd

from advlab.attacks import cw_l2_attack, fgsm_attack, pgd_attack
from advlab.data import get_opixray_loaders
from advlab.defenses import apply_defense
from advlab.metrics import (
    Timer,
    image_difference_metrics,
    init_class_stats,
    perturbation_norms,
    summarize_class_stats,
    update_class_stats,
)
from advlab.models import SimpleCNN
from advlab.utils import get_device, set_seed
from advlab.visualization import save_adversarial_examples


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate adversarial attacks on OPIXray")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model-path", type=str, default="checkpoints/opixray_cnn.pt")
    parser.add_argument("--fgsm-eps", nargs="*", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--pgd-eps", nargs="*", type=float, default=[0.01, 0.02])
    parser.add_argument("--pgd-alpha", type=float, default=0.002)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--cw", action="store_true", help="Run untargeted C&W L2 attack")
    parser.add_argument("--cw-c", type=float, default=1.0)
    parser.add_argument("--cw-kappa", type=float, default=0.0)
    parser.add_argument("--cw-steps", type=int, default=100)
    parser.add_argument("--cw-lr", type=float, default=0.01)
    parser.add_argument(
        "--cw-max-samples",
        type=int,
        default=128,
        help="Limit C&W to a small subset because it is optimization-based and slow.",
    )
    parser.add_argument("--defense", choices=["none", "jpeg", "smoothing"], default="none")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--smoothing-kernel", type=int, default=3)
    parser.add_argument("--save-examples", type=int, default=0)
    parser.add_argument("--examples-dir", type=str, default="results/examples")
    parser.add_argument(
        "--per-class-output",
        type=str,
        default=None,
        help="Optional path for per-class robust accuracy report.",
    )
    parser.add_argument("--output", type=str, default="results/opixray_attack_report.csv")
    return parser.parse_args()


def evaluate_attack(
    model,
    loader,
    device,
    attack_name,
    class_names,
    max_samples=None,
    defense="none",
    jpeg_quality=75,
    smoothing_kernel=3,
    examples_dir=None,
    save_examples=0,
    example_name=None,
    **kwargs,
):
    model.eval()
    total_clean_correct = 0
    total_correct = 0
    total_successful = 0
    total_l2 = 0.0
    total_linf = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total = 0
    class_stats = init_class_stats(class_names)
    saved_examples = 0

    with Timer() as timer:
        for x, y in loader:
            if max_samples is not None:
                remaining = max_samples - total
                if remaining <= 0:
                    break
                x = x[:remaining]
                y = y[:remaining]

            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                clean_logits = model(apply_defense(x, defense, jpeg_quality, smoothing_kernel))
                clean_preds = clean_logits.argmax(dim=1)

            x_adv = generate_attack(model, x, y, attack_name, **kwargs)
            x_adv_eval = apply_defense(x_adv, defense, jpeg_quality, smoothing_kernel)

            with torch.no_grad():
                logits_adv = model(x_adv_eval)
                adv_preds = logits_adv.argmax(dim=1)

            clean_ok = clean_preds == y
            adv_ok = adv_preds == y
            successful = clean_ok & ~adv_ok
            l2, linf = perturbation_norms(x, x_adv)
            mse, mae, psnr, ssim = image_difference_metrics(x, x_adv)

            total_clean_correct += clean_ok.sum().item()
            total_correct += adv_ok.sum().item()
            total_successful += successful.sum().item()
            total_l2 += l2.sum().item()
            total_linf += linf.sum().item()
            total_mse += mse.sum().item()
            total_mae += mae.sum().item()
            total_psnr += psnr.sum().item()
            total_ssim += ssim.sum().item()
            total += y.size(0)
            update_class_stats(class_stats, y, clean_preds, adv_preds, class_names)

            if examples_dir and saved_examples < save_examples:
                saved_examples += save_adversarial_examples(
                    x,
                    x_adv,
                    y,
                    clean_preds,
                    adv_preds,
                    class_names,
                    examples_dir,
                    attack_name=example_name or attack_name,
                    start_index=saved_examples,
                    max_images=save_examples - saved_examples,
                )

    clean_correct = total_clean_correct
    metrics = {
        "acc": total_correct / total,
        "clean_acc": total_clean_correct / total,
        "attack_success_rate": total_successful / clean_correct if clean_correct else 0.0,
        "avg_l2": total_l2 / total,
        "avg_linf": total_linf / total,
        "avg_mse": total_mse / total,
        "avg_mae": total_mae / total,
        "avg_psnr": total_psnr / total,
        "avg_ssim": total_ssim / total,
        "runtime_seconds": timer.elapsed,
        "num_samples": total,
        "defense": defense,
    }
    return metrics, class_stats


def generate_attack(model, x, y, attack_name, **kwargs):
    if attack_name == "fgsm":
        return fgsm_attack(model, x, y, eps=kwargs["eps"])
    if attack_name == "pgd":
        return pgd_attack(
            model,
            x,
            y,
            eps=kwargs["eps"],
            alpha=kwargs["alpha"],
            steps=kwargs["steps"],
        )
    if attack_name == "cw_l2":
        return cw_l2_attack(
            model,
            x,
            y,
            c=kwargs["c"],
            kappa=kwargs["kappa"],
            steps=kwargs["steps"],
            lr=kwargs["lr"],
        )
    raise ValueError(f"Unsupported attack: {attack_name}")


@torch.no_grad()
def evaluate_clean_with_defense(
    model,
    loader,
    device,
    defense="none",
    jpeg_quality=75,
    smoothing_kernel=3,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_eval = apply_defense(x, defense, jpeg_quality, smoothing_kernel)
        logits = model(x_eval)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, total_correct / total, total


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    _, test_loader, class_names = get_opixray_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        image_size=args.image_size,
    )
    in_channels = 3
    num_classes = len(class_names)

    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        ckpt_dataset = state.get("dataset")
        ckpt_in_channels = state.get("in_channels", in_channels)
        ckpt_num_classes = state.get("num_classes", num_classes)

        if ckpt_dataset and ckpt_dataset != "opixray":
            raise ValueError(
                f"Checkpoint dataset '{ckpt_dataset}' is not compatible with OPIXray."
            )
        if ckpt_in_channels != in_channels or ckpt_num_classes != num_classes:
            raise ValueError(
                "Checkpoint model dimensions do not match selected dataset. "
                f"Expected (in_channels={in_channels}, num_classes={num_classes}), "
                f"got (in_channels={ckpt_in_channels}, num_classes={ckpt_num_classes})."
            )

        model = SimpleCNN(in_channels=ckpt_in_channels, num_classes=ckpt_num_classes).to(device)
        model.load_state_dict(state["state_dict"])
    else:
        model = SimpleCNN(in_channels=in_channels, num_classes=num_classes).to(device)
        model.load_state_dict(state)

    print(f"Dataset=OPIXray | num_classes={num_classes} | in_channels={in_channels}")
    print(f"OPIXray labels: {class_names}")

    clean_loss, clean_acc, clean_samples = evaluate_clean_with_defense(
        model,
        test_loader,
        device,
        defense=args.defense,
        jpeg_quality=args.jpeg_quality,
        smoothing_kernel=args.smoothing_kernel,
    )
    print(f"Clean test: loss={clean_loss:.4f}, acc={clean_acc:.4f}")

    rows = [
        {
            "attack": "clean",
            "eps": 0.0,
            "acc": clean_acc,
            "clean_acc": clean_acc,
            "attack_success_rate": 0.0,
            "avg_l2": 0.0,
            "avg_linf": 0.0,
            "avg_mse": 0.0,
            "avg_mae": 0.0,
            "avg_psnr": float("inf"),
            "avg_ssim": 1.0,
            "runtime_seconds": 0.0,
            "num_samples": clean_samples,
            "defense": args.defense,
        }
    ]
    per_class_rows = []

    for eps in args.fgsm_eps:
        metrics, class_stats = evaluate_attack(
            model,
            test_loader,
            device,
            "fgsm",
            class_names,
            defense=args.defense,
            jpeg_quality=args.jpeg_quality,
            smoothing_kernel=args.smoothing_kernel,
            examples_dir=args.examples_dir if args.save_examples else None,
            save_examples=args.save_examples,
            example_name=f"fgsm_eps{eps:.3f}",
            eps=eps,
        )
        row = {"attack": "fgsm", "eps": eps}
        row.update(metrics)
        rows.append(row)
        per_class_rows.extend(
            summarize_class_stats(class_stats, "fgsm", {"eps": eps, "defense": args.defense})
        )
        print(
            f"FGSM eps={eps:.3f} -> acc={metrics['acc']:.4f}, "
            f"asr={metrics['attack_success_rate']:.4f}, "
            f"l2={metrics['avg_l2']:.4f}, linf={metrics['avg_linf']:.4f}, "
            f"psnr={metrics['avg_psnr']:.2f}, ssim={metrics['avg_ssim']:.4f}, "
            f"time={metrics['runtime_seconds']:.1f}s"
        )

    for eps in args.pgd_eps:
        metrics, class_stats = evaluate_attack(
            model,
            test_loader,
            device,
            "pgd",
            class_names,
            defense=args.defense,
            jpeg_quality=args.jpeg_quality,
            smoothing_kernel=args.smoothing_kernel,
            examples_dir=args.examples_dir if args.save_examples else None,
            save_examples=args.save_examples,
            example_name=f"pgd_eps{eps:.3f}",
            eps=eps,
            alpha=args.pgd_alpha,
            steps=args.pgd_steps,
        )
        row = {"attack": "pgd", "eps": eps, "alpha": args.pgd_alpha, "steps": args.pgd_steps}
        row.update(metrics)
        rows.append(row)
        per_class_rows.extend(
            summarize_class_stats(
                class_stats,
                "pgd",
                {
                    "eps": eps,
                    "alpha": args.pgd_alpha,
                    "steps": args.pgd_steps,
                    "defense": args.defense,
                },
            )
        )
        print(
            f"PGD  eps={eps:.3f} -> acc={metrics['acc']:.4f}, "
            f"asr={metrics['attack_success_rate']:.4f}, "
            f"l2={metrics['avg_l2']:.4f}, linf={metrics['avg_linf']:.4f}, "
            f"psnr={metrics['avg_psnr']:.2f}, ssim={metrics['avg_ssim']:.4f}, "
            f"time={metrics['runtime_seconds']:.1f}s"
        )

    if args.cw:
        metrics, class_stats = evaluate_attack(
            model,
            test_loader,
            device,
            "cw_l2",
            class_names,
            max_samples=args.cw_max_samples,
            defense=args.defense,
            jpeg_quality=args.jpeg_quality,
            smoothing_kernel=args.smoothing_kernel,
            examples_dir=args.examples_dir if args.save_examples else None,
            save_examples=args.save_examples,
            example_name="cw_l2",
            c=args.cw_c,
            kappa=args.cw_kappa,
            steps=args.cw_steps,
            lr=args.cw_lr,
        )
        row = {
            "attack": "cw_l2",
            "eps": None,
            "c": args.cw_c,
            "kappa": args.cw_kappa,
            "steps": args.cw_steps,
            "lr": args.cw_lr,
            "max_samples": args.cw_max_samples,
        }
        row.update(metrics)
        rows.append(row)
        per_class_rows.extend(
            summarize_class_stats(
                class_stats,
                "cw_l2",
                {
                    "c": args.cw_c,
                    "kappa": args.cw_kappa,
                    "steps": args.cw_steps,
                    "lr": args.cw_lr,
                    "max_samples": args.cw_max_samples,
                    "defense": args.defense,
                },
            )
        )
        print(
            f"C&W L2 c={args.cw_c:.3f}, steps={args.cw_steps}, "
            f"max_samples={args.cw_max_samples} -> acc={metrics['acc']:.4f}, "
            f"asr={metrics['attack_success_rate']:.4f}, "
            f"l2={metrics['avg_l2']:.4f}, linf={metrics['avg_linf']:.4f}, "
            f"psnr={metrics['avg_psnr']:.2f}, ssim={metrics['avg_ssim']:.4f}, "
            f"time={metrics['runtime_seconds']:.1f}s"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved report to: {out_path}")

    if args.per_class_output:
        per_class_path = Path(args.per_class_output)
    else:
        per_class_path = out_path.with_name(f"{out_path.stem}_per_class.csv")
    if per_class_rows:
        per_class_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False, encoding="utf-8-sig")
        print(f"Saved per-class report to: {per_class_path}")


if __name__ == "__main__":
    main()
