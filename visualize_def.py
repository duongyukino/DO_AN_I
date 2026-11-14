from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image

import attack_eval as A
import ppd_resnet as P


def save_defended_examples(
    x,
    x_adv,
    y,
    clean_preds,
    adv_preds,
    class_names,
    output_dir,
    attack_name: str,
    defense,
    defense_name: str = "defense",
    start_index: int = 0,
    max_images: int = 8,
    eps: float | None = None,
) -> int:
    """Lưu mỗi ảnh thành một hàng 6 ô:
    gốc | đối kháng | nhiễu | gốc sau phòng thủ | đối kháng sau phòng thủ | |chênh lệch| sau phòng thủ.

    x, x_adv: (B, 3, H, W) trong [0, 1], cùng thiết bị với `defense`.
    clean_preds, adv_preds: dự đoán của MÔ HÌNH ĐẦY ĐỦ (đã gồm phòng thủ) trên ảnh gốc và ảnh đối kháng.
    defense: nn.Module nhận ảnh [0, 1] và trả ảnh [0, 1] (ví dụ ppd_resnet.make_defense("ppd32m")).
    eps: nếu truyền (ví dụ 4 / 255), ô "nhiễu" dùng thang cố định eps nên so sánh được giữa các ảnh;
         nếu None, mỗi ảnh chuẩn hóa theo giá trị lớn nhất của chính nó.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(max_images, x.size(0))

    with torch.no_grad():
        x_def = defense(x[:count])
        x_adv_def = defense(x_adv[:count])

    for i in range(count):
        diff = (x_adv[i] - x[i]).abs()
        scale = eps if eps is not None else diff.max().clamp_min(1e-12)
        diff_vis = (diff / scale).clamp(0.0, 1.0)

        post = (x_adv_def[i] - x_def[i]).abs()
        post_vis = post / post.max().clamp_min(1e-12)

        grid = torch.stack(
            [x[i], x_adv[i], diff_vis, x_def[i], x_adv_def[i], post_vis], dim=0
        ).cpu()

        true_name = _safe_name(class_names[int(y[i])])
        clean_name = _safe_name(class_names[int(clean_preds[i])])
        adv_name = _safe_name(class_names[int(adv_preds[i])])
        filename = (
            f"{attack_name}_{defense_name}_{start_index + i:04d}_"
            f"true-{true_name}_clean-{clean_name}_adv-{adv_name}.png"
        )
        save_image(grid, output_dir / filename, nrow=6)

    return count


def _safe_name(value: str) -> str:
    return value.replace("/", "-").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and save a grid showing images before and after a defense."
    )
    parser.add_argument("--ckpt", required=True, help="Checkpoint, e.g. artifacts/checkpoints/ppd/resnet50_ppd16m.pt")
    parser.add_argument("--defense", default=None, help="Override checkpoint defense, e.g. ppd32m or jpeg75")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--attack", choices=["noise", "fgsm", "pgd", "cw"], default="pgd")
    parser.add_argument("--eps", type=float, default=4, help="Attack strength in /255 units")
    parser.add_argument("--steps", type=int, default=30, help="PGD/C&W steps")
    parser.add_argument("--samples", type=int, default=8, help="Number of images to save")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--adaptive", action="store_true", help="Generate attack through the defense (BPDA)")
    parser.add_argument("--median-grad", choices=["true", "mean", "identity"], default="mean")
    args = parser.parse_args()

    device = P.DEV
    ckpt_path = Path(args.ckpt).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).resolve().parent / ckpt_path
    ck = torch.load(ckpt_path, map_location="cpu")
    defense_name = args.defense or ck["defense"]
    model = P.Net(ck.get("arch", "resnet50"), nn.Identity(), pretrained=False)
    model.backbone.load_state_dict(ck["state"])
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    defense = P.make_defense(defense_name).to(device)
    if hasattr(defense, "ste"):
        defense.ste = args.adaptive
    if hasattr(defense, "median_grad"):
        defense.median_grad = args.median_grad
    model.defense = defense if args.adaptive else nn.Identity()

    cache_path = P.CACHE / f"{args.split}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing {cache_path}; create the cache with `python ppd_resnet.py cache` first.")
    data = torch.load(cache_path, map_location="cpu")
    count = min(args.samples, len(data["x"]))
    x = data["x"][:count].float().div(255).to(device)
    y = data["y"][:count].to(device)
    eps = args.eps / 255

    # Attack generation uses the same helpers as attack_eval.py.
    if args.attack == "noise":
        x_adv = A.sign_noise(x, eps)
    elif args.attack == "fgsm":
        x_adv = A.fgsm_attack(model, x, y, eps)
    elif args.attack == "pgd":
        x_adv = A.pgd_attack(model, x, y, eps, eps / 4, args.steps)
    else:
        x_adv = A.cw_l2_attack(model, x, y, steps=args.steps)

    # Predictions shown in filenames reflect the deployed, defended model.
    model.defense = defense
    with torch.no_grad():
        clean_preds = model(x).argmax(1)
        adv_preds = model(x_adv).argmax(1)

    mode = "adaptive" if args.adaptive else "oblivious"
    output_dir = Path(args.output_dir) if args.output_dir else (
        P.FIGURES / "defense_examples" / ckpt_path.stem / f"{defense_name}_{args.split}_{args.attack}_eps{args.eps:g}_{mode}"
    )
    saved = save_defended_examples(
        x, x_adv, y, clean_preds, adv_preds, P.CLASSES, output_dir,
        attack_name=f"{args.attack}_eps{args.eps:g}_{mode}", defense=defense,
        defense_name=defense_name, max_images=count, eps=eps,
    )
    print(f"Device: {device} | defense: {defense_name} | saved {saved} image grids to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
