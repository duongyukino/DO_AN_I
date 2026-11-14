"""Xem ảnh gốc so với ảnh sau PPD-K và PPD-Km (PPD + median 3x3).

Cách dùng:
  python view_ppd_compare.py                          # xem 10 ảnh đầu val
  python view_ppd_compare.py --count 5                # chỉ xem 5 ảnh
  python view_ppd_compare.py --index 0 50 100         # xem ảnh tại chỉ số 0, 50, 100
  python view_ppd_compare.py --split test             # dùng tập test thay vì val

Ảnh được lưu vào experiments/ppd_classification/figures/view_ppd_compare/
Mỗi ảnh: ghép ngang [gốc | PPD16 | PPD16m | PPD32 | PPD32m | PPD64 | PPD64m]
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
CACHE = BASE_DIR / "data" / "processed" / "cache"
PAL = BASE_DIR / "data" / "processed" / "palettes"
FIGURES = BASE_DIR / "experiments" / "ppd_classification" / "figures" / "view_ppd_compare"
CLASSES = ["Folding_Knife", "Multi-tool_Knife", "Scissor", "Straight_Knife", "Utility_Knife"]


def project_ppd(x, palette):
    """Chiếu batch ảnh x về palette gần nhất.
    x: (B, 3, H, W) trong [0, 1], float32
    palette: (K, 3) float32 trong [0, 1]
    Trả về: (B, 3, H, W) float32 trong [0, 1]
    """
    B, _, H, W = x.shape
    C = torch.tensor(palette, dtype=torch.float32)
    c2 = (C ** 2).sum(1)  # (K,)
    out = torch.empty_like(x)
    for i in range(0, B, 8):
        xb = x[i:i + 8]
        p = xb.permute(0, 2, 3, 1).reshape(-1, 3)  # (N, 3)
        d = (p ** 2).sum(1, keepdim=True) - 2 * p @ C.T + c2  # (N, K)
        q = C[d.argmin(1)]  # (N, 3)
        out[i:i + 8] = q.reshape(xb.shape[0], H, W, 3).permute(0, 3, 1, 2)
    return out


def median3(x):
    """Áp dụng median filter 3x3 lên ảnh x.
    x: (B, 3, H, W) trong [0, 1], float32
    Trả về: (B, 3, H, W) float32 trong [0, 1]
    """
    out = []
    for i in range(0, x.shape[0], 8):
        xb = x[i:i + 8]
        pad = F.pad(xb, (1, 1, 1, 1), mode="reflect")
        patches = pad.unfold(2, 3, 1).unfold(3, 3, 1)  # (b, 3, H, W, 3, 3)
        out.append(patches.reshape(*patches.shape[:4], 9).median(-1).values)
    return torch.cat(out)


def tensor_to_pil(t):
    """Tensor (3, H, W) float [0,1] -> PIL Image RGB uint8."""
    arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--count", type=int, default=10, help="số ảnh cần xem")
    ap.add_argument("--index", nargs="+", type=int, default=None, help="chỉ số cụ thể cần xem")
    args = ap.parse_args()

    # Tải dữ liệu
    d = torch.load(CACHE / f"{args.split}.pt", weights_only=False)
    X = d["x"]  # (N, 3, H, W) uint8
    y = d["y"]
    names = d["names"]

    # Xác định các chỉ số cần xem
    if args.index is not None:
        indices = args.index
    else:
        indices = list(range(min(args.count, len(X))))

    # Tải các palette (16, 32, 64)
    ppd_levels = [16, 32, 64]
    palettes = {}
    for k in ppd_levels:
        pal_path = PAL / f"palette_{k}.npy"
        if pal_path.exists():
            palettes[k] = np.load(pal_path) / 255.0
            print(f"Đã tải palette_{k} ({palettes[k].shape[0]} màu)")
        else:
            print(f"Không tìm thấy {pal_path}, bỏ qua")

    if not palettes:
        print("Không có palette nào, hãy chạy `python ppd_resnet.py cache` trước.")
        return

    # Tạo thư mục đầu ra
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Xử lý từng ảnh
    for idx in indices:
        if idx >= len(X):
            print(f"Chỉ số {idx} vượt quá số lượng ({len(X)}), bỏ qua")
            continue

        x_orig = X[idx].float() / 255.0  # (3, H, W) [0,1]
        label = CLASSES[y[idx].item()]
        name = names[idx]

        x_batch = x_orig.unsqueeze(0).contiguous()  # (1, 3, H, W)

        # Tạo danh sách ảnh để ghép: [gốc, PPD16, PPD16m, PPD32, PPD32m, PPD64, PPD64m]
        images = [tensor_to_pil(x_orig)]
        labels = [f"Gốc\n{label}"]

        for k in sorted(palettes.keys()):
            # PPD-K (chỉ PPD, không median)
            x_ppd = project_ppd(x_batch, palettes[k])
            images.append(tensor_to_pil(x_ppd[0]))
            labels.append(f"PPD{k}")

            # PPD-Km (PPD + median 3x3)
            x_ppd_m = median3(x_ppd)
            images.append(tensor_to_pil(x_ppd_m[0]))
            labels.append(f"PPD{k}m")

        # Ghép ngang
        widths = [im.width for im in images]
        total_width = sum(widths)
        height = images[0].height
        combined = Image.new("RGB", (total_width + len(images) - 1, height + 30))

        draw_obj = ImageDraw.Draw(combined)
        x_offset = 0
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            font = ImageFont.load_default()

        for im, lbl in zip(images, labels):
            draw_obj.text((x_offset + 5, 2), lbl, fill="black", font=font)
            combined.paste(im, (x_offset, 20))
            x_offset += im.width + 1

        out_path = FIGURES / f"{name}_idx{idx}.png"
        combined.save(out_path, "PNG")
        print(f"Đã lưu: {out_path}")

    print(f"\nTất cả ảnh đã lưu vào: {FIGURES}")
    print("Mở thư mục này để xem ảnh gốc so với PPD-K và PPD-Km.")


if __name__ == "__main__":
    main()
