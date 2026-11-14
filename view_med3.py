"""Xem ảnh gốc so với ảnh sau bộ lọc trung vị 3x3 (med3).

Cách dùng:
  python view_med3.py                          # xem 10 ảnh đầu val
  python view_med3.py --count 5                # chỉ xem 5 ảnh
  python view_med3.py --index 0 50 100         # xem ảnh tại chỉ số 0, 50, 100
  python view_med3.py --split test             # dùng tập test thay vì val

Ảnh được lưu vào experiments/ppd_classification/figures/view_med3/
Mỗi ảnh: ghép ngang [gốc | med3]
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
CACHE = BASE_DIR / "data" / "processed" / "cache"
FIGURES = BASE_DIR / "experiments" / "ppd_classification" / "figures" / "view_med3"
CLASSES = ["Folding_Knife", "Multi-tool_Knife", "Scissor", "Straight_Knife", "Utility_Knife"]


def median3(x):
    """Áp dụng median filter 3x3 lên ảnh x.
    x: (B, 3, H, W) trong [0, 1], float32
    Trả về: (B, 3, H, W) float32 trong [0, 1]
    """
    out = []
    for i in range(0, x.shape[0], 8):
        xb = x[i:i + 8]
        pad = F.pad(xb, (1, 1, 1, 1), mode="reflect")
        patches = pad.unfold(2, 3, 1).unfold(3, 3, 1)          # (b, 3, H, W, 3, 3)
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

        # Áp dụng median filter 3x3
        x_batch = x_orig.unsqueeze(0).contiguous()  # (1, 3, H, W)
        x_med = median3(x_batch)  # (1, 3, H, W)

        # Tạo danh sách ảnh để ghép: [gốc, med3]
        images = [tensor_to_pil(x_orig), tensor_to_pil(x_med[0])]
        labels = [f"Gốc\n{label}", "Med3 3x3"]

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
    print("Mở thư mục này để xem ảnh gốc so với ảnh sau bộ lọc trung vị 3x3.")


if __name__ == "__main__":
    main()
