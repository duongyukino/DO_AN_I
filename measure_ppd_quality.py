"""Đo mức độ biến dạng của PPD so với ảnh gốc.

Tính các chỉ số khách quan:
- MSE: Mean Squared Error (sai số bình phương trung bình)
- PSNR: Peak Signal-to-Noise Ratio (dB) — càng cao càng giống
- SSIM: Structural Similarity Index — càng gần 1 càng giống
- Số màu thực tế: số màu duy nhất trong ảnh sau PPD

Cách dùng:
  python measure_ppd_quality.py
"""
import numpy as np
import torch
from pathlib import Path
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

BASE_DIR = Path(__file__).resolve().parent
CACHE = BASE_DIR / "data" / "processed" / "cache"
PAL = BASE_DIR / "data" / "processed" / "palettes"

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


def count_unique_colors(x):
    """Đếm số màu duy nhất trong ảnh.
    x: (3, H, W) float [0,1]
    """
    arr = (x.numpy() * 255).round().astype(np.uint8)  # (3, H, W)
    pixels = arr.reshape(3, -1).T  # (H*W, 3)
    unique = np.unique(pixels, axis=0)
    return len(unique)


def main():
    # Tải dữ liệu val
    d = torch.load(CACHE / "val.pt", weights_only=False)
    X = d["x"].float() / 255.0  # (N, 3, 224, 224) [0,1]
    y = d["y"]
    names = d["names"]

    # Tải các palette
    ppd_levels = [16, 32, 64, 128]
    palettes = {}
    for k in ppd_levels:
        pal_path = PAL / f"palette_{k}.npy"
        if pal_path.exists():
            palettes[k] = np.load(pal_path) / 255.0
            print(f"Palette {k}: {palettes[k].shape[0]} tâm")
        else:
            print(f"Không tìm thấy {pal_path}")

    # Tính chỉ số trên toàn bộ tập val
    print(f"\n{'='*80}")
    print(f"ĐÁNH GIÁ CHẤT LƯỢNG PPD TRÊN TOÀN BỘ TẬP VAL ({len(X)} ảnh)")
    print(f"{'='*80}\n")

    for k in sorted(palettes.keys()):
        pal = palettes[k]
        mse_sum = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        colors_sum = 0
        colors_per_img = []

        for i in range(0, len(X), 16):
            xb = X[i:i+16]  # (B, 3, H, W)
            xp = project_ppd(xb, pal)

            for j in range(xb.shape[0]):
                orig = xb[j].numpy()  # (3, H, W)
                proc = xp[j].numpy()

                # MSE và PSNR
                mse_val = mean_squared_error(orig, proc)
                mse_sum += mse_val

                try:
                    psnr_val = peak_signal_noise_ratio(orig, proc, data_range=1.0)
                    if np.isfinite(psnr_val):
                        psnr_sum += psnr_val
                    else:
                        psnr_sum += 0.0
                except:
                    psnr_sum += 0.0

                # SSIM (tính trên từng kênh rồi lấy trung bình)
                ssim_vals = []
                for c in range(3):
                    s = structural_similarity(orig[c], proc[c], data_range=1.0)
                    ssim_vals.append(s)
                ssim_sum += np.mean(ssim_vals)

                # Số màu thực tế
                nc = count_unique_colors(xp[j])
                colors_sum += nc
                colors_per_img.append(nc)

        n = len(X)
        print(f"PPD{k:3d} (palette {len(pal):3d} tâm):")
        print(f"  MSE:       {mse_sum/n:.6f}")
        print(f"  PSNR:      {psnr_sum/n:.2f} dB")
        print(f"  SSIM:      {ssim_sum/n:.4f}")
        print(f"  Màu trung bình: {colors_sum/n:.0f} màu/ảnh")
        print(f"  Màu min/max:    {min(colors_per_img)}/{max(colors_per_img)}")
        print()

    # Phân tích theo lớp
    print(f"\n{'='*80}")
    print(f"PHÂN TÍCH THEO LỚP (chỉ PPD32 và PPD64)")
    print(f"{'='*80}\n")

    for k in [32, 64]:
        if k not in palettes:
            continue
        pal = palettes[k]
        print(f"\n--- PPD{k} ---")
        for cls_idx, cls_name in enumerate(CLASSES):
            mask = (y == cls_idx)
            if mask.sum() == 0:
                continue
            cls_x = X[mask]
            cls_p = project_ppd(cls_x, pal)

            mse_sum = 0.0
            ssim_sum = 0.0
            for j in range(cls_x.shape[0]):
                orig = cls_x[j].numpy()
                proc = cls_p[j].numpy()
                mse_sum += mean_squared_error(orig, proc)
                ssim_vals = []
                for c in range(3):
                    s = structural_similarity(orig[c], proc[c], data_range=1.0)
                    ssim_vals.append(s)
                ssim_sum += np.mean(ssim_vals)

            n = cls_x.shape[0]
            print(f"  {cls_name:20s} (n={n:4d}): MSE={mse_sum/n:.6f}  SSIM={ssim_sum/n:.4f}")


if __name__ == "__main__":
    main()
