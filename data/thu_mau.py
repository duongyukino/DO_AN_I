import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from pathlib import Path
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

root = Path("OPIXray")
CLASSES = ["Folding_Knife", "Multi-tool_Knife", "Scissor", "Straight_Knife", "Utility_Knife"]
BG_T = 235          # pixel có min(R,G,B) > BG_T coi là nền trắng
N_TRAIN, N_VAL, PIX_PER_CROP = 1500, 300, 400
rng = random.Random(0)
nprng = np.random.default_rng(0)

def read_list(p):
    return [Path(l.strip()).stem for l in open(p, encoding="utf-8") if l.strip()]

def parse_ann(path):
    out = []
    for line in open(path, encoding="utf-8"):
        t = line.split()
        cls = [x for x in t if x in CLASSES]
        nums = [x for x in t if x.lstrip("-").isdigit()]
        if cls and len(nums) >= 4:
            out.append(tuple(map(int, nums[-4:])))
    return out

def sample_pixels(names, split, n_crops, per_crop):
    pix, uniq = [], []
    names = rng.sample(names, min(n_crops, len(names)))
    for n in names:
        img = Image.open(root / split / f"{split}_image" / f"{n}.jpg").convert("RGB")
        for (x1, y1, x2, y2) in parse_ann(root / split / f"{split}_annotation" / f"{n}.txt"):
            a = np.asarray(img.crop((x1, y1, x2, y2))).reshape(-1, 3)
            uniq.append(len(np.unique(a, axis=0)) / len(a))
            idx = nprng.choice(len(a), min(per_crop, len(a)), replace=False)
            pix.append(a[idx])
    return np.concatenate(pix).astype(np.float32), uniq

train_names = read_list(root / "splits/train.txt")
val_names = read_list(root / "splits/val.txt")
Ptr, uniq = sample_pixels(train_names, "train", N_TRAIN, PIX_PER_CROP)
Pva, _ = sample_pixels(val_names, "train", N_VAL, PIX_PER_CROP)

bg_tr = Ptr.min(1) > BG_T
print(f"Pixel train: {len(Ptr)} | tỷ lệ nền (min>{BG_T}): {bg_tr.mean():.2%}")
for t in (220, 235, 245):
    print(f"  tỷ lệ nền ở ngưỡng {t}: {(Ptr.min(1) > t).mean():.2%}")
print(f"Tỷ lệ màu duy nhất / pixel trong crop (trung vị): {np.median(uniq):.2f}  (gần 1 = nhiều màu lẻ do JPEG)")

# --- hình phân bố màu ---
fg = Ptr[~bg_tr]
sub = fg[nprng.choice(len(fg), min(20000, len(fg)), replace=False)]
hsv = rgb_to_hsv(sub / 255.0)
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1)
m = hsv[:, 1] > 0.2
ax1.hist(hsv[m, 0] * 360, bins=90, color="gray")
ax1.set_xlabel("Hue (độ)"); ax1.set_ylabel("Số pixel"); ax1.set_title("Phân bố hue (S > 0.2)")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
s3 = sub[:5000]
ax2.scatter(s3[:, 0], s3[:, 1], s3[:, 2], c=s3 / 255.0, s=1)
ax2.set_xlabel("R"); ax2.set_ylabel("G"); ax2.set_zlabel("B"); ax2.set_title("Pixel không phải nền trong RGB")
plt.tight_layout(); plt.savefig("color_analysis.png", dpi=120)
print("Đã lưu color_analysis.png")

# --- bảng màu và biên an toàn ---
bg_center = Ptr[bg_tr].mean(0) if bg_tr.any() else np.array([255, 255, 255], np.float32)
bg_va = Pva.min(1) > BG_T

def dists(P, C):
    d = np.sqrt(((P[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    return np.sort(d, axis=1)

print("\nK | sai số L2 TB (không nền) | p95 | sai số L2 TB (cả nền) | "
      "tỷ lệ pixel an toàn ε=2 / 4 / 8")
for K in (16, 32, 64):
    km = MiniBatchKMeans(K - 1, batch_size=4096, n_init=3, random_state=0).fit(fg)
    C = np.vstack([km.cluster_centers_, bg_center])
    d = dists(Pva, C)
    d1, d2 = d[:, 0], d[:, 1]
    margin = (d2 - d1) / 2
    safe = [(margin > eps * np.sqrt(3)).mean() for eps in (2, 4, 8)]
    print(f"{K} | {d1[~bg_va].mean():.2f} | {np.percentile(d1[~bg_va], 95):.2f} | "
          f"{d1.mean():.2f} | {safe[0]:.1%} / {safe[1]:.1%} / {safe[2]:.1%}")