"""PPD (Palette Projection Defense) trên OPIXray crop: huấn luyện, đánh giá clean accuracy, nhiều backbone.

Các bước (từ thư mục chứa `data/`):
  python ppd_resnet.py selftest                              # kiểm tra nhanh khung chạy (vài chục giây)
  python ppd_resnet.py cache                                 # cắt crop, lưu cache, học bảng màu K=16/32/64/128
  python ppd_resnet.py train --arch resnet50 --defense none  # fine-tune mô hình, phòng thủ đứng trước
  python ppd_resnet.py eval --ckpt artifacts/checkpoints/ppd/resnet50_none.pt

Backbone (--arch): resnet50 | efficientnet_b0 | vit_small   (vit_small cần `pip install timm`)
Phòng thủ (--defense):
  none | bit3 | bit4 | jpeg75 | jpeg50 | blur0.5 | blur1.0 | ppd16 | ppd32 | ppd64 | ppd128 | ppd32m
  med3 (chỉ median 3x3) | bit3m | bit4m
  (bitN = N bit/kênh, jpegQ = chất lượng Q, blurS = sigma S, ppdK = K màu, hậu tố m = thêm median 3x3)

Mô hình nhận ảnh [0,1] cỡ 224x224; phòng thủ và chuẩn hóa nằm BÊN TRONG mô hình,
nên tấn công có thể sinh nhiễu trực tiếp trên không gian [0,1].
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # toán tử MPS chưa hỗ trợ sẽ chạy trên CPU

import argparse
import csv
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

CLASSES = ["Folding_Knife", "Multi-tool_Knife", "Scissor", "Straight_Knife", "Utility_Knife"]
IMG = 224        # kích thước đầu vào của mô hình
BG_T = 235       # pixel có min(R,G,B) > BG_T coi là nền trắng
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "data" / "OPIXray"
CACHE = BASE_DIR / "data" / "processed" / "cache"
PAL = BASE_DIR / "data" / "processed" / "palettes"
RUNS = BASE_DIR / "experiments" / "ppd_classification"
METRICS = RUNS / "metrics"
FIGURES = RUNS / "figures"
CHECKPOINTS = BASE_DIR / "artifacts" / "checkpoints" / "ppd"
DEV = torch.device("mps" if torch.backends.mps.is_available()
                   else "cuda" if torch.cuda.is_available() else "cpu")

IMAGENET = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# lr là điểm xuất phát, chưa được tinh chỉnh riêng cho từng backbone
ARCH = {
    "resnet50": dict(**IMAGENET, lr=1e-4),
    "efficientnet_b0": dict(**IMAGENET, lr=1e-4),
    "vit_small": dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), lr=5e-5),
}


# ----------------------------------------------------------------------------- dữ liệu
def read_list(p):
    return [Path(l.strip()).stem for l in open(p, encoding="utf-8") if l.strip()]


def parse_ann(path):
    """Mỗi dòng: tên.jpg Lớp x1 y1 x2 y2 -> [(lớp, x1, y1, x2, y2)]."""
    out = []
    for line in open(path, encoding="utf-8"):
        t = line.split()
        cls = [x for x in t if x in CLASSES]
        nums = [x for x in t if x.lstrip("-").isdigit()]
        if cls and len(nums) >= 4:
            out.append((cls[0], *map(int, nums[-4:])))
    return out


def build_crops(names, src, collect_pixels=False, per_crop=400, seed=0):
    rng = np.random.default_rng(seed)
    X, y, meta, pix = [], [], [], []
    for n in names:
        img = Image.open(ROOT / src / f"{src}_image" / f"{n}.jpg").convert("RGB")
        W, H = img.size
        for c, x1, y1, x2, y2 in parse_ann(ROOT / src / f"{src}_annotation" / f"{n}.txt"):
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crop = img.crop((x1, y1, x2, y2))
            if collect_pixels:  # lấy mẫu pixel từ crop gốc (chưa resize) để học bảng màu
                a = np.asarray(crop).reshape(-1, 3)
                pix.append(a[rng.choice(len(a), min(per_crop, len(a)), replace=False)])
            arr = np.asarray(crop.resize((IMG, IMG), Image.BILINEAR)).copy()
            X.append(torch.from_numpy(arr).permute(2, 0, 1))
            y.append(CLASSES.index(c))
            meta.append(n)
    pix = np.concatenate(pix) if pix else None
    return torch.stack(X), torch.tensor(y), meta, pix


def fit_palette(pixels, K):
    """K-1 tâm từ pixel không phải nền (k-means) + 1 tâm cho nền trắng."""
    from sklearn.cluster import MiniBatchKMeans
    pixels = pixels.astype(np.float32)
    bg = pixels.min(1) > BG_T
    km = MiniBatchKMeans(K - 1, batch_size=4096, n_init=3, random_state=0).fit(pixels[~bg])
    bgc = pixels[bg].mean(0) if bg.any() else np.full(3, 255.0, np.float32)
    return np.vstack([km.cluster_centers_, bgc]).astype(np.float32)


def cmd_cache(a):
    CACHE.mkdir(parents=True, exist_ok=True)
    PAL.mkdir(parents=True, exist_ok=True)
    jobs = {"train": (ROOT / "splits/train.txt", "train", True),
            "val": (ROOT / "splits/val.txt", "train", False),
            "test": (ROOT / "test/test_knife.txt", "test", False)}
    grp = {}
    for g in (1, 2, 3):
        for n in read_list(ROOT / f"test/test_knife-{g}.txt"):
            grp[n] = g
    for split, (lst, src, collect) in jobs.items():
        t0 = time.time()
        X, y, names, pix = build_crops(read_list(lst), src, collect)
        d = {"x": X, "y": y, "names": names}
        if split == "test":
            d["group"] = torch.tensor([grp.get(n, 0) for n in names])
        torch.save(d, CACHE / f"{split}.pt")
        if collect:
            np.save(CACHE / "train_pixels.npy", pix)
        print(f"{split}: {len(X)} crop, {time.time() - t0:.0f}s, "
              f"số crop mỗi lớp: {torch.bincount(y, minlength=len(CLASSES)).tolist()}")
    pix = np.load(CACHE / "train_pixels.npy")
    for K in (16, 32, 64, 128):
        np.save(PAL / f"palette_{K}.npy", fit_palette(pix, K))
        print(f"Đã lưu {PAL / ('palette_' + str(K) + '.npy')}")


# ----------------------------------------------------------------------------- phòng thủ
class Defense(nn.Module):
    """Lớp cơ sở. Đặt `ste = True` để gradient đi thẳng qua phép biến đổi không khả vi
    (BPDA, dùng cho tấn công thích nghi). Giá trị đầu ra không đổi."""
    differentiable = False
    ste = False

    def forward(self, x):
        if self.ste and not self.differentiable:
            with torch.no_grad():
                y = self.apply_defense(x.detach())
            return y + (x - x.detach())          # giá trị = y, gradient = đơn vị
        return self.apply_defense(x)


class BitDepth(Defense):
    def __init__(self, bits):
        super().__init__()
        self.levels = 2 ** bits - 1

    def apply_defense(self, x):
        return torch.round(x * self.levels) / self.levels


class PPD(Defense):
    """Chiếu mỗi pixel về màu gần nhất trong bảng màu; tùy chọn median 3x3 sau đó.
    Khi `ste = True` (tấn công thích nghi): phép chiếu luôn dùng gradient thẳng (BPDA). Gradient qua median chọn bằng
    `median_grad`: "true" (gradient thật của median, rất chậm trên MPS), "mean" (gradient của bộ lọc trung bình 3x3)
    hoặc "identity" (gradient thẳng). Giá trị đầu ra không đổi."""
    median_grad = "true"

    def __init__(self, palette, median=False):
        super().__init__()
        self.register_buffer("C", torch.tensor(palette, dtype=torch.float32) / 255.0)  # (K,3)
        self.median = median

    def project(self, x):                      # x: (B,3,H,W) trong [0,1]
        B, _, H, W = x.shape
        C, c2 = self.C, (self.C ** 2).sum(1)
        out = torch.empty_like(x)
        for i in range(0, B, 8):               # chia nhỏ để tiết kiệm bộ nhớ
            xb = x[i:i + 8]
            p = xb.permute(0, 2, 3, 1).reshape(-1, 3)
            d = (p ** 2).sum(1, keepdim=True) - 2 * p @ C.T + c2
            q = C[d.argmin(1)]
            out[i:i + 8] = q.reshape(xb.shape[0], H, W, 3).permute(0, 3, 1, 2)
        return out

    @staticmethod
    def median3(x):
        out = []
        for i in range(0, x.shape[0], 8):      # chia nhỏ: mỗi lát cắt tạo tensor tạm (8,3,H,W,9)
            pad = F.pad(x[i:i + 8], (1, 1, 1, 1), mode="reflect")
            patches = pad.unfold(2, 3, 1).unfold(3, 3, 1)          # (b,3,H,W,3,3)
            out.append(patches.reshape(*patches.shape[:4], 9).median(-1).values)
        return torch.cat(out)

    def apply_defense(self, x):
        out = self.project(x)
        return self.median3(out) if self.median else out

    def forward(self, x):
        if self.ste and self.median and self.median_grad != "identity":
            with torch.no_grad():
                q = self.project(x.detach())
            q = q + (x - x.detach())           # gradient thẳng cho phép chiếu
            if self.median_grad == "true":
                return self.median3(q)         # median: gradient thật
            with torch.no_grad():               # "mean": giá trị = median, gradient = của lọc trung bình 3x3
                m = self.median3(q.detach())
            smooth = F.avg_pool2d(F.pad(q, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
            return m + (smooth - smooth.detach())
        return super().forward(x)


class WithMedian(Defense):
    """[phòng thủ `inner`] rồi median 3x3 (inner rỗng = chỉ median). Dùng cho `med3`, `bit3m`, `bit4m`.
    Khi `ste = True`: `inner` dùng gradient thẳng; gradient qua median chọn bằng `median_grad`
    ("true" | "mean" | "identity"), giống PPD. Giá trị đầu ra không đổi."""
    median_grad = "true"

    def __init__(self, inner=None):
        super().__init__()
        self.inner = inner if inner is not None else nn.Identity()

    def apply_defense(self, x):
        return PPD.median3(self.inner(x))

    def forward(self, x):
        if not self.ste:
            return self.apply_defense(x)
        if hasattr(self.inner, "ste"):
            self.inner.ste = True
        q = self.inner(x)
        if self.median_grad == "true":
            return PPD.median3(q)
        with torch.no_grad():
            m = PPD.median3(q.detach())
        if self.median_grad == "mean":
            smooth = F.avg_pool2d(F.pad(q, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
            return m + (smooth - smooth.detach())
        return m + (q - q.detach())            # identity


class GaussianBlur(Defense):
    differentiable = True                      # tấn công thích nghi dùng gradient thật

    def __init__(self, sigma):
        super().__init__()
        self.r = max(1, int(np.ceil(3 * sigma)))
        ax = torch.arange(-self.r, self.r + 1, dtype=torch.float32)
        k = torch.exp(-ax ** 2 / (2 * sigma ** 2))
        k = k / k.sum()
        self.register_buffer("k", (k[:, None] * k[None, :])[None, None].repeat(3, 1, 1, 1))

    def apply_defense(self, x):
        return F.conv2d(F.pad(x, (self.r,) * 4, mode="reflect"), self.k, groups=3)


class JPEG(Defense):
    """Nén rồi giải nén JPEG (Pillow, chạy trên CPU). Không khả vi."""

    def __init__(self, quality):
        super().__init__()
        self.q = quality

    def apply_defense(self, x):
        arr = (x.detach().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        out = []
        for a in arr:
            buf = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(a)).save(buf, format="JPEG", quality=self.q)
            buf.seek(0)
            out.append(np.asarray(Image.open(buf).convert("RGB")))
        y = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0
        return y.contiguous().to(x.device)   # permute tạo tensor kiểu channels-last, backward trên MPS cần bố cục chuẩn


def make_defense(name):
    if name == "none":
        return nn.Identity()
    if name == "med3":
        return WithMedian()
    m = re.fullmatch(r"bit(\d)(m?)", name)
    if m:
        d = BitDepth(int(m[1]))
        return WithMedian(d) if m[2] else d
    m = re.fullmatch(r"jpeg(\d+)", name)
    if m:
        return JPEG(int(m[1]))
    m = re.fullmatch(r"blur(\d+(?:\.\d+)?)", name)
    if m:
        return GaussianBlur(float(m[1]))
    m = re.fullmatch(r"ppd(\d+)(m?)", name)
    if m:
        return PPD(np.load(PAL / f"palette_{m[1]}.npy"), median=bool(m[2]))
    raise ValueError(f"Phòng thủ không hợp lệ: {name}")


# ----------------------------------------------------------------------------- mô hình
def build_backbone(arch, pretrained):
    n = len(CLASSES)
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(2048, n)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    elif arch == "vit_small":
        import timm
        m = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=n)
    else:
        raise ValueError(f"Backbone không hợp lệ: {arch}")
    return m


class Net(nn.Module):
    """[0,1] -> phòng thủ -> chuẩn hóa -> backbone."""

    def __init__(self, arch, defense, pretrained=True):
        super().__init__()
        self.defense = defense
        self.backbone = build_backbone(arch, pretrained)
        self.register_buffer("mean", torch.tensor(ARCH[arch]["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(ARCH[arch]["std"]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((self.defense(x).contiguous() - self.mean) / self.std)


def batches(X, y, bs, shuffle=False, aug=False):
    idx = torch.randperm(len(X)) if shuffle else torch.arange(len(X))
    for i in range(0, len(X), bs):
        j = idx[i:i + bs]
        xb = X[j].float() / 255.0
        if aug:                                  # lật ngang/dọc ngẫu nhiên
            h = torch.rand(len(j)) < 0.5
            v = torch.rand(len(j)) < 0.5
            xb[h] = xb[h].flip(3)
            xb[v] = xb[v].flip(2)
        yield xb.to(DEV), y[j].to(DEV)


@torch.no_grad()
def predict(net, X, y, bs=64):
    net.eval()
    return torch.cat([net(xb).argmax(1).cpu() for xb, _ in batches(X, y, bs)])


def metrics(p, y):
    acc = (p == y).float().mean().item()
    recalls = [(p[y == c] == c).float().mean().item() for c in range(len(CLASSES)) if (y == c).any()]
    return acc, float(np.mean(recalls))


def seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)


def run_name(arch, defense, seed):
    return f"{arch}_{defense}" + (f"_s{seed}" if seed else "")


def cmd_train(a):
    seed_all(a.seed)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    tr, va = torch.load(CACHE / "train.pt"), torch.load(CACHE / "val.pt")
    net = Net(a.arch, make_defense(a.defense), pretrained=not a.no_pretrained).to(DEV)
    lr = a.lr or ARCH[a.arch]["lr"]
    opt = torch.optim.AdamW(net.backbone.parameters(), lr=lr, weight_decay=1e-4)
    steps = a.epochs * ((len(tr["x"]) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    name = run_name(a.arch, a.defense, a.seed)
    best, hist, out = -1.0, [], CHECKPOINTS / f"{name}.pt"
    print(f"Thiết bị: {DEV} | backbone: {a.arch} | phòng thủ khi train: {a.defense} | lr {lr} | "
          f"seed {a.seed} | {len(tr['x'])} crop train, {len(va['x'])} crop val")
    for ep in range(a.epochs):
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for xb, yb in batches(tr["x"], tr["y"], a.bs, shuffle=True, aug=True):
            loss = F.cross_entropy(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * len(yb), n + len(yb)
        acc, bal = metrics(predict(net, va["x"], va["y"]), va["y"])
        hist.append({"epoch": ep + 1, "loss": tot / n, "val_acc": acc, "val_bal_acc": bal})
        print(f"epoch {ep + 1}/{a.epochs} | loss {tot / n:.4f} | val acc {acc:.4f} | "
              f"val balanced acc {bal:.4f} | {time.time() - t0:.0f}s")
        if acc > best:
            best = acc
            torch.save({"state": net.backbone.state_dict(), "arch": a.arch, "defense": a.defense,
                        "val_acc": acc, "epoch": ep + 1, "seed": a.seed}, out)
    json.dump(hist, open(METRICS / f"{name}_history.json", "w"), indent=1)
    print(f"Checkpoint tốt nhất (val acc {best:.4f}) -> {out}")


def cmd_eval(a):
    ckpt_path = Path(a.ckpt).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = BASE_DIR / ckpt_path
    ck = torch.load(ckpt_path, map_location="cpu")
    arch = ck.get("arch", "resnet50")           # checkpoint cũ không ghi arch
    net = Net(arch, nn.Identity(), pretrained=False)
    net.backbone.load_state_dict(ck["state"])
    net.to(DEV)
    d = torch.load(CACHE / f"{a.split}.pt")
    print(f"Checkpoint: {ckpt_path} ({arch}, train với '{ck['defense']}', epoch {ck['epoch']}) | "
          f"đánh giá trên {a.split}: {len(d['x'])} crop")
    rows = []
    for name in a.defenses:
        net.defense = make_defense(name).to(DEV)
        p = predict(net, d["x"], d["y"])
        acc, bal = metrics(p, d["y"])
        row = {"defense": name, "acc": round(acc, 4), "bal_acc": round(bal, 4)}
        if "group" in d:
            for g in (1, 2, 3):
                m = d["group"] == g
                row[f"acc_grp{g}"] = round((p[m] == d["y"][m]).float().mean().item(), 4)
        rows.append(row)
        print(row)
    METRICS.mkdir(parents=True, exist_ok=True)
    out = METRICS / f"eval_{ckpt_path.stem}_{a.split}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Đã lưu {out}")


# ----------------------------------------------------------------------------- kiểm tra nhanh
def cmd_selftest(a):
    """Chạy trên CPU, không cần tải trọng số. In OK/LỖI cho từng mục."""
    torch.manual_seed(0)
    x = torch.rand(4, 3, IMG, IMG)
    names = ["none", "bit4", "bit3", "jpeg75", "jpeg50", "blur0.5", "blur1.0", "med3", "bit4m", "bit3m"]
    names += [f"ppd{k}" for k in (16, 32, 64, 128)] + ["ppd16m", "ppd32m", "ppd64m"]
    ok = True

    print("== 1. Phòng thủ: hình dạng, miền giá trị, độ méo trên ảnh ngẫu nhiên")
    for n in names:
        try:
            d = make_defense(n)
        except FileNotFoundError:
            print(f"  {n}: bỏ qua (chưa có bảng màu, hãy chạy `cache`)")
            continue
        y = d(x)
        good = y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1 + 1e-6
        ok &= good
        print(f"  {n}: {'OK' if good else 'LỖI'} | L2 TB mỗi pixel (thang 255): "
              f"{((y - x) * 255).pow(2).sum(1).sqrt().mean():.1f}")

    print("== 2. Blur giữ nguyên ảnh phẳng; JPEG và bit-depth giữ đúng cỡ")
    flat = torch.full((1, 3, IMG, IMG), 0.5)
    good = torch.allclose(make_defense("blur1.0")(flat), flat, atol=1e-5)
    ok &= good
    print(f"  blur trên ảnh phẳng: {'OK' if good else 'LỖI'}")

    print("== 3. Gradient thẳng (BPDA): đầu ra giữ nguyên giá trị, gradient bằng 1")
    for n in ["bit4", "jpeg75"] + (["ppd32"] if (PAL / "palette_32.npy").exists() else []):
        d = make_defense(n)
        ref = d(x)
        d.ste = True
        xx = x.clone().requires_grad_(True)
        out = d(xx)
        out.sum().backward()
        good = torch.equal(out.detach(), ref) and torch.allclose(xx.grad, torch.ones_like(xx))
        ok &= good
        print(f"  {n}: {'OK' if good else 'LỖI'}")

    for dn in ["med3", "bit3m"] + (["ppd32m"] if (PAL / "palette_32.npy").exists() else []):
        for mode in ("true", "mean", "identity"):
            d = make_defense(dn)
            ref = d(x)
            d.ste, d.median_grad = True, mode
            xx = x.clone().requires_grad_(True)
            out = d(xx)
            out.sum().backward()
            g = xx.grad
            good = torch.equal(out.detach(), ref) and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0
            if mode == "identity":
                good = good and torch.allclose(g, torch.ones_like(g))
            else:
                good = good and not torch.allclose(g, torch.ones_like(g))
            ok &= good
            print(f"  {dn}, gradient median = {mode}: {'OK' if good else 'LỖI'}")
        try:
            for mode in ("true", "mean"):
                dd = make_defense(dn).to(DEV)
                dd.ste, dd.median_grad = True, mode
                xd = x.to(DEV).clone().requires_grad_(True)
                dd(xd).sum().backward()
                print(f"  {dn} backward trên {DEV} (median = {mode}): OK")
        except Exception as e:
            ok = False
            print(f"  {dn} backward trên {DEV}: LỖI {repr(e)[:200]}")

    print("== 4. Backbone: đầu ra (batch, 5 lớp) và có gradient theo đầu vào")
    for arch in ARCH:
        try:
            net = Net(arch, nn.Identity(), pretrained=False).eval()
        except ImportError:
            print(f"  {arch}: bỏ qua (chưa cài timm: pip install timm)")
            continue
        xx = x[:2].clone().requires_grad_(True)
        out = net(xx)
        out.sum().backward()
        good = tuple(out.shape) == (2, len(CLASSES)) and xx.grad.abs().sum() > 0
        ok &= bool(good)
        print(f"  {arch}: {'OK' if good else 'LỖI'} | tham số: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M")

    if (CACHE / "val.pt").exists():
        print("== 5. Độ méo L2 TB mỗi pixel trên 300 crop val thật (thang 255)")
        xv = torch.load(CACHE / "val.pt")["x"][:300].float() / 255
        for n in names[1:]:
            try:
                with torch.no_grad():
                    yv = make_defense(n)(xv)
            except FileNotFoundError:
                continue
            print(f"  {n}: {((yv - xv) * 255).pow(2).sum(1).sqrt().mean():.2f}")
    print("\nKẾT QUẢ:", "tất cả đạt" if ok else "CÓ MỤC LỖI, xem phía trên")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT), help="thư mục OPIXray")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cache")
    sub.add_parser("selftest")
    t = sub.add_parser("train")
    t.add_argument("--arch", default="resnet50", choices=list(ARCH))
    t.add_argument("--defense", default="none")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=None, help="mặc định theo backbone")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--no_pretrained", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--split", default="val", choices=["val", "test"])
    e.add_argument("--defenses", nargs="+",
                   default=["none", "bit4", "bit3", "jpeg75", "jpeg50", "blur0.5", "blur1.0",
                            "ppd16", "ppd32", "ppd64", "ppd128", "ppd32m"])
    a = ap.parse_args()
    requested_root = Path(a.root).expanduser()
    ROOT = requested_root if requested_root.is_absolute() else BASE_DIR / requested_root
    {"cache": cmd_cache, "train": cmd_train, "eval": cmd_eval, "selftest": cmd_selftest}[a.cmd](a)