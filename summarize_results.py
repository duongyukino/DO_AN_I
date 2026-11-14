"""Gom các file experiments/ppd_classification/metrics/attack_*.csv thành một bảng: phòng thủ x (kịch bản, eps).

Đọc các file theo mẫu (có thể sửa bằng --patterns):
  experiments/ppd_classification/metrics/attack_*_val_obl30.csv                  PGD 30 bước, oblivious   (chạy với --tag _obl30)
  experiments/ppd_classification/metrics/attack_*_val_adaptive_adapt30.csv       PGD 30 bước, thích nghi  (chạy với --adaptive --tag _adapt30)
  experiments/ppd_classification/metrics/attack_resnet50_ppd32_val_ctrl30.csv    ppd32 oblivious 30 bước  (đã chạy trước đó)
  experiments/ppd_classification/metrics/attack_resnet50_ppd32_val_adaptive_adapt1.csv   ppd32 thích nghi 30 bước (đã chạy trước đó)

Ví dụ:  python summarize_results.py
"""
import argparse
import csv
import glob
from pathlib import Path

# Độ méo L2 trung bình mỗi pixel (thang 255) trên 300 crop val, đo bằng `ppd_resnet.py selftest` (mục 5).
# Sửa lại nếu bạn đo lại.
DIST = {"none": 0.0, "bit4": 8.15, "bit3": 17.82, "jpeg75": 7.16, "blur1.0": 11.26,
        "ppd16": 23.49, "ppd32": 17.56, "ppd64": 13.88, "ppd128": 10.77, "ppd32m": 19.95,
        "med3": 8.23, "bit4m": 13.06, "bit3m": 21.43, "ppd16m": 25.45, "ppd64m": 16.87}
ORDER = ["none", "bit4", "bit3", "jpeg75", "blur1.0", "med3", "bit4m", "bit3m",
         "ppd16", "ppd32", "ppd64", "ppd128", "ppd16m", "ppd32m", "ppd64m"]   # phòng thủ chưa có dữ liệu sẽ tự bỏ qua
EPS = ["2", "4", "8"]

ap = argparse.ArgumentParser()
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS = BASE_DIR / "experiments" / "ppd_classification" / "metrics"
DEFAULT_FIGURES = BASE_DIR / "experiments" / "ppd_classification" / "figures"
ap.add_argument("--runs", default=str(DEFAULT_RUNS))
ap.add_argument("--patterns", nargs="+", default=[
    "attack_*_val_obl30.csv",
    "attack_*_val_adaptive_adapt30.csv",
    "attack_*_val_adaptive_adapt30id.csv",      # biến thể gradient qua median: identity
    "attack_*_val_adaptive_adapt30mean.csv",    # biến thể gradient qua median: lọc trung bình
    "attack_resnet50_ppd32_val_ctrl30.csv",
    "attack_resnet50_ppd32_val_adaptive_adapt1.csv",
])
ap.add_argument("--pattern100", default="attack_*_val_adaptive_adapt100.csv",
                help="file PGD thích nghi 100 bước (chỉ lấy eps = 2)")
ap.add_argument("--out", default="summary.csv")
a = ap.parse_args()

cells, clean, ns = {}, {}, {}
for pat in a.patterns:
    for f in sorted(glob.glob(str(Path(a.runs) / pat))):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if not r["attack"].startswith("pgd") or r["param"] not in EPS:
                continue
            scen = "adapt" if r["attack"].endswith("_adapt") else "obl"
            key = (r["defense"], scen, r["param"])
            val = float(r["robust_acc"]) * 100
            if key in cells:
                if scen == "adapt":      # nhiều biến thể tấn công thích nghi: lấy giá trị thấp nhất (tấn công mạnh nhất)
                    print(f"Lưu ý: {key} có nhiều biến thể tấn công, lấy giá trị thấp nhất ({f})")
                    val = min(val, cells[key])
                else:
                    print(f"Lưu ý: {key} xuất hiện nhiều lần, dùng giá trị trong {f}")
            cells[key] = val
            clean[r["defense"]] = float(r["clean_acc"]) * 100
            ns[r["defense"]] = r["n"]

a100 = {}
for f in sorted(glob.glob(str(Path(a.runs) / a.pattern100))):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if r["attack"] == "pgd_adapt" and r["param"] == "2":
            a100[r["defense"]] = float(r["robust_acc"]) * 100


def cell(d, scen, e):
    if scen == "adapt" and d == "none":          # không có phòng thủ thì hai kịch bản trùng nhau
        v = cells.get((d, "obl", e))
        return f"{v:.1f}*" if v is not None else "-"
    v = cells.get((d, scen, e))
    return f"{v:.1f}" if v is not None else "-"

header = ["Phòng thủ", "n", "Clean", "Độ méo L2"] + [f"Obl {e}/255" for e in EPS] + [f"Thích nghi {e}/255" for e in EPS] + ["Thích nghi 100 bước 2/255"]
rows = []
for d in ORDER:
    if d not in clean:
        continue
    rows.append([d, ns[d], f"{clean[d]:.1f}", f"{DIST.get(d, float('nan')):.2f}"]
                + [cell(d, "obl", e) for e in EPS] + [cell(d, "adapt", e) for e in EPS]
                + [f"{a100[d]:.1f}" if d in a100 else "-"])

print("| " + " | ".join(header) + " |")
print("|" + "|".join(["---"] * len(header)) + "|")
for r in rows:
    print("| " + " | ".join(r) + " |")
print("\nCác số là robust accuracy (%) của mô hình đầy đủ dưới PGD 30 bước; Clean là accuracy sạch trên cùng tập con.")
print("Cột 'Thích nghi 100 bước' chỉ đo ở eps = 2; so với cột 30 bước để biết tấn công đã hội tụ chưa.")
print("* 'none' không có phòng thủ nên kịch bản thích nghi trùng oblivious.")
print("Sai số chuẩn xấp xỉ: ~1,0 điểm ở 98%, ~2,2 điểm ở 89%, ~2,7 điểm ở 18% (n = 200).")

with open(Path(a.runs) / a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
print(f"\nĐã lưu {Path(a.runs) / a.out}")


# Biểu đồ robust accuracy (thích nghi, 30 bước) theo độ méo sạch; ô vuông = có median
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("Không có matplotlib, bỏ qua biểu đồ.")
if plt is not None:
    fig, axes = plt.subplots(1, len(EPS), figsize=(5 * len(EPS), 4.5), sharey=True)
    for ax, e in zip(axes, EPS):
        for d in ORDER:
            v = cells.get((d, "obl" if d == "none" else "adapt", e))
            x = DIST.get(d)
            if v is None or x is None or x != x:
                continue
            ax.scatter(x, v, marker="s" if d.endswith("m") or d == "med3" else "o")
            ax.annotate(d, (x, v), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(f"Thích nghi, eps = {e}/255")
        ax.set_xlabel("Độ méo sạch (L2 mỗi pixel, thang 255)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Robust accuracy (%)")
    plt.tight_layout()
    out_png = DEFAULT_FIGURES / "robust_vs_distortion.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=130)
    print(f"Đã lưu {out_png}")


# ----- Bảng phụ: FGSM và C&W (chạy với --tag _fc_obl và --adaptive --tag _fc_adapt) -----
fc = {}
for pat in ["attack_*_val_fc_obl.csv", "attack_*_val_adaptive_fc_adapt.csv"]:
    for f in sorted(glob.glob(str(Path(a.runs) / pat))):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            base = r["attack"].replace("_adapt", "")
            if base not in ("fgsm", "cw"):
                continue
            scen = "adapt" if r["attack"].endswith("_adapt") else "obl"
            fc[(r["defense"], scen, base, r["param"])] = float(r["robust_acc"]) * 100
if fc:
    cols = [("fgsm", "2"), ("fgsm", "4"), ("fgsm", "8"), ("cw", "k0"), ("cw", "k10")]
    name = {"fgsm": "FGSM", "cw": "C&W"}
    hdr2 = ["Phòng thủ"] + [f"{'Obl' if sc == 'obl' else 'Thích nghi'} {name[b]} {p}"
                            for sc in ("obl", "adapt") for b, p in cols]
    rows2 = []
    for d in ORDER:
        if not any(k[0] == d for k in fc):
            continue
        row = [d]
        for sc in ("obl", "adapt"):
            for b, p in cols:
                v = fc.get((d, "obl" if (d == "none" and sc == "adapt") else sc, b, p))
                row.append("-" if v is None else f"{v:.1f}" + ("*" if d == "none" and sc == "adapt" else ""))
        rows2.append(row)
    print("\nBảng phụ: FGSM (eps /255) và C&W (kappa), robust accuracy (%)")
    print("| " + " | ".join(hdr2) + " |")
    print("|" + "|".join(["---"] * len(hdr2)) + "|")
    for r in rows2:
        print("| " + " | ".join(r) + " |")
    with open(Path(a.runs) / "summary_fc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr2)
        w.writerows(rows2)
    print(f"Đã lưu {Path(a.runs) / 'summary_fc.csv'}")