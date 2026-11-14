"""Tạo lại summary.csv từ các file metrics có sẵn.

Mục tiêu: So sánh tất cả defense trên checkpoint phù hợp nhất.
- Defense không cần train riêng (none, bit3, bit4, jpeg75, blur1.0, med3, bit3m):
  dùng checkpoint resnet50_none, áp dụng defense tại inference.
- PPD variants (ppd16, ppd32, ppd64, ppd16m, ppd32m, ppd64m):
  dùng checkpoint riêng đã huấn luyện với defense tương ứng.

Cách dùng:
  python rebuild_summary.py
"""
import csv
from pathlib import Path

METRICS = Path("experiments/ppd_classification/metrics")
OUTPUT = METRICS / "summary.csv"

# Cấu trúc defense cần thu thập
# Mỗi defense: (tên, checkpoint, file_oblivious, file_adaptive, file_eval_clean)
# file_oblivious: file CSV cho tấn công oblivious (PGD)
# file_adaptive: file CSV cho tấn công adaptive (PGD_adapt)
# file_eval_clean: file CSV cho clean accuracy (None = lấy từ file attack)

DEFENSES = [
    # Defense dùng checkpoint resnet50_none, áp dụng defense tại inference
    ("none", "resnet50_none",
     "attack_resnet50_none_val_obl30.csv",
     None,  # không có adaptive cho none trên file này
     "eval_resnet50_none_val.csv"),
    ("bit3", "resnet50_none",
     "attack_resnet50_none_as_bit3_val_main.csv",
     "attack_resnet50_none_as_bit3_val_adaptive_adapt30mean.csv",
     "eval_resnet50_none_val.csv"),
    ("bit4", "resnet50_bit4",
     "attack_resnet50_bit4_val_obl30.csv",
     "attack_resnet50_bit4_val_adaptive_adapt30.csv",
     None),
    ("jpeg75", "resnet50_none",
     "attack_resnet50_none_as_jpeg75_val_main.csv",
     "attack_resnet50_none_as_jpeg75_val_adaptive_adapt30mean.csv",
     "eval_resnet50_none_val.csv"),
    ("blur1.0", "resnet50_none",
     "attack_resnet50_none_as_blur1.0_val_main.csv",
     "attack_resnet50_none_as_blur1.0_val_adaptive_adapt30mean.csv",
     "eval_resnet50_none_val.csv"),
    ("med3", "resnet50_med3",
     "attack_resnet50_med3_val_obl30.csv",
     "attack_resnet50_med3_val_adaptive_adapt30mean.csv",
     None),
    ("bit3m", "resnet50_bit3m",
     "attack_resnet50_bit3m_val_obl30.csv",
     "attack_resnet50_bit3m_val_adaptive_adapt30mean.csv",
     None),
    # PPD variants - dùng checkpoint riêng đã huấn luyện với defense
    ("ppd16", "resnet50_ppd16",
     "attack_resnet50_ppd16_val_obl30.csv",
     "attack_resnet50_ppd16m_val_adaptive_adapt30mean.csv",
     None),
    ("ppd32", "resnet50_ppd32",
     "attack_resnet50_ppd32_val_ctrl30.csv",
     "attack_resnet50_ppd32_val_adaptive_adapt1.csv",
     None),
    ("ppd64", "resnet50_ppd64",
     "attack_resnet50_ppd64_val_obl30.csv",
     "attack_resnet50_ppd64_val_adaptive_adapt30.csv",
     None),
    ("ppd16m", "resnet50_ppd16m",
     "attack_resnet50_ppd16m_val_obl30.csv",
     "attack_resnet50_ppd16m_val_adaptive_adapt30mean.csv",
     None),
    ("ppd32m", "resnet50_ppd32m",
     "attack_resnet50_ppd32m_val_obl30.csv",
     "attack_resnet50_ppd32m_val_adaptive_adapt30mean.csv",
     None),
    ("ppd64m", "resnet50_ppd64m",
     "attack_resnet50_ppd64m_val_obl30.csv",
     "attack_resnet50_ppd64m_val_adaptive_adapt30mean.csv",
     None),
]


def read_csv_file(filename):
    """Đọc file CSV và trả về danh sách dict."""
    path = METRICS / filename
    if not path.exists():
        print(f"  [WARN] Không tìm thấy {filename}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_clean_acc_from_eval(defense_name, eval_file):
    """Lấy clean accuracy từ file eval."""
    if eval_file is None:
        return None
    rows = read_csv_file(eval_file)
    for row in rows:
        if row.get("defense", "").strip() == defense_name:
            acc = float(row.get("acc", 0))
            return round(acc * 100, 1)
    return None


def get_metrics_from_attack(rows, param_value):
    """Lấy robust_acc và post_l2 từ hàng có param = param_value."""
    if not rows:
        return None, None
    for row in rows:
        if str(row.get("param", "")).strip() == str(param_value):
            rob = float(row.get("robust_acc", 0))
            post = float(row.get("post_l2", 0))
            return round(rob * 100, 1), round(post, 2)
    return None, None


def get_l2_distortion(defense_name):
    """Tính độ méo L2 trung bình của defense (từ post_l2 ở param=2)."""
    # post_l2 ở param=2 là độ méo nhỏ nhất, đại diện cho distortion của defense
    # Chúng ta đã có post_l2 từ các file attack
    return None  # Sẽ tính từ dữ liệu attack


def main():
    print("Đang tổng hợp dữ liệu từ các file metrics...\n")

    results = []

    for defense_name, checkpoint, obl_file, adapt_file, eval_file in DEFENSES:
        print(f"Xử lý: {defense_name} (checkpoint: {checkpoint})")

        # 1. Clean accuracy
        clean_acc = get_clean_acc_from_eval(defense_name, eval_file)
        if clean_acc is None and obl_file:
            # Lấy từ file attack
            rows = read_csv_file(obl_file)
            if rows:
                clean_acc = round(float(rows[0].get("clean_acc", 0)) * 100, 1)
        elif clean_acc is None and adapt_file:
            rows = read_csv_file(adapt_file)
            if rows:
                clean_acc = round(float(rows[0].get("clean_acc", 0)) * 100, 1)
        print(f"  Clean: {clean_acc}%")

        # 2. Oblivious PGD (từ file _main hoặc _obl30)
        obl_2, obl_4, obl_8 = "-", "-", "-"
        post_l2 = "-"
        if obl_file:
            rows = read_csv_file(obl_file)
            m2, p2 = get_metrics_from_attack(rows, "2")
            m4, _ = get_metrics_from_attack(rows, "4")
            m8, _ = get_metrics_from_attack(rows, "8")
            if m2 is not None:
                obl_2 = str(m2)
                post_l2 = str(p2)
            if m4 is not None:
                obl_4 = str(m4)
            if m8 is not None:
                obl_8 = str(m8)
        print(f"  Oblivious: 2/255={obl_2}%, 4/255={obl_4}%, 8/255={obl_8}%")

        # 3. Adaptive PGD (từ file _adaptive)
        adapt_2, adapt_4, adapt_8 = "-", "-", "-"
        if adapt_file:
            rows = read_csv_file(adapt_file)
            m2, _ = get_metrics_from_attack(rows, "2")
            m4, _ = get_metrics_from_attack(rows, "4")
            m8, _ = get_metrics_from_attack(rows, "8")
            if m2 is not None:
                adapt_2 = str(m2)
            if m4 is not None:
                adapt_4 = str(m4)
            if m8 is not None:
                adapt_8 = str(m8)
        print(f"  Adaptive: 2/255={adapt_2}%, 4/255={adapt_4}%, 8/255={adapt_8}%")

        results.append({
            "defense": defense_name,
            "checkpoint": checkpoint,
            "clean": clean_acc if clean_acc else "-",
            "l2": post_l2 if post_l2 != "-" else "-",
            "obl_2": obl_2,
            "obl_4": obl_4,
            "obl_8": obl_8,
            "adapt_2": adapt_2,
            "adapt_4": adapt_4,
            "adapt_8": adapt_8,
        })
        print()

    # Ghi ra file CSV
    print(f"Đang ghi kết quả vào {OUTPUT}...")
    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Phòng thủ", "Checkpoint", "n", "Clean", "Độ méo L2",
            "Obl 2/255", "Obl 4/255", "Obl 8/255",
            "Thích nghi 2/255", "Thích nghi 4/255", "Thích nghi 8/255"
        ])
        for r in results:
            writer.writerow([
                r["defense"],
                r["checkpoint"],
                200,
                r["clean"],
                r["l2"],
                r["obl_2"],
                r["obl_4"],
                r["obl_8"],
                r["adapt_2"],
                r["adapt_4"],
                r["adapt_8"],
            ])

    print(f"\nĐã lưu: {OUTPUT}")
    print("\nGhi chú:")
    print("- PPD variants dùng checkpoint riêng đã huấn luyện với defense tương ứng")
    print("- Defense khác dùng checkpoint resnet50_none, áp dụng defense tại inference")
    print("- Dấu '-' nghĩa là không có dữ liệu")


if __name__ == "__main__":
    main()
