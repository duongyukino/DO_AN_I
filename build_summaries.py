"""Tạo 2 file summary từ các file metrics có sẵn.

1. summary_same_checkpoint.csv: Tất cả defense áp dụng trên cùng checkpoint resnet50_none
   → Đo mức độ biến dạng của từng defense khi áp dụng lên cùng một mô hình.

2. summary_best_checkpoint.csv: Mỗi defense dùng checkpoint riêng đã huấn luyện phù hợp
   → Tìm defense tốt nhất cho triển khai thực tế.

Cách dùng:
  python build_summaries.py
"""
import csv
from pathlib import Path

METRICS = Path("experiments/ppd_classification/metrics")
OUTPUT_SAME = METRICS / "summary_same_checkpoint.csv"
OUTPUT_BEST = METRICS / "summary_best_checkpoint.csv"


def read_csv_file(filename):
    """Đọc file CSV và trả về danh sách dict."""
    path = METRICS / filename
    if not path.exists():
        print(f"  [WARN] Không tìm thấy {filename}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_pgd_rows(rows, param_value):
    """Lấy hàng có attack=pgd và param=param_value."""
    for row in rows:
        if row.get("attack", "").strip() == "pgd" and str(row.get("param", "")).strip() == str(param_value):
            return row
    return None


def get_pgd_adapt_rows(rows, param_value):
    """Lấy hàng có attack=pgd_adapt và param=param_value."""
    for row in rows:
        if row.get("attack", "").strip() == "pgd_adapt" and str(row.get("param", "")).strip() == str(param_value):
            return row
    return None


def extract_metrics(row):
    """Trích xuất robust_acc và post_l2 từ một hàng CSV."""
    if row is None:
        return None, None
    rob = round(float(row.get("robust_acc", 0)) * 100, 1)
    post = round(float(row.get("post_l2", 0)), 2)
    return rob, post


def write_summary(filepath, title, rows):
    """Ghi danh sách dict vào file CSV."""
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Phòng thủ", "Checkpoint", "n", "Clean", "Độ méo L2",
            "Obl 2/255", "Obl 4/255", "Obl 8/255",
            "Thích nghi 2/255", "Thích nghi 4/255", "Thích nghi 8/255"
        ])
        for r in rows:
            writer.writerow([
                r["defense"],
                r["checkpoint"],
                r["n"],
                r["clean"],
                r["l2"],
                r["obl_2"],
                r["obl_4"],
                r["obl_8"],
                r["adapt_2"],
                r["adapt_4"],
                r["adapt_8"],
            ])
    print(f"Đã lưu: {filepath}")


# =============================================================================
# DỮ LIỆU: Same checkpoint (resnet50_none)
# =============================================================================
# Từ eval_resnet50_none_val.csv:
# none=0.986, bit4=0.9679, bit3=0.9022, ppd16=0.6899, ppd32=0.743, ppd64=0.7696, ppd128=0.8715, ppd32m=0.7193
# Từ các file attack_resnet50_none_as_*_val_main.csv (oblivious PGD):
# Từ các file attack_resnet50_none_as_*_val_adaptive_adapt30mean.csv (adaptive PGD):

SAME_CKPT_DATA = {
    # defense: (clean_acc, obl_file, adapt_file)
    # Chỉ bao gồm defense có dữ liệu trên checkpoint resnet50_none.
    # med3 và bit3m bị loại vì không có file eval nào áp dụng chúng lên checkpoint none.
    "none":    (0.986, "attack_resnet50_none_val_obl30.csv", None),
    "bit3":    (0.9022, "attack_resnet50_none_as_bit3_val_main.csv", "attack_resnet50_none_as_bit3_val_adaptive_adapt30mean.csv"),
    "bit4":    (0.9679, None, None),  # không có file attack cho bit4 trên none
    "jpeg75":  (0.985, "attack_resnet50_none_as_jpeg75_val_main.csv", "attack_resnet50_none_as_jpeg75_val_adaptive_adapt30mean.csv"),
    "blur1.0": (0.97, "attack_resnet50_none_as_blur1.0_val_main.csv", "attack_resnet50_none_as_blur1.0_val_adaptive_adapt30mean.csv"),
    "ppd16":   (0.6899, "attack_resnet50_none_as_ppd16_val_main.csv", "attack_resnet50_none_as_ppd16_val_adaptive_adapt30mean.csv"),
    "ppd32":   (0.743, None, "attack_resnet50_none_as_ppd32_val_adaptive_adapt30mean.csv"),  # không có oblivious file
    "ppd64":   (0.7696, None, "attack_resnet50_none_as_ppd64_val_adaptive_adapt30mean.csv"),  # không có oblivious file
    "ppd16m":  (None, None, "attack_resnet50_none_as_ppd16m_val_adaptive_adapt30mean.csv"),  # không có oblivious file, clean từ adaptive
    "ppd32m":  (0.7193, "attack_resnet50_none_as_ppd32m_val_main.csv", "attack_resnet50_none_as_ppd32m_val_adaptive_adapt30mean.csv"),
    "ppd64m":  (None, None, "attack_resnet50_none_as_ppd64m_val_adaptive_adapt30mean.csv"),  # không có oblivious file
}

# ppd16m clean_acc không có trong eval_resnet50_none_val.csv, lấy từ file adaptive
# ppd64m clean_acc không có trong eval_resnet50_none_val.csv, lấy từ file adaptive


def build_same_checkpoint_summary():
    """Xây dựng summary_same_checkpoint.csv."""
    print("=" * 60)
    print("XÂY DỰNG: summary_same_checkpoint.csv")
    print("=" * 60)

    results = []
    defense_order = ["none", "bit3", "bit4", "jpeg75", "blur1.0",
                     "ppd16", "ppd32", "ppd64", "ppd16m", "ppd32m", "ppd64m"]

    for defense in defense_order:
        if defense not in SAME_CKPT_DATA:
            continue
        clean_acc, obl_file, adapt_file = SAME_CKPT_DATA[defense]

        print(f"\nXử lý: {defense}")

        # Clean accuracy
        if clean_acc is None:
            # Thử lấy từ adaptive file
            if adapt_file:
                rows = read_csv_file(adapt_file)
                if rows:
                    clean_acc = float(rows[0].get("clean_acc", 0))
                    print(f"  Clean (từ adaptive): {clean_acc}")
        if clean_acc is None:
            if obl_file:
                rows = read_csv_file(obl_file)
                if rows:
                    clean_acc = float(rows[0].get("clean_acc", 0))
                    print(f"  Clean (từ oblivious): {clean_acc}")

        clean_pct = round(clean_acc * 100, 1) if clean_acc else "-"

        # Oblivious PGD
        obl_2, obl_4, obl_8 = "-", "-", "-"
        post_l2 = "-"
        if obl_file:
            rows = read_csv_file(obl_file)
            r2 = get_pgd_rows(rows, "2")
            r4 = get_pgd_rows(rows, "4")
            r8 = get_pgd_rows(rows, "8")
            m2, p2 = extract_metrics(r2)
            m4, _ = extract_metrics(r4)
            m8, _ = extract_metrics(r8)
            if m2 is not None:
                obl_2 = str(m2)
                post_l2 = str(p2)
            if m4 is not None:
                obl_4 = str(m4)
            if m8 is not None:
                obl_8 = str(m8)
        print(f"  Oblivious: 2={obl_2}%, 4={obl_4}%, 8={obl_8}%")

        # Adaptive PGD
        adapt_2, adapt_4, adapt_8 = "-", "-", "-"
        if adapt_file:
            rows = read_csv_file(adapt_file)
            r2 = get_pgd_adapt_rows(rows, "2")
            r4 = get_pgd_adapt_rows(rows, "4")
            r8 = get_pgd_adapt_rows(rows, "8")
            m2, _ = extract_metrics(r2)
            m4, _ = extract_metrics(r4)
            m8, _ = extract_metrics(r8)
            if m2 is not None:
                adapt_2 = str(m2)
            if m4 is not None:
                adapt_4 = str(m4)
            if m8 is not None:
                adapt_8 = str(m8)
        print(f"  Adaptive: 2={adapt_2}%, 4={adapt_4}%, 8={adapt_8}%")

        # n
        n = 200
        if obl_file:
            rows = read_csv_file(obl_file)
            if rows:
                n = int(rows[0].get("n", 200))
        elif adapt_file:
            rows = read_csv_file(adapt_file)
            if rows:
                n = int(rows[0].get("n", 200))

        results.append({
            "defense": defense,
            "checkpoint": "resnet50_none",
            "n": n,
            "clean": clean_pct,
            "l2": post_l2,
            "obl_2": obl_2,
            "obl_4": obl_4,
            "obl_8": obl_8,
            "adapt_2": adapt_2,
            "adapt_4": adapt_4,
            "adapt_8": adapt_8,
        })

    write_summary(OUTPUT_SAME, "Same Checkpoint", results)
    return results


# =============================================================================
# DỮ LIỆU: Best checkpoint (mỗi defense dùng checkpoint riêng)
# =============================================================================

BEST_CKPT_DATA = {
    # defense: (checkpoint, clean_acc, obl_file, adapt_file)
    "none":    ("resnet50_none", 0.986, "attack_resnet50_none_val_obl30.csv", None),
    "bit3":    ("resnet50_bit3", 0.98, "attack_resnet50_bit3_val_obl30.csv", "attack_resnet50_bit3_val_adaptive_adapt30.csv"),
    "bit4":    ("resnet50_bit4", 0.98, "attack_resnet50_bit4_val_obl30.csv", "attack_resnet50_bit4_val_adaptive_adapt30.csv"),
    "jpeg75":  ("resnet50_jpeg75", 0.985, "attack_resnet50_jpeg75_val_obl30.csv", "attack_resnet50_jpeg75_val_adaptive_adapt30.csv"),
    "blur1.0": ("resnet50_blur1.0", 0.99, "attack_resnet50_blur1.0_val_obl30.csv", "attack_resnet50_blur1.0_val_adaptive_adapt30.csv"),
    "med3":    ("resnet50_med3", 0.985, "attack_resnet50_med3_val_obl30.csv", "attack_resnet50_med3_val_adaptive_adapt30mean.csv"),
    "bit3m":   ("resnet50_bit3m", 0.985, "attack_resnet50_bit3m_val_obl30.csv", "attack_resnet50_bit3m_val_adaptive_adapt30mean.csv"),
    "ppd16":   ("resnet50_ppd16", 0.975, "attack_resnet50_ppd16_val_obl30.csv", "attack_resnet50_ppd16_val_adaptive_adapt30.csv"),
    "ppd32":   ("resnet50_ppd32", 0.985, "attack_resnet50_ppd32_val_ctrl30.csv", "attack_resnet50_ppd32_val_adaptive_adapt1.csv"),
    "ppd64":   ("resnet50_ppd64", 0.99, "attack_resnet50_ppd64_val_obl30.csv", "attack_resnet50_ppd64_val_adaptive_adapt30.csv"),
    "ppd16m":  ("resnet50_ppd16m", 0.98, "attack_resnet50_ppd16m_val_obl30.csv", "attack_resnet50_ppd16m_val_adaptive_adapt30mean.csv"),
    "ppd32m":  ("resnet50_ppd32m", 0.98, "attack_resnet50_ppd32m_val_obl30.csv", "attack_resnet50_ppd32m_val_adaptive_adapt30.csv"),
    "ppd64m":  ("resnet50_ppd64m", 0.99, "attack_resnet50_ppd64m_val_obl30.csv", "attack_resnet50_ppd64m_val_adaptive_adapt30mean.csv"),
}


def build_best_checkpoint_summary():
    """Xây dựng summary_best_checkpoint.csv."""
    print("\n" + "=" * 60)
    print("XÂY DỰNG: summary_best_checkpoint.csv")
    print("=" * 60)

    results = []
    defense_order = ["none", "bit3", "bit4", "jpeg75", "blur1.0", "med3", "bit3m",
                     "ppd16", "ppd32", "ppd64", "ppd16m", "ppd32m", "ppd64m"]

    for defense in defense_order:
        if defense not in BEST_CKPT_DATA:
            continue
        checkpoint, clean_acc, obl_file, adapt_file = BEST_CKPT_DATA[defense]

        print(f"\nXử lý: {defense} (checkpoint: {checkpoint})")

        clean_pct = round(clean_acc * 100, 1) if clean_acc else "-"

        # Oblivious PGD
        obl_2, obl_4, obl_8 = "-", "-", "-"
        post_l2 = "-"
        if obl_file:
            rows = read_csv_file(obl_file)
            r2 = get_pgd_rows(rows, "2")
            r4 = get_pgd_rows(rows, "4")
            r8 = get_pgd_rows(rows, "8")
            m2, p2 = extract_metrics(r2)
            m4, _ = extract_metrics(r4)
            m8, _ = extract_metrics(r8)
            if m2 is not None:
                obl_2 = str(m2)
                post_l2 = str(p2)
            if m4 is not None:
                obl_4 = str(m4)
            if m8 is not None:
                obl_8 = str(m8)
        print(f"  Oblivious: 2={obl_2}%, 4={obl_4}%, 8={obl_8}%")

        # Adaptive PGD
        adapt_2, adapt_4, adapt_8 = "-", "-", "-"
        if adapt_file:
            rows = read_csv_file(adapt_file)
            r2 = get_pgd_adapt_rows(rows, "2")
            r4 = get_pgd_adapt_rows(rows, "4")
            r8 = get_pgd_adapt_rows(rows, "8")
            m2, _ = extract_metrics(r2)
            m4, _ = extract_metrics(r4)
            m8, _ = extract_metrics(r8)
            if m2 is not None:
                adapt_2 = str(m2)
            if m4 is not None:
                adapt_4 = str(m4)
            if m8 is not None:
                adapt_8 = str(m8)
        print(f"  Adaptive: 2={adapt_2}%, 4={adapt_4}%, 8={adapt_8}%")

        n = 200
        if obl_file:
            rows = read_csv_file(obl_file)
            if rows:
                n = int(rows[0].get("n", 200))

        results.append({
            "defense": defense,
            "checkpoint": checkpoint,
            "n": n,
            "clean": clean_pct,
            "l2": post_l2,
            "obl_2": obl_2,
            "obl_4": obl_4,
            "obl_8": obl_8,
            "adapt_2": adapt_2,
            "adapt_4": adapt_4,
            "adapt_8": adapt_8,
        })

    write_summary(OUTPUT_BEST, "Best Checkpoint", results)
    return results


def print_comparison_table(same_results, best_results):
    """In bảng so sánh giữa 2 summary."""
    print("\n" + "=" * 80)
    print("BẢNG SO SÁNH: Same Checkpoint vs Best Checkpoint")
    print("=" * 80)
    print(f"{'Defense':<10} {'Same Clean':>10} {'Best Clean':>10} {'Same Adapt 2':>12} {'Best Adapt 2':>12}")
    print("-" * 60)

    same_map = {r["defense"]: r for r in same_results}
    best_map = {r["defense"]: r for r in best_results}

    for defense in same_map:
        if defense in best_map:
            s = same_map[defense]
            b = best_map[defense]
            print(f"{s['defense']:<10} {s['clean']:>10} {b['clean']:>10} {s['adapt_2']:>12} {b['adapt_2']:>12}")


def main():
    same_results = build_same_checkpoint_summary()
    best_results = build_best_checkpoint_summary()
    print_comparison_table(same_results, best_results)

    print("\n" + "=" * 80)
    print("HOÀN THÀNH!")
    print(f"  1. {OUTPUT_SAME}")
    print(f"     → So sánh tất cả defense trên cùng checkpoint resnet50_none")
    print(f"     → Clean accuracy thấp = defense biến dạng ảnh nhiều")
    print(f"  2. {OUTPUT_BEST}")
    print(f"     → Mỗi defense dùng checkpoint riêng đã huấn luyện phù hợp")
    print(f"     → Robust accuracy cao + clean accuracy cao = defense tốt nhất")
    print("=" * 80)


if __name__ == "__main__":
    main()
