# Hướng dẫn gửi dự án PPD qua GitHub

## Tổng quan

Dự án gồm hai phần:

| Phần | Dung lượng | Cách gửi |
|---|---|---|
| **Code + kết quả số liệu** | ~20 MB | Push trực tiếp lên GitHub repository |
| **Checkpoint + cache dữ liệu** | ~2,5 GB | Đóng gói ZIP, tải lên Google Drive |

---

## Bước 1: Chuẩn bị repository

### 1.1. Cập nhật .gitignore

File `.gitignore` hiện tại đã loại trừ checkpoint, cache và metrics. Giữ nguyên cấu hình này để repository nhẹ.

### 1.2. Thêm file cần thiết vào Git

```bash
cd adversarial-attack-project

# Thêm code nguồn
git add attack_eval.py ppd_resnet.py setup.py README.md
git add data/thu_mau.py data/tach_validation.py kiem_ke_crop.py

# Thêm kết quả thí nghiệm (CSV nhẹ)
git add experiments/ppd_classification/metrics/summary*.csv
git add experiments/ppd_classification/configs/

# Thêm palette (nhẹ, ~16 KB)
git add data/processed/palettes/

# Thêm script phân tích
git add view_med3.py view_ppd.py view_ppd_compare.py measure_ppd_quality.py
git add build_summaries.py rebuild_summary.py summarize_results.py visualize_def.py

# Thêm ZIP kết quả test (nhẹ, ~50 KB tổng)
git add ppd_test_results_v3.zip ppd_new_results.zip ppd_final_results.zip

# Commit
git commit -m "Thêm code và kết quả số liệu PPD"
```

### 1.3. Push lên GitHub

```bash
git push origin main
```

Nếu branch có tên khác, thay `main` bằng tên branch thực tế.

---

## Bước 2: Đóng gói file lớn cho Google Drive

### 2.1. Tạo ZIP checkpoint và cache

Trên máy Mac, chạy lệnh sau:

```bash
cd adversarial-attack-project

# Tạo ZIP checkpoint (~1,2 GB)
zip -r ~/Desktop/ppd-checkpoints.zip artifacts/checkpoints/ppd/

# Tạo ZIP cache (~1,3 GB)
zip -r ~/Desktop/ppd-cache.zip data/processed/cache/ data/processed/palettes/
```

Hai file ZIP sẽ nằm trên Desktop.

### 2.2. Kiểm tra ZIP

```bash
unzip -l ~/Desktop/ppd-checkpoints.zip | head -20
unzip -l ~/Desktop/ppd-cache.zip | head -20
```

Kiểm tra xem ZIP có chứa đủ các file sau không:

- `artifacts/checkpoints/ppd/resnet50_none.pt`
- `artifacts/checkpoints/ppd/resnet50_ppd16m.pt`
- `artifacts/checkpoints/ppd/resnet50_ppd32m.pt`
- `artifacts/checkpoints/ppd/resnet50_ppd64m.pt`
- `artifacts/checkpoints/ppd/resnet50_jpeg75.pt`
- `artifacts/checkpoints/ppd/resnet50_med3.pt`
- `artifacts/checkpoints/ppd/resnet50_bit3m.pt`
- `artifacts/checkpoints/ppd/efficientnet_b0_none.pt`
- `data/processed/cache/train.pt`
- `data/processed/cache/val.pt`
- `data/processed/cache/test.pt`
- `data/processed/palettes/palette_*.npy`

---

## Bước 3: Google Drive — ✅ Đã hoàn thành

Đã tải `ppd-checkpoints.zip` và `ppd-cache.zip` lên Google Drive.

Link chia sẻ: https://drive.google.com/drive/folders/1Jo_bcdaPgHwukp1W0UZetOMp8nVL_q-B?usp=drive_link

Đã cập nhật `README.md` với link trên.

---

## Bước 4: Cập nhật README

Thêm phần hướng dẫn tải dữ liệu vào `README.md`:

```markdown
## Tải dữ liệu và checkpoint

Checkpoint mô hình và cache dữ liệu (~2,5 GB) có sẵn trên Google Drive.

1. Tải hai file ZIP từ [Google Drive](DÁN_LINK_ĐÂY):
   - `ppd-checkpoints.zip` (~1,2 GB) — checkpoint mô hình
   - `ppd-cache.zip` (~1,3 GB) — dữ liệu OPIXray đã tiền xử lý
2. Giải nén vào thư mục dự án:
   ```bash
   unzip ppd-checkpoints.zip -d adversarial-attack-project/
   unzip ppd-cache.zip -d adversarial-attack-project/
   ```
3. Cài đặt: `pip install -e .`
```

---

## Bước 5: Kiểm tra cuối cùng

Sau khi hoàn thành, kiểm tra:

- [ ] Repository có code và kết quả CSV
- [ ] Google Drive có ZIP checkpoint + cache
- [ ] README có hướng dẫn tải dữ liệu
- [ ] Link Google Drive hoạt động

---

## Lưu ý

- Không push trực tiếp file `.pt` vào Git vì sẽ làm repository nặng và khó clone.
- Google Drive miễn phí 15 GB, đủ cho checkpoint và cache (~2,5 GB).
- Đảm bảo link chia sẻ là **"Anyone with the link"** để giảng viên có thể tải được.

---

## Gửi cho giảng viên

1. **Email:** Gửi link Google Drive + link GitHub repository
2. **Nội dung email mẫu:**

```
Kính gửi thầy/cô,

Em gửi kết quả dự án "Palette Projection Defense cho phân loại ảnh X-quang" gồm:

1. Code nguồn và kết quả số liệu: https://github.com/duongyukino/-n-I
2. Checkpoint và dữ liệu (~2,5 GB): https://drive.google.com/... (link chia sẻ)

Cảm ơn thầy/cô đã hướng dẫn.
```
