 # Adversarial Attack Research Project (OPIXray)

Du an mau cho de tai: **"Nghien cuu tan cong mau doi khang len mo hinh hoc may"**.

Muc tieu:
- Hieu duoc ban chat tan cong doi khang va cac kieu tan cong co ban.
- Trien khai va so sanh cac phuong phap: **FGSM**, **PGD** va **C&W L2**.
- Thuc hien thu nghiem thuc te tren bai toan phan loai anh OPIXray.

## 1) Cấu trúc dự án

```
adversarial-attack-project/
├── advlab/                         # Thư viện mô hình, dữ liệu, attack và metric cho pipeline YOLO classification
├── advlab_examples/                # Lệnh train, đánh giá và xem metadata checkpoint
├── ppd_resnet.py                   # Pipeline phân loại OPIXray và phòng thủ PPD
├── attack_eval.py                  # Đánh giá robustness của checkpoint PPD
├── summarize_results.py            # Tổng hợp CSV của thí nghiệm PPD
├── visualize_def.py                # Sinh ảnh ghép minh họa trước/sau defense
├── data/
│   ├── OPIXray/                    # Dữ liệu gốc và split
│   └── processed/
│       ├── cache/                  # Crop tensor dùng lại khi train/evaluate
│       └── palettes/               # Palette K-means cho PPD
├── assets/pretrained/              # Trọng số pretrained YOLO classification
├── artifacts/
│   └── checkpoints/
│       ├── ppd/                    # Checkpoint các backbone của nhánh PPD
│       └── yolov5_classification/  # Checkpoint CNN/YOLO của nhánh advlab
├── experiments/
│   ├── ppd_classification/
│   │   ├── configs/                # JSON protocol và ma trận chạy
│   │   ├── README.md               # Hướng dẫn và phạm vi kết quả PPD
│   │   ├── metrics/                # CSV, lịch sử train và bảng tổng hợp PPD
│   │   └── figures/                # Biểu đồ và ảnh sau defense
│   ├── yolov5_classification/
│   │   ├── README.md               # Hướng dẫn và phạm vi kết quả YOLO classification
│   │   ├── metrics/                # CSV attack, clean accuracy và per-class
│   │   └── figures/examples/       # Ảnh ví dụ được lưu từ pipeline advlab
│   └── physical_robustness/
│       ├── README.md               # Phạm vi đánh giá phòng vệ bằng mẫu vô hại
│       ├── protocol.md             # Template ghi quy trình và metadata
│       ├── manifests/
│       ├── metrics/
│       └── figures/
├── archives/colab/                 # ZIP trao đổi với Google Colab
├── docs/                           # Hướng dẫn thí nghiệm và cấu trúc
│   └── project_structure.md         # Giải thích vai trò từng thư mục và giới hạn pipeline
└── tests/                          # Kiểm thử các hàm cốt lõi
```

Hai pipeline được giữ riêng: `ppd_resnet.py` dùng các crop cache và palette để nghiên cứu defense; `advlab_examples/` là pipeline YOLO classification với dataset loader riêng. Không gộp CSV của hai pipeline nếu chưa chuẩn hóa model, split và metric. Giải thích chi tiết từng thành phần nằm trong `docs/project_structure.md`.

## 2) Hai pipeline nghiên cứu

- **PPD classification:** `ppd_resnet.py` tạo cache và palette, train ResNet/EfficientNet/ViT, rồi `attack_eval.py` đánh giá defense và robustness. Đây là nhánh chứa thí nghiệm palette projection.
- **YOLO classification:** `advlab_examples/` train/evaluate checkpoint YOLO classification với loader riêng. Metadata hiện có ghi model `yolov5n-cls`; đây là classifier, không phải YOLO object detector.
- **Physical robustness:** chỉ ghi nhận độ bền dưới biến thiên thu nhận ảnh có kiểm soát và mẫu thử vô hại. Không có pipeline tối ưu vật thể để tránh bị phát hiện.

Các đường dẫn dữ liệu và đầu ra mặc định được tính theo vị trí file dự án, nên có thể gọi script từ thư mục dự án hoặc truyền đường dẫn tuyệt đối.

### Lệnh nhanh cho nhánh PPD

```bash
python ppd_resnet.py cache
python ppd_resnet.py train --arch resnet50 --defense ppd32m
python attack_eval.py --ckpt artifacts/checkpoints/ppd/resnet50_none.pt --defense ppd32m --attacks pgd --eps 2 4 8 --pgd_steps 30 --n 200
python summarize_results.py
python visualize_def.py --ckpt artifacts/checkpoints/ppd/resnet50_none.pt --defense ppd32m
```

## 3) Cai dat

```bash
cd adversarial-attack-project
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision tqdm pandas scikit-learn ultralytics
pip install -e .
```

## 3) Tai du lieu va checkpoint

Checkpoint mo hinh va cache du lieu (~2,5 GB) co san tren Google Drive.

1. Tai hai file ZIP tu [Google Drive](https://drive.google.com/drive/folders/1Jo_bcdaPgHwukp1W0UZetOMp8nVL_q-B?usp=drive_link):
   - `ppd-checkpoints.zip` (~1,2 GB) — checkpoint mo hinh (ResNet-50 va EfficientNet-B0)
   - `ppd-cache.zip` (~1,3 GB) — du lieu OPIXray da tien xu ly (train, val, test)
2. Giai nen vao thu muc du an:
   ```bash
   unzip ppd-checkpoints.zip -d adversarial-attack-project/
   unzip ppd-cache.zip -d adversarial-attack-project/
   ```
3. Cai dat: `pip install -e .`

## 4) Cau truc dataset OPIXray

Dat du lieu vao `data/OPIXray` voi cau truc:
- `train/train_image`, `train/train_annotation`
- `test/test_image`, `test/test_annotation`

## 5) Train mo hinh clean (OPIXray)

Pipeline `advlab_examples` hỗ trợ `SimpleCNN` và YOLOv5 classification. Để fine-tune YOLOv5n-cls, dùng lệnh sau:

```bash
python advlab_examples/train_opixray.py \
  --data-dir data \
  --image-size 224 \
  --stage1-epochs 5 \
  --epochs 25 \
  --batch-size 64 \
  --lr 0.001 \
  --stage2-lr 0.0001 \
  --model yolov5n-cls \
  --save-path artifacts/checkpoints/yolov5_classification/opixray_yolov5n.pt
```

Script su dung ky thuat **2-Stage Fine-tuning**:
1. **Stage 1**: Dong bang backbone, chi train lop Classify moi trong 5 epoch dau (giup on dinh trong so pre-trained).
2. **Stage 2**: Mo bang toan bo mo hinh trong 25 epoch tiep theo voi learning rate thap hon (1e-4) de tinh chinh sau.

File `.pt` la checkpoint nhi phan cua PyTorch. Khi train xong, script se luu them metadata tai `artifacts/checkpoints/yolov5_classification/opixray_yolov5n.json`.

Script danh gia se tu doc kien truc tu checkpoint.

Neu muon xem thong tin cua checkpoint da co:

```bash
python advlab_examples/inspect_checkpoint.py \
  --model-path artifacts/checkpoints/yolov5_classification/opixray_yolov5n.pt \
  --output artifacts/checkpoints/yolov5_classification/opixray_yolov5n_info.json
```

## 6) Danh gia duoi FGSM + PGD (OPIXray)

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path artifacts/checkpoints/yolov5_classification/opixray_yolov5n.pt \
  --batch-size 64 \
  --fgsm-eps 0.005 0.01 0.02 \
  --pgd-eps 0.01 0.02 \
  --pgd-alpha 0.002 \
  --pgd-steps 20 \
  --save-examples 6 \
  --output experiments/yolov5_classification/metrics/opixray_attack_report.csv
```

Lenh tren tao them:
- `experiments/yolov5_classification/metrics/opixray_attack_report.csv`: ket qua tong hop voi `acc`, `attack_success_rate`, `avg_l2`, `avg_linf`, `avg_mse`, `avg_mae`, `avg_psnr`, `avg_ssim`, `runtime_seconds`.
- `experiments/yolov5_classification/metrics/opixray_attack_report_per_class.csv`: ket qua theo tung lop vu khi.
- `experiments/yolov5_classification/figures/examples/`: mot so anh minh hoa gom anh goc, anh adversarial va nhieu phong dai.

## 7) Danh gia them voi C&W L2 (OPIXray)

C&W la tan cong dua tren toi uu hoa, chay cham hon FGSM/PGD. Nen chay tren mot tap con nho
bang `--cw-max-samples` de phu hop laptop ca nhan.

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path artifacts/checkpoints/yolov5_classification/opixray_yolov5n.pt \
  --batch-size 64 \
  --fgsm-eps 0.005 0.01 0.02 \
  --pgd-eps 0.01 0.02 \
  --pgd-alpha 0.002 \
  --pgd-steps 20 \
  --cw \
  --cw-c 1.0 \
  --cw-kappa 0.0 \
  --cw-steps 100 \
  --cw-lr 0.01 \
  --cw-max-samples 128 \
  --save-examples 6 \
  --output experiments/yolov5_classification/metrics/opixray_attack_report.csv
```

Neu chi muon test nhanh C&W:

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 128 \
  --model-path artifacts/checkpoints/yolov5_classification/opixray_yolov5n.pt \
  --batch-size 32 \
  --fgsm-eps \
  --pgd-eps \
  --cw \
  --cw-steps 50 \
  --cw-max-samples 32 \
  --output experiments/yolov5_classification/metrics/opixray_cw_quick.csv
```

### 7.1) Tan cong AdvGAN (demo nhanh)

AdvGAN trong repo nay duoc cai dat o muc demo nhe: Generator duoc huan luyen cuc bo tren batch de tao nhieu delta = G(x) va kẹp trong [-eps, eps]. Ban co the luu/tai trong so G de tai su dung.

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path artifacts/checkpoints/yolov5_classification/opixray_yolov5n.pt \
  --batch-size 64 \
  --run-advgan \
  --advgan-eps 0.01 0.02 \
  --advgan-iters 10 \
  --advgan-lr 0.001 \
  --save-examples 6 \
  --output experiments/yolov5_classification/metrics/opixray_attack_report.csv
```

Tham so chinh:
- `--advgan-eps`: bien do nhieu toi da.
- `--advgan-iters`: so buoc huan luyen Generator tren moi batch.
- `--advgan-lr`: learning rate cho Generator.
- `--advgan-l2`, `--advgan-tv`: he so regularization (tuy chon) de lam min nhieu.
- `--advgan-load`, `--advgan-save`: duong dan load/save trong so Generator.

## 8) Danh gia voi defense nhe

Co the bat tien xu ly nhe khi danh gia anh adversarial. Day khong phai phong thu manh nhu adversarial training,
nhung phu hop lam baseline nhanh trong Do an 1.

JPEG compression:

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path artifacts/checkpoints/yolov5_classification/opixray_yolov5n.pt \
  --batch-size 64 \
  --fgsm-eps 0.005 0.01 0.02 \
  --pgd-eps 0.01 0.02 \
  --pgd-alpha 0.002 \
  --pgd-steps 20 \
  --defense jpeg \
  --jpeg-quality 75 \
  --output experiments/yolov5_classification/metrics/opixray_attack_jpeg_report.csv
```

Spatial smoothing:

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path artifacts/checkpoints/yolov5_classification/opixray_yolov5n.pt \
  --batch-size 64 \
  --fgsm-eps 0.005 0.01 0.02 \
  --pgd-eps 0.01 0.02 \
  --pgd-alpha 0.002 \
  --pgd-steps 20 \
  --defense smoothing \
  --smoothing-kernel 3 \
  --output experiments/yolov5_classification/metrics/opixray_attack_smoothing_report.csv
```

## 9) Ket qua mong doi

- Khi tang `eps`, do chinh xac duoi tan cong giam dan.
- PGD thuong manh hon FGSM tai cung muc `eps`.
- C&W co the tao nhieu nho theo chuan L2 nhung ton thoi gian hon, nen phu hop lam thi nghiem bo sung tren subset.
- `attack_success_rate` tang khi tan cong hieu qua hon.
- `avg_l2`, `avg_linf`, `avg_mse`, `avg_mae` cho biet muc nhieu trung binh cua tung phuong phap.
- `avg_psnr` cang cao thi anh adversarial cang gan anh goc theo sai so pixel.
- `avg_ssim` cang gan 1 thi cau truc anh adversarial cang giong anh goc.
- Bao cao per-class giup nhan ra lop vu khi nao de bi tan cong hon.

## 10) Mo rong de tai

- Thu nghiem them CIFAR-10.
- So sanh voi adversarial training.
- Ve do thi accuracy theo `eps` de bao cao.

## 11) Cac muc nen bao cao

Khi thuyet trinh Do an 1, nen tap trung vao cac phan sau:

- Gioi thieu bai toan: nhan dang anh vu khi tren OPIXray va rui ro adversarial examples.
- Du lieu va tien xu ly: doc annotation, crop bounding box, resize anh ve `224x224`, chuyen thanh tensor.
- Mo hinh baseline: `YOLOv5` (hoac `SimpleCNN`), Cross-Entropy loss, Adam optimizer, train clean model.
- Phuong phap tan cong: FGSM, PGD, C&W L2.
- Input/output chung cua attack: input gom `model`, anh `x`, nhan `y`, tham so attack; output la anh doi khang `x_adv`.
- Chi so danh gia: `acc`, `attack_success_rate`, `avg_l2`, `avg_linf`, `avg_mse`, `avg_mae`, `avg_psnr`, `avg_ssim`, `runtime_seconds`, ket qua theo tung lop.
- Ket qua thuc nghiem: bang/plot accuracy theo `eps`, so sanh FGSM va PGD, ket qua C&W tren subset.
- Anh minh hoa: anh goc, anh adversarial, nhieu phong dai trong `experiments/yolov5_classification/figures/examples/`.
- Defense nhe: JPEG compression va spatial smoothing neu co thoi gian.
- Ket luan: `eps` tang thi accuracy giam, PGD thuong manh hon FGSM, C&W co tinh toi uu hoa nhung chay cham hon.

## 12) Cac lenh nen chay de lay ket qua bao cao

Train model (2-Stage Fine-tuning):

```bash
python advlab_examples/train_opixray.py \
  --data-dir data \
  --image-size 224 \
  --stage1-epochs 5 \
  --epochs 25 \
  --batch-size 64 \
  --lr 0.001 \
  --stage2-lr 0.0001 \
  --model yolov5n-cls \
  --save-path artifacts/checkpoints/yolov5_classification/opixray_yolov5n.pt
```

Chay FGSM + PGD chinh:

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path artifacts/checkpoints/yolov5_classification/opixray_yolov5n.pt \
  --batch-size 64 \
  --fgsm-eps 0.005 0.01 0.02 \
  --pgd-eps 0.01 0.02 \
  --pgd-alpha 0.002 \
  --pgd-steps 20 \
  --save-examples 6 \
  --output experiments/yolov5_classification/metrics/opixray_attack_report.csv
```

Chay C&W tren subset:

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path artifacts/checkpoints/yolov5_classification/opixray_yolov5n.pt \
  --batch-size 64 \
  --fgsm-eps \
  --pgd-eps \
  --cw \
  --cw-c 1.0 \
  --cw-kappa 0.0 \
  --cw-steps 100 \
  --cw-lr 0.01 \
  --cw-max-samples 128 \
  --save-examples 6 \
  --output experiments/yolov5_classification/metrics/opixray_cw_report.csv
```

Chay test tu dong:

```bash
python -m pytest -q
```

Neu co them thoi gian, chay defense JPEG:

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path artifacts/checkpoints/yolov5_classification/opixray_yolov5n.pt \
  --batch-size 64 \
  --fgsm-eps 0.005 0.01 0.02 \
  --pgd-eps 0.01 0.02 \
  --pgd-alpha 0.002 \
  --pgd-steps 20 \
  --defense jpeg \
  --jpeg-quality 75 \
  --output experiments/yolov5_classification/metrics/opixray_attack_jpeg_report.csv
```

Neu co them thoi gian, chay defense smoothing:

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path artifacts/checkpoints/yolov5_classification/opixray_yolov5n.pt \
  --batch-size 64 \
  --fgsm-eps 0.005 0.01 0.02 \
  --pgd-eps 0.01 0.02 \
  --pgd-alpha 0.002 \
  --pgd-steps 20 \
  --defense smoothing \
  --smoothing-kernel 3 \
  --output experiments/yolov5_classification/metrics/opixray_attack_smoothing_report.csv
```
