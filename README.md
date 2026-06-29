 # Adversarial Attack Research Project (OPIXray)

Du an mau cho de tai: **"Nghien cuu tan cong mau doi khang len mo hinh hoc may"**.

Muc tieu:
- Hieu duoc ban chat tan cong doi khang va cac kieu tan cong co ban.
- Trien khai va so sanh cac phuong phap: **FGSM**, **PGD** va **C&W L2**.
- Thuc hien thu nghiem thuc te tren bai toan phan loai anh OPIXray.

## 1) Cau truc du an

```
adversarial-attack-project/
├── advlab/                      # Thu vien loi
│   ├── attacks.py               # FGSM, PGD, C&W L2
│   ├── data.py                  # DataLoader OPIXray
│   ├── defenses.py              # JPEG, smoothing defense nhe
│   ├── metrics.py               # ASR, L2/Linf, PSNR/SSIM, per-class metrics
│   ├── models.py                # CNN model linh hoat
│   ├── trainer.py               # Train/eval clean
│   ├── utils.py                 # set_seed, device
│   └── visualization.py         # Luu anh goc/adversarial/nhieu
├── advlab_examples/             # Script chay thu nghiem
│   ├── train_opixray.py
│   ├── inspect_checkpoint.py    # Xuat thong tin checkpoint ra JSON
│   └── attack_opixray.py        # Danh gia FGSM, PGD, C&W L2
├── checkpoints/                 # Luu model da train
├── results/                     # Luu csv va anh minh hoa
├── docs/
│   └── experiment_guide.md
├── tests/
│   └── test_attacks.py          # Kiem thu FGSM, PGD, C&W L2
└── setup.py
```

## 2) Cai dat

```bash
cd adversarial-attack-project
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision tqdm pandas
pip install -e .
```

## 3) Cau truc dataset OPIXray

Dat du lieu vao `data/OPIXray` voi cau truc:
- `train/train_image`, `train/train_annotation`
- `test/test_image`, `test/test_annotation`

## 4) Train mo hinh clean (OPIXray)

```bash
python advlab_examples/train_opixray.py \
  --data-dir data \
  --image-size 224 \
  --epochs 5 \
  --batch-size 64 \
  --lr 0.001 \
  --save-path checkpoints/opixray_cnn.pt
```

File `.pt` la checkpoint nhi phan cua PyTorch, khong nen mo bang text editor/Excel. Khi train xong,
script se luu them metadata de doc bang nguoi dung tai `checkpoints/opixray_cnn.json`.

Neu muon xem thong tin cua checkpoint da co:

```bash
python advlab_examples/inspect_checkpoint.py \
  --model-path checkpoints/opixray_cnn.pt \
  --output checkpoints/opixray_cnn_info.json
```

## 5) Danh gia duoi FGSM + PGD (OPIXray)

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path checkpoints/opixray_cnn.pt \
  --batch-size 64 \
  --fgsm-eps 0.005 0.01 0.02 \
  --pgd-eps 0.01 0.02 \
  --pgd-alpha 0.002 \
  --pgd-steps 20 \
  --save-examples 6 \
  --output results/opixray_attack_report.csv
```

Lenh tren tao them:
- `results/opixray_attack_report.csv`: ket qua tong hop voi `acc`, `attack_success_rate`, `avg_l2`, `avg_linf`, `avg_mse`, `avg_mae`, `avg_psnr`, `avg_ssim`, `runtime_seconds`.
- `results/opixray_attack_report_per_class.csv`: ket qua theo tung lop vu khi.
- `results/examples/`: mot so anh minh hoa gom anh goc, anh adversarial va nhieu phong dai.

## 6) Danh gia them voi C&W L2 (OPIXray)

C&W la tan cong dua tren toi uu hoa, chay cham hon FGSM/PGD. Nen chay tren mot tap con nho
bang `--cw-max-samples` de phu hop laptop ca nhan.

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path checkpoints/opixray_cnn.pt \
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
  --output results/opixray_attack_report.csv
```

Neu chi muon test nhanh C&W:

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 128 \
  --model-path checkpoints/opixray_cnn.pt \
  --batch-size 32 \
  --fgsm-eps \
  --pgd-eps \
  --cw \
  --cw-steps 50 \
  --cw-max-samples 32 \
  --output results/opixray_cw_quick.csv
```

## 7) Danh gia voi defense nhe

Co the bat tien xu ly nhe khi danh gia anh adversarial. Day khong phai phong thu manh nhu adversarial training,
nhung phu hop lam baseline nhanh trong Do an 1.

JPEG compression:

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path checkpoints/opixray_cnn.pt \
  --batch-size 64 \
  --fgsm-eps 0.005 0.01 0.02 \
  --pgd-eps 0.01 0.02 \
  --pgd-alpha 0.002 \
  --pgd-steps 20 \
  --defense jpeg \
  --jpeg-quality 75 \
  --output results/opixray_attack_jpeg_report.csv
```

Spatial smoothing:

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path checkpoints/opixray_cnn.pt \
  --batch-size 64 \
  --fgsm-eps 0.005 0.01 0.02 \
  --pgd-eps 0.01 0.02 \
  --pgd-alpha 0.002 \
  --pgd-steps 20 \
  --defense smoothing \
  --smoothing-kernel 3 \
  --output results/opixray_attack_smoothing_report.csv
```

## 8) Ket qua mong doi

- Khi tang `eps`, do chinh xac duoi tan cong giam dan.
- PGD thuong manh hon FGSM tai cung muc `eps`.
- C&W co the tao nhieu nho theo chuan L2 nhung ton thoi gian hon, nen phu hop lam thi nghiem bo sung tren subset.
- `attack_success_rate` tang khi tan cong hieu qua hon.
- `avg_l2`, `avg_linf`, `avg_mse`, `avg_mae` cho biet muc nhieu trung binh cua tung phuong phap.
- `avg_psnr` cang cao thi anh adversarial cang gan anh goc theo sai so pixel.
- `avg_ssim` cang gan 1 thi cau truc anh adversarial cang giong anh goc.
- Bao cao per-class giup nhan ra lop vu khi nao de bi tan cong hon.

## 9) Mo rong de tai

- Thu nghiem them CIFAR-10.
- So sanh voi adversarial training.
- Ve do thi accuracy theo `eps` de bao cao.

## 10) Cac muc nen bao cao

Khi thuyet trinh Do an 1, nen tap trung vao cac phan sau:

- Gioi thieu bai toan: nhan dang anh vu khi tren OPIXray va rui ro adversarial examples.
- Du lieu va tien xu ly: doc annotation, crop bounding box, resize anh ve `224x224`, chuyen thanh tensor.
- Mo hinh baseline: `SimpleCNN`, Cross-Entropy loss, Adam optimizer, train clean model.
- Phuong phap tan cong: FGSM, PGD, C&W L2.
- Input/output chung cua attack: input gom `model`, anh `x`, nhan `y`, tham so attack; output la anh doi khang `x_adv`.
- Chi so danh gia: `acc`, `attack_success_rate`, `avg_l2`, `avg_linf`, `avg_mse`, `avg_mae`, `avg_psnr`, `avg_ssim`, `runtime_seconds`, ket qua theo tung lop.
- Ket qua thuc nghiem: bang/plot accuracy theo `eps`, so sanh FGSM va PGD, ket qua C&W tren subset.
- Anh minh hoa: anh goc, anh adversarial, nhieu phong dai trong `results/examples/`.
- Defense nhe: JPEG compression va spatial smoothing neu co thoi gian.
- Ket luan: `eps` tang thi accuracy giam, PGD thuong manh hon FGSM, C&W co tinh toi uu hoa nhung chay cham hon.

## 11) Cac lenh nen chay de lay ket qua bao cao

Train model:

```bash
python advlab_examples/train_opixray.py \
  --data-dir data \
  --image-size 224 \
  --epochs 5 \
  --batch-size 64 \
  --lr 0.001 \
  --save-path checkpoints/opixray_cnn.pt
```

Chay FGSM + PGD chinh:

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path checkpoints/opixray_cnn.pt \
  --batch-size 64 \
  --fgsm-eps 0.005 0.01 0.02 \
  --pgd-eps 0.01 0.02 \
  --pgd-alpha 0.002 \
  --pgd-steps 20 \
  --save-examples 6 \
  --output results/opixray_attack_report.csv
```

Chay C&W tren subset:

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path checkpoints/opixray_cnn.pt \
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
  --output results/opixray_cw_report.csv
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
  --model-path checkpoints/opixray_cnn.pt \
  --batch-size 64 \
  --fgsm-eps 0.005 0.01 0.02 \
  --pgd-eps 0.01 0.02 \
  --pgd-alpha 0.002 \
  --pgd-steps 20 \
  --defense jpeg \
  --jpeg-quality 75 \
  --output results/opixray_attack_jpeg_report.csv
```

Neu co them thoi gian, chay defense smoothing:

```bash
python advlab_examples/attack_opixray.py \
  --data-dir data \
  --image-size 224 \
  --model-path checkpoints/opixray_cnn.pt \
  --batch-size 64 \
  --fgsm-eps 0.005 0.01 0.02 \
  --pgd-eps 0.01 0.02 \
  --pgd-alpha 0.002 \
  --pgd-steps 20 \
  --defense smoothing \
  --smoothing-kernel 3 \
  --output results/opixray_attack_smoothing_report.csv
```

