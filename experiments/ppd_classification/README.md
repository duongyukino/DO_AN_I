# Nhánh PPD classification

Đây là pipeline nghiên cứu phòng thủ palette trên crop OPIXray.

- Cấu hình và ma trận: `configs/` (JSON/CSV là manifest tham khảo, không tự chạy).
- Checkpoint: `../../artifacts/checkpoints/ppd/`.
- Kết quả CSV và lịch sử train: `metrics/`.
- Biểu đồ và ảnh ghép sau defense: `figures/`.
- Cache/palette đầu vào: `../../data/processed/`.

Các checkpoint cũ được train riêng theo defense có trong tên file. Các file cấu hình mới đề xuất phép so sánh có kiểm soát hơn: dùng checkpoint clean `*_none.pt`, rồi chọn defense lúc đánh giá bằng `attack_eval.py --defense <name>`. CSV có hậu tố `_as_<defense>` và vẫn nằm trong `metrics/`. File JSON/CSV trong `configs/` là manifest để ghi protocol, không tự được script nạp.


Ví dụ đánh giá cùng ResNet-50 qua PPD32m bằng FGSM, PGD, C&W:

```bash
python attack_eval.py --ckpt artifacts/checkpoints/ppd/resnet50_none.pt --defense ppd32m --attacks fgsm pgd cw --eps 2 4 8 --pgd_steps 30 --cw_steps 100 --n 200 --n_cw 100 --tag _main
```

Thêm `--adaptive --median_grad mean` để chạy lượt adaptive; đổi `--defense` để đánh giá defense khác. Dùng cùng checkpoint clean để so sánh ảnh sau defense bằng `visualize_def.py --ckpt artifacts/checkpoints/ppd/resnet50_none.pt --defense ppd32m`. Với EfficientNet-B0, train checkpoint clean trước: `python ppd_resnet.py train --arch efficientnet_b0 --defense none`.
