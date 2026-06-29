from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OPIXrayCropDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        class_to_idx: Dict[str, int],
        image_size: int = 224,
    ) -> None:
        self.image_dir = root / split / f"{split}_image"
        annotation_dir = root / split / f"{split}_annotation"
        self.class_to_idx = class_to_idx
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.samples: List[Tuple[Path, int, Tuple[int, int, int, int]]] = []

        if not self.image_dir.exists() or not annotation_dir.exists():
            raise FileNotFoundError(
                f"Cannot find OPIXray split '{split}' in: {root}. "
                "Expected *_image and *_annotation folders."
            )

        for ann_path in sorted(annotation_dir.glob("*.txt")):
            for filename, label, bbox in _parse_annotation_file(ann_path):
                if label not in self.class_to_idx:
                    continue
                self.samples.append((self.image_dir / filename, self.class_to_idx[label], bbox))

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {annotation_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label_idx, bbox = self.samples[idx]

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            crop = _safe_crop(image, bbox)

        return self.transform(crop), label_idx


def _parse_annotation_file(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            filename, label = parts[0], parts[1]
            x1, y1, x2, y2 = (int(float(v)) for v in parts[2:])
            rows.append((filename, label, (x1, y1, x2, y2)))
    return rows


def _safe_crop(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(1, min(x2, width))
    y2 = max(1, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return image
    return image.crop((x1, y1, x2, y2))


def _collect_opixray_classes(root: Path):
    labels = set()
    for split in ("train", "test"):
        ann_dir = root / split / f"{split}_annotation"
        for ann_path in ann_dir.glob("*.txt"):
            for _, label, _ in _parse_annotation_file(ann_path):
                labels.add(label)
    if not labels:
        raise RuntimeError(f"Cannot infer OPIXray labels from {root}")
    return sorted(labels)


def get_opixray_loaders(
    batch_size: int = 64,
    data_dir: str = "data",
    image_size: int = 224,
):
    root = Path(data_dir) / "OPIXray"
    class_names = _collect_opixray_classes(root)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_ds = OPIXrayCropDataset(root, "train", class_to_idx, image_size=image_size)
    test_ds = OPIXrayCropDataset(root, "test", class_to_idx, image_size=image_size)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, class_names
