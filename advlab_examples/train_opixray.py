import argparse
import json
from pathlib import Path

import torch

from advlab.data import get_opixray_loaders
from advlab.models import SimpleCNN
from advlab.trainer import evaluate_clean, train_one_epoch
from advlab.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train clean model on OPIXray")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-path", type=str, default="checkpoints/opixray_cnn.pt")
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Optional JSON path for human-readable checkpoint metadata.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_loader, test_loader, class_names = get_opixray_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        image_size=args.image_size,
    )
    in_channels = 3
    num_classes = len(class_names)
    model = SimpleCNN(in_channels=in_channels, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Dataset=OPIXray | num_classes={num_classes} | in_channels={in_channels}")
    print(f"OPIXray labels: {class_names}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate_clean(model, test_loader, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "dataset": "opixray",
        "in_channels": in_channels,
        "num_classes": num_classes,
        "class_names": class_names,
    }
    torch.save(checkpoint, save_path)
    print(f"Saved model to: {save_path}")

    metadata_path = Path(args.metadata_path) if args.metadata_path else save_path.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint_path": str(save_path),
        "dataset": checkpoint["dataset"],
        "in_channels": in_channels,
        "num_classes": num_classes,
        "class_names": class_names,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "train_args": vars(args),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved readable metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
