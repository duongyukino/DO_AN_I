import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a PyTorch checkpoint")
    parser.add_argument("--model-path", type=str, default="checkpoints/opixray_cnn.pt")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    state = torch.load(model_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
        metadata = {
            "checkpoint_path": str(model_path),
            "dataset": state.get("dataset"),
            "in_channels": state.get("in_channels"),
            "num_classes": state.get("num_classes"),
            "class_names": state.get("class_names"),
            "num_tensors": len(state_dict),
            "num_parameters": int(sum(t.numel() for t in state_dict.values())),
            "tensors": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": int(tensor.numel()),
                }
                for name, tensor in state_dict.items()
            },
        }
    else:
        metadata = {
            "checkpoint_path": str(model_path),
            "format": "raw_state_dict",
            "num_tensors": len(state),
            "num_parameters": int(sum(t.numel() for t in state.values())),
            "tensors": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": int(tensor.numel()),
                }
                for name, tensor in state.items()
            },
        }

    output = Path(args.output) if args.output else model_path.with_name(f"{model_path.stem}_info.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in metadata.items() if k != "tensors"}, indent=2, ensure_ascii=False))
    print(f"Saved readable checkpoint info to: {output}")


if __name__ == "__main__":
    main()
