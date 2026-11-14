import random
from pathlib import Path

root = Path("OPIXray")
names = [l.strip() for l in open(root / "train/train_knife.txt", encoding="utf-8") if l.strip()]
names = [Path(n).stem for n in names]

random.Random(42).shuffle(names)
n_val = round(len(names) * 0.1)
val, train = sorted(names[:n_val]), sorted(names[n_val:])

out = root / "splits"
out.mkdir(exist_ok=True)
(out / "train.txt").write_text("\n".join(train), encoding="utf-8")
(out / "val.txt").write_text("\n".join(val), encoding="utf-8")
print("train:", len(train), "| val:", len(val), "| giao nhau:", len(set(train) & set(val)))