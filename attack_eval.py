"""Đánh giá độ bền dưới tấn công, kịch bản OBLIVIOUS, cách A.

Với mỗi checkpoint (backbone + phòng thủ đã dùng khi huấn luyện):
  1. Sinh ảnh đối kháng bằng CHÍNH trọng số của mô hình, nhưng TẮT phòng thủ (kẻ tấn công không biết phòng thủ).
  2. Đưa ảnh đối kháng qua mô hình đầy đủ (phòng thủ BẬT) và đo accuracy.
  3. Đo thêm accuracy của backbone khi phòng thủ tắt (kiểm tra tấn công có đủ mạnh không).

Ví dụ:
  python attack_eval.py --ckpt artifacts/checkpoints/ppd/resnet50_none.pt --n 20 --n_cw 20 --attacks fgsm pgd   # chạy thử, đo thời gian
  python attack_eval.py --ckpt artifacts/checkpoints/ppd/resnet50_ppd32.pt
  python attack_eval.py --ckpt artifacts/checkpoints/ppd/resnet50_ppd32.pt --split test                          # chỉ làm ở bước cuối
  python attack_eval.py --ckpt artifacts/checkpoints/ppd/resnet50_ppd32.pt --adaptive --attacks pgd --pgd_steps 30   # kẻ tấn công biết phòng thủ (BPDA)

Cột kết quả:
  clean_acc   accuracy sạch của mô hình đầy đủ trên tập con
  robust_acc  accuracy của mô hình đầy đủ (phòng thủ bật) trên ảnh đối kháng, tính trên toàn bộ tập con
  asr         tỷ lệ mẫu đúng ở ảnh sạch nhưng sai sau tấn công (mô hình đầy đủ)
  bb_acc      accuracy của backbone khi TẮT phòng thủ trên ảnh đối kháng (white-box; phải thấp)
  l2, linf    chuẩn L2 (thang [0,1]) và L-vô-cùng (thang 255) trung bình của nhiễu
  post_l2     L2 của chênh lệch giữa ảnh đối kháng và ảnh sạch SAU KHI qua phòng thủ (thang [0,1]);
              so với l2 để biết phòng thủ xóa bao nhiêu nhiễu (với 'none' thì bằng l2)
  px_changed  tỷ lệ pixel có ít nhất một kênh khác nhau sau phòng thủ (có ý nghĩa với ppd/bit; với jpeg/blur
              gần như luôn bằng 1)
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import ppd_resnet as P

BASE_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- tấn công
# model(x) nhận ảnh [0,1] và trả logits. Với kịch bản oblivious, model phải TẮT phòng thủ.
def _grad(model, x, y):
    x = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    return torch.autograd.grad(loss, x)[0]


def sign_noise(x, eps):
    """Đối chứng: nhiễu ngẫu nhiên cùng độ lớn L-vô-cùng."""
    return (x + eps * torch.sign(torch.randn_like(x))).clamp(0, 1).detach()


def fgsm_attack(model, x, y, eps):
    return (x + eps * _grad(model, x, y).sign()).clamp(0, 1).detach()


def pgd_attack(model, x, y, eps, alpha, steps):
    x0 = x.detach()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa = xa.detach() + alpha * _grad(model, xa, y).sign()
        xa = (x0 + (xa - x0).clamp(-eps, eps)).clamp(0, 1)
    return xa.detach()


def cw_l2_attack(model, x, y, c=1.0, kappa=0.0, steps=100, lr=0.01):
    """C&W L2 không nhắm đích, tối ưu trong không gian tanh, giữ ảnh tốt nhất đã thành công.
    Thành công = biên (logit lớp sai lớn nhất - logit lớp đúng) >= kappa và > 0."""
    x0 = x.detach()
    xs = (x0 * 2 - 1).clamp(-0.999999, 0.999999)
    w = (0.5 * torch.log((1 + xs) / (1 - xs))).requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    onehot = F.one_hot(y, num_classes=len(P.CLASSES)).float()
    best_adv = x0.clone()
    best_l2 = torch.full((len(x0),), float("inf"), device=x0.device)
    for _ in range(steps):
        adv = 0.5 * (torch.tanh(w) + 1)
        logits = model(adv)
        correct = (onehot * logits).sum(1)
        other = ((1 - onehot) * logits - onehot * 1e4).max(1).values
        margin = other - correct
        l2sq = (adv - x0).flatten(1).pow(2).sum(1)
        loss = (l2sq + c * torch.clamp(kappa - margin, min=0)).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            ok = (margin >= kappa) & (margin > 0) & (l2sq < best_l2)
            best_l2[ok] = l2sq[ok]
            best_adv[ok] = adv.detach()[ok]
    final = (0.5 * (torch.tanh(w) + 1)).detach()
    success = torch.isfinite(best_l2)
    best_adv[~success] = final[~success]
    return best_adv.detach()


# ----------------------------------------------------------------------------- đánh giá
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint, e.g. artifacts/checkpoints/ppd/resnet50_ppd16m.pt")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--defense", default=None,
                    help="Override the checkpoint defense for inference, e.g. none, jpeg75, blur1.0, bit3, ppd32m")
    ap.add_argument("--attacks", nargs="+", default=["noise", "fgsm", "pgd", "cw"],
                    choices=["noise", "fgsm", "pgd", "cw"])
    ap.add_argument("--eps", nargs="+", type=float, default=[2, 4, 8], help="đơn vị /255")
    ap.add_argument("--pgd_steps", type=int, default=10)
    ap.add_argument("--kappas", nargs="+", type=float, default=[0, 10])
    ap.add_argument("--cw_steps", type=int, default=100)
    ap.add_argument("--cw_c", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=500, help="số crop cho noise/FGSM/PGD")
    ap.add_argument("--n_cw", type=int, default=100, help="số crop cho C&W (tập con của --n)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quantize", action="store_true", help="làm tròn ảnh đối kháng về 8 bit trước khi đưa qua phòng thủ")
    ap.add_argument("--tag", default="", help="hậu tố tên file CSV, tránh ghi đè kết quả của lần chạy khác")
    ap.add_argument("--median_grad", default="true", choices=["true", "mean", "identity"],
                    help="với ppdNm: gradient qua median (true = thật, rất chậm trên MPS; mean = lọc trung bình; identity = thẳng)")
    ap.add_argument("--adaptive", action="store_true",
                    help="kẻ tấn công BIẾT phòng thủ: gradient đi thẳng qua phép biến đổi không khả vi (BPDA)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    ckpt_path = Path(a.ckpt).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = BASE_DIR / ckpt_path
    ck = torch.load(ckpt_path, map_location="cpu")
    arch, dep_name = ck.get("arch", "resnet50"), (a.defense or ck["defense"])
    net = P.Net(arch, nn.Identity(), pretrained=False)
    net.backbone.load_state_dict(ck["state"])
    net.to(P.DEV).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    off, deployed = nn.Identity(), P.make_defense(dep_name).to(P.DEV)
    if hasattr(deployed, "ste"):
        deployed.ste = a.adaptive            # chỉ đổi gradient, không đổi giá trị đầu ra
    if hasattr(deployed, "median_grad"):
        deployed.median_grad = a.median_grad
    att_def = deployed if a.adaptive else off   # phòng thủ mà kẻ tấn công nhìn thấy
    tag = "_adaptive" if a.adaptive else ""
    defense_tag = f"_as_{dep_name}" if a.defense is not None else ""
    csv_path = P.METRICS / f"attack_{ckpt_path.stem}{defense_tag}_{a.split}{tag}{a.tag}.csv"

    d = torch.load(P.CACHE / f"{a.split}.pt")
    perm = torch.randperm(len(d["x"]), generator=torch.Generator().manual_seed(a.seed))
    print(f"Thiết bị: {P.DEV} | {a.ckpt} ({arch}, phòng thủ '{dep_name}') | {a.split}: "
          f"{min(a.n, len(perm))} crop, C&W {min(a.n_cw, len(perm))} crop | quantize={a.quantize}")

    rows = []

    def run(label, param, idx, make_adv):
        t0 = time.time()
        s = dict(clean=0, rob=0, asr=0, bb=0, l2=0.0, linf=0.0, post=0.0, pxc=0.0)
        for i in range(0, len(idx), a.bs):
            j = idx[i:i + a.bs]
            xb, yb = (d["x"][j].float() / 255).to(P.DEV), d["y"][j].to(P.DEV)
            net.defense = att_def
            xa = make_adv(xb, yb)
            if a.quantize:
                xa = (xa * 255).round() / 255
            with torch.no_grad():
                net.defense = deployed
                pc, pa = net(xb).argmax(1), net(xa).argmax(1)
                net.defense = off
                pb = net(xa).argmax(1)
                dd = deployed(xa) - deployed(xb)                              # chênh lệch còn lại SAU phòng thủ
                post = dd.flatten(1).norm(dim=1)
                pxc = (dd.abs().amax(1) > 1e-6).float().flatten(1).mean(1)   # tỷ lệ pixel bị khác
            dl = (xa - xb).flatten(1)
            s["clean"] += (pc == yb).sum().item()
            s["rob"] += (pa == yb).sum().item()
            s["asr"] += ((pc == yb) & (pa != yb)).sum().item()
            s["bb"] += (pb == yb).sum().item()
            s["post"] += post.sum().item()
            s["pxc"] += pxc.sum().item()
            s["l2"] += dl.norm(dim=1).sum().item()
            s["linf"] += dl.abs().max(1).values.sum().item() * 255
            if P.DEV.type == "mps":
                torch.mps.empty_cache()          # giảm phân mảnh bộ nhớ giữa các batch
        n = len(idx)
        row = {"ckpt": ckpt_path.stem, "defense": dep_name, "attack": label, "param": param, "n": n,
               "clean_acc": round(s["clean"] / n, 4), "robust_acc": round(s["rob"] / n, 4),
               "asr": round(s["asr"] / max(s["clean"], 1), 4), "bb_acc": round(s["bb"] / n, 4),
               "l2": round(s["l2"] / n, 3), "linf": round(s["linf"] / n, 2),
               "post_l2": round(s["post"] / n, 3), "px_changed": round(s["pxc"] / n, 4)}
        rows.append(row)
        print(f"{label:>6} {param:>5} | clean {row['clean_acc']:.3f} | robust {row['robust_acc']:.3f} | "
              f"asr {row['asr']:.3f} | bb {row['bb_acc']:.3f} | L2 {row['l2']:.2f} | "
              f"Linf {row['linf']:.1f} | post-L2 {row['post_l2']:.2f} | px_changed {row['px_changed']:.3f} | {time.time() - t0:.0f}s")
        P.METRICS.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return row

    idx = perm[:a.n]
    for name in a.attacks:
        if name == "cw":
            for k in a.kappas:
                run("cw" + ("_adapt" if a.adaptive else ""), f"k{k:g}", perm[:a.n_cw],
                    lambda x, y: cw_l2_attack(net, x, y, a.cw_c, k, a.cw_steps))
            continue
        for e in a.eps:
            eps = e / 255
            fn = {"noise": lambda x, y: sign_noise(x, eps),
                  "fgsm": lambda x, y: fgsm_attack(net, x, y, eps),
                  "pgd": lambda x, y: pgd_attack(net, x, y, eps, eps / 4, a.pgd_steps)}[name]
            r = run(name + ("_adapt" if a.adaptive and name != "noise" else ""), f"{e:g}", idx, fn)
            if name == "pgd" and e == 4 and not a.adaptive and r["bb_acc"] > 0.10:
                print("  CẢNH BÁO: PGD ở eps=4/255 mà backbone (tắt phòng thủ) vẫn đúng "
                      f"{r['bb_acc']:.0%}. Tấn công có thể đang lỗi, hãy kiểm tra trước khi tin các số khác.")
    print(f"Đã lưu {csv_path}")


if __name__ == "__main__":
    main()
