"""Train the NumPy DnCNN on the Duke training patches.

Training data : (a) 4000 noisy/clean 15x15 patch pairs from 10 Duke *training*
                    B-scans (not part of the 18 evaluation images), and
                (b) random 40x40 noisy/clean patches from development images 01-03
                    (gives the network background / vitreous context, which the
                    retina-only 15x15 patches lack).
Model selection: best PSNR on development image 04.
Test images 05-18 are NEVER seen during training or model selection.
Output        : models/dncnn_numpy.npz, results/dncnn_training_curve.png

Usage:  python train_dncnn.py [--epochs 40] [--depth 6] [--features 32]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from oct_denoise.data import PROJECT_ROOT, VAL_IDS, load_split, load_train_patches  # noqa: E402
from oct_denoise.dncnn import Adam, DnCNN, augment  # noqa: E402
from oct_denoise.metrics import psnr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--features", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_every", type=int, default=2)
    ap.add_argument("--big_patch", type=int, default=40)
    ap.add_argument("--big_batches", type=int, default=60, help="40x40 batches per epoch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    xn, xc = load_train_patches()
    xn, xc = xn[:, None], xc[:, None]  # (N,1,15,15)
    dev = load_split(VAL_IDS[:3])          # images 01-03 -> extra training patches
    val = load_split(VAL_IDS[3:])          # image 04 -> model selection
    print(f"train patches: {xn.shape[0]} (15x15) + random {args.big_patch}x{args.big_patch} "
          f"patches from {[p.image_id for p in dev]}  |  model selection on {[p.image_id for p in val]}")

    def big_batch(n=16):
        P = args.big_patch
        xs, ys = [], []
        for _ in range(n):
            q = dev[rng.integers(len(dev))]
            r = rng.integers(0, q.noisy.shape[0] - P)
            c = rng.integers(0, q.noisy.shape[1] - P)
            xs.append(q.noisy[r:r + P, c:c + P])
            ys.append(q.clean[r:r + P, c:c + P])
        return np.stack(xs)[:, None], np.stack(ys)[:, None]

    model = DnCNN(args.depth, args.features, seed=args.seed)
    opt = Adam(model, lr=args.lr)
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    best_path = os.path.join(PROJECT_ROOT, "models", "dncnn_numpy.npz")

    hist = {"epoch": [], "train_loss": [], "val_psnr": []}
    best, t0 = -1.0, time.time()
    for ep in range(1, args.epochs + 1):
        if ep == int(0.6 * args.epochs) or ep == int(0.85 * args.epochs):
            opt.lr *= 0.3  # step learning-rate decay
        perm = rng.permutation(xn.shape[0])
        small = [perm[s:s + args.batch] for s in range(0, len(perm) - args.batch + 1, args.batch)]
        jobs = [("small", i) for i in small] + [("big", None)] * args.big_batches
        order = rng.permutation(len(jobs))
        losses = []
        for j in order:
            kind, idx = jobs[j]
            x, y = (xn[idx], xc[idx]) if kind == "small" else big_batch()
            x, y = augment(x, y, rng)
            out = model.forward(x)
            diff = out - y
            losses.append(float(np.mean(diff ** 2)))
            model.backward(2.0 * diff / diff.size)
            opt.step()
        hist["epoch"].append(ep)
        hist["train_loss"].append(float(np.mean(losses)))
        msg = f"epoch {ep:3d}  loss {np.mean(losses):.5f}"
        if ep % args.val_every == 0 or ep == args.epochs:
            v = float(np.mean([psnr(p.clean, model.denoise(p.noisy)) for p in val]))
            hist["val_psnr"].append(v)
            msg += f"  val PSNR {v:.2f} dB"
            if v > best:
                best = v
                model.save(best_path)
                msg += "  (saved)"
        else:
            hist["val_psnr"].append(None)
        print(msg + f"  [{time.time() - t0:.0f}s]", flush=True)

    with open(os.path.join(PROJECT_ROOT, "results", "dncnn_training_history.json"), "w") as f:
        json.dump({"args": vars(args), "best_val_psnr": best, **hist}, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist["epoch"], hist["train_loss"], color="#2a6fdb")
    ax[0].set(xlabel="epoch", ylabel="training MSE", title="DnCNN training loss")
    ve = [e for e, v in zip(hist["epoch"], hist["val_psnr"]) if v is not None]
    vv = [v for v in hist["val_psnr"] if v is not None]
    ax[1].plot(ve, vv, "o-", color="#d9622b")
    ax[1].set(xlabel="epoch", ylabel="PSNR (dB)", title="Model-selection PSNR (image 04)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PROJECT_ROOT, "results", "dncnn_training_curve.png"), dpi=150)
    print(f"best validation PSNR {best:.2f} dB -> {best_path}")


if __name__ == "__main__":
    main()
