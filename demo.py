"""Denoise any OCT B-scan with any method.

Examples
  python demo.py                                   # image 11 of the dataset, all methods
  python demo.py --input my_scan.png --method NLM  # your own image, one method
  python demo.py --list                            # list available methods

Output: results/demo/<name>_<method>.png  (+ a side-by-side comparison figure)
"""
import argparse
import json
import os
import sys

import numpy as np
from skimage import io

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from oct_denoise import filters as F  # noqa: E402
from oct_denoise import metrics as M  # noqa: E402
from oct_denoise.data import PROJECT_ROOT, load_pair  # noqa: E402
from oct_denoise.dncnn import DnCNN  # noqa: E402

DNCNN = "DnCNN"
MODEL = os.path.join(PROJECT_ROOT, "models", "dncnn_numpy.npz")
PARAMS = os.path.join(PROJECT_ROOT, "results", "best_params.json")


def run(method, img, params):
    if method == DNCNN:
        return DnCNN.load(MODEL).denoise(img)
    return F.METHODS[method][0](img, **params.get(method, {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="path to a grayscale OCT image (default: dataset image 11)")
    ap.add_argument("--image_id", default="11", help="dataset image id 01-18 (if --input not given)")
    ap.add_argument("--method", default="all", help="method name or 'all'")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    names = list(F.METHODS) + ([DNCNN] if os.path.exists(MODEL) else [])
    if args.list:
        print("\n".join(names))
        return
    params = json.load(open(PARAMS)) if os.path.exists(PARAMS) else {}

    if args.input:
        img = io.imread(args.input)
        img = img[..., 0] if img.ndim == 3 else img
        img = img.astype(np.float32) / (255.0 if img.max() > 1 else 1.0)
        clean, tag = None, os.path.splitext(os.path.basename(args.input))[0]
    else:
        p = load_pair(args.image_id)
        img, clean, tag = p.noisy, p.clean, f"image_{args.image_id}"

    todo = names if args.method == "all" else [args.method]
    out_dir = os.path.join(PROJECT_ROOT, "results", "demo")
    os.makedirs(out_dir, exist_ok=True)
    outs = {}
    for m in todo:
        y = run(m, img, params)
        outs[m] = y
        fname = os.path.join(out_dir, f"{tag}_{''.join(c if c.isalnum() else '_' for c in m)}.png")
        io.imsave(fname, (y * 255).astype(np.uint8), check_contrast=False)
        line = f"{m:18s} -> {os.path.relpath(fname)}"
        if clean is not None:
            line += f"   PSNR {M.psnr(clean, y):.2f}  SSIM {M.ssim(clean, y):.3f}  EPI {M.epi(clean, y):.3f}"
        print(line)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("Input (noisy)", img)] + list(outs.items()) + ([("Averaged reference", clean)] if clean is not None else [])
    cols = min(3, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 2.4), squeeze=False)
    for a, (t, im) in zip(ax.ravel(), panels):
        a.imshow(im, cmap="gray", vmin=0, vmax=1)
        a.set_title(t)
        a.axis("off")
    for a in ax.ravel()[len(panels):]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_comparison.png"), dpi=130)
    print("comparison figure ->", os.path.relpath(os.path.join(out_dir, f"{tag}_comparison.png")))


if __name__ == "__main__":
    main()
