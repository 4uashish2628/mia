"""Figures for the report."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402
from skimage import io  # noqa: E402
from skimage.feature import canny  # noqa: E402

from . import filters as F  # noqa: E402
from . import metrics as M  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 110})
REF = "Averaged (reference)"
BASELINE_COLOR, EDGE_COLOR, DL_COLOR = "#9aa3ad", "#2a6fdb", "#d9622b"


def _color(m):
    if m.startswith("DnCNN"):
        return DL_COLOR
    if m in F.METHODS:
        return EDGE_COLOR if F.METHODS[m][2] else BASELINE_COLOR
    return "#333333"


def _methods(outs):
    return [k for k in outs if k not in ("Noisy", REF)]


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved", os.path.relpath(path))


def _retina_window(clean, h=130, w=320):
    """Crop window centred on the brightest retinal band (around the fovea)."""
    sm = ndimage.gaussian_filter(clean, 4)
    r = int(np.argmax(sm.mean(1)))
    H, W = clean.shape
    r0 = int(np.clip(r - h // 2, 0, H - h))
    c0 = int(np.clip(W // 2 - w // 2, 0, W - w))
    return r0, c0, h, w


# --------------------------------------------------------------------------- #
def fig_dataset(q, path):
    bg, sig = M.select_rois(q.clean)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
    for a, im, t in zip(ax, [q.noisy, q.clean],
                        [f"Noisy single B-scan (image {q.image_id})",
                         "Registered & averaged B-scan (reference)"]):
        a.imshow(im, cmap="gray", vmin=0, vmax=1)
        a.set_title(t)
        a.axis("off")
        for (r, c, h, w) in bg:
            a.add_patch(Rectangle((c, r), w, h, fill=False, ec="red", lw=1.2))
        for (r, c, h, w) in sig:
            a.add_patch(Rectangle((c, r), w, h, fill=False, ec="lime", lw=1.2))
    ax[1].text(5, q.clean.shape[0] - 8, "red = background ROI, green = homogeneous signal ROIs "
               "(used for SNR / CNR / ENL)", color="w", fontsize=7)
    _save(fig, path)


def fig_noise_analysis(pairs, path):
    res_bg, res_sig, means, stds = [], [], [], []
    for q in pairs:
        r = q.noisy.astype(np.float64) - q.clean
        bg, sig = M.select_rois(q.clean)
        for (rr, c, h, w) in bg:
            res_bg.append(r[rr:rr + h, c:c + w].ravel())
        for (rr, c, h, w) in sig:
            res_sig.append(r[rr:rr + h, c:c + w].ravel())
        # local mean of the reference vs local std of the noise
        m = ndimage.uniform_filter(q.clean.astype(np.float64), 9)
        s = np.sqrt(np.maximum(ndimage.uniform_filter(r * r, 9) - ndimage.uniform_filter(r, 9) ** 2, 0))
        means.append(m[::9, ::9].ravel())
        stds.append(s[::9, ::9].ravel())
    res_bg, res_sig = np.concatenate(res_bg), np.concatenate(res_sig)
    means, stds = np.concatenate(means), np.concatenate(stds)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    bins = np.linspace(-0.6, 0.6, 81)
    ax[0].hist(res_bg, bins, density=True, alpha=0.6, label="background ROI", color="#c0392b")
    ax[0].hist(res_sig, bins, density=True, alpha=0.6, label="retina ROI", color="#27ae60")
    ax[0].set(title="Noise residual (noisy - averaged)", xlabel="intensity difference", ylabel="density")
    ax[0].legend()
    edges = np.linspace(0.05, 0.9, 18)
    idx = np.digitize(means, edges)
    cx = [means[idx == k].mean() for k in range(1, len(edges)) if (idx == k).sum() > 50]
    cy = [np.median(stds[idx == k]) for k in range(1, len(edges)) if (idx == k).sum() > 50]
    ax[1].scatter(means[::7], stds[::7], s=1, alpha=0.15, color="#555")
    ax[1].plot(cx, cy, "o-", color=DL_COLOR, label="binned median")
    ax[1].set(title="Noise std vs. local signal level", xlabel="local mean of reference",
              ylabel="local noise std", ylim=(0, 0.35))
    ax[1].legend()
    q = pairs[0]
    r0, c0, h, w = _retina_window(q.clean)
    ax[2].imshow((q.noisy - q.clean)[r0:r0 + h, c0:c0 + w], cmap="coolwarm", vmin=-0.4, vmax=0.4)
    ax[2].set_title("Residual noise pattern (crop) - grainy speckle")
    ax[2].axis("off")
    for a in ax[:2]:
        a.grid(alpha=0.3)
    _save(fig, path)


def fig_qualitative(outs, path):
    names = ["Noisy"] + _methods(outs) + [REF]
    n = len(names)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 2.5))
    ref = outs[REF]
    for a, nm in zip(ax.ravel(), names):
        im = outs[nm]
        a.imshow(im, cmap="gray", vmin=0, vmax=1)
        t = nm
        if nm != REF:
            t += f"\nPSNR {M.psnr(ref, im):.2f} dB | SSIM {M.ssim(ref, im):.3f} | EPI {M.epi(ref, im):.3f}"
        a.set_title(t, color=_color(nm) if nm in F.METHODS or nm.startswith("DnCNN") else "k")
        a.axis("off")
    for a in ax.ravel()[n:]:
        a.axis("off")
    fig.suptitle("Grey title = non edge-preserving baseline, blue = edge-preserving filter, "
                 "orange = learned (DnCNN)", y=1.0, fontsize=9)
    fig.tight_layout()
    _save(fig, path)


def fig_zoom(outs, path):
    names = ["Noisy"] + _methods(outs) + [REF]
    r0, c0, h, w = _retina_window(outs[REF])
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.0))
    for a, nm in zip(ax.ravel(), names):
        a.imshow(outs[nm][r0:r0 + h, c0:c0 + w], cmap="gray", vmin=0, vmax=1)
        a.set_title(nm, color=_color(nm))
        a.axis("off")
    for a in ax.ravel()[len(names):]:
        a.axis("off")
    fig.suptitle(f"Zoom on the retinal layers / fovea (rows {r0}-{r0 + h}, cols {c0}-{c0 + w})", y=1.0)
    fig.tight_layout()
    _save(fig, path)


def fig_profile(outs, path):
    ref = outs[REF]
    r0, _, h, _ = _retina_window(ref, h=150)
    col = ref.shape[1] // 2 + 60
    cs = slice(col - 2, col + 3)  # average 5 neighbouring A-scans
    rows = np.arange(r0, r0 + h)
    prof = lambda im: im[r0:r0 + h, cs].mean(1)
    base = [m for m in _methods(outs) if m in F.METHODS and not F.METHODS[m][2]]
    edge = [m for m in _methods(outs) if m not in base]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for a, group, title in [(ax[0], base, "Non edge-preserving baselines"),
                            (ax[1], edge, "Edge-preserving & learned methods")]:
        a.plot(rows, prof(outs["Noisy"]), color="#cccccc", lw=1, label="Noisy")
        a.plot(rows, prof(ref), color="k", lw=2.2, label="Averaged (reference)")
        for m in group:
            a.plot(rows, prof(outs[m]), lw=1.3, label=m)
        a.set(title=title, xlabel="depth (pixel row)")
        a.grid(alpha=0.3)
        a.legend(fontsize=7, ncol=2)
    ax[0].set_ylabel("intensity (mean of 5 A-scans)")
    fig.suptitle(f"A-scan intensity profile at column {col}: sharp steps = retinal layer boundaries", y=1.02)
    _save(fig, path)


def fig_edges(outs, path):
    names = ["Noisy"] + _methods(outs) + [REF]
    r0, c0, h, w = _retina_window(outs[REF], h=150, w=400)
    ref_e = canny(outs[REF][r0:r0 + h, c0:c0 + w], sigma=2.0, low_threshold=0.80,
                  high_threshold=0.90, use_quantiles=True)
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.0))
    for a, nm in zip(ax.ravel(), names):
        e = canny(outs[nm][r0:r0 + h, c0:c0 + w], sigma=2.0, low_threshold=0.80,
                  high_threshold=0.90, use_quantiles=True)
        # overlap with the reference edges (within 1 px) = "true" edges
        hit = (e & ndimage.binary_dilation(ref_e, iterations=1)).sum() / max(e.sum(), 1)
        rgb = np.zeros(e.shape + (3,))
        rgb[ref_e] = [0.35, 0.35, 0.35]
        rgb[e] = [1, 1, 1]
        a.imshow(rgb)
        a.set_title(f"{nm}\nedge precision {hit:.2f}", color=_color(nm))
        a.axis("off")
    for a in ax.ravel()[len(names):]:
        a.axis("off")
    fig.suptitle("Canny edge maps (white) over reference edges (grey). Precision = fraction of "
                 "detected edges lying on true layer boundaries.", y=1.01)
    fig.tight_layout()
    _save(fig, path)


def fig_bars(summ, path):
    s = summ.drop("Noisy input")
    names = list(s.index)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, c in zip(ax, ["PSNR", "SSIM", "EPI"]):
        a.bar(range(len(names)), s[c], yerr=s[c + "_std"], color=[_color(m) for m in names], capsize=2)
        a.axhline(summ.loc["Noisy input", c], color="r", ls="--", lw=1, label="noisy input")
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=55, ha="right")
        a.set_title(f"{c} (mean ± std over test images)")
        a.grid(axis="y", alpha=0.3)
        lo = min(summ.loc["Noisy input", c], (s[c] - s[c + "_std"]).min())
        a.set_ylim(lo * 0.9 if lo > 0 else lo, (s[c] + s[c + "_std"]).max() * 1.03)
        a.legend(fontsize=7)
    _save(fig, path)


def fig_tradeoff(pairs, summ, path):
    sigmas = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8]
    ps, es = [], []
    for s in sigmas:
        outs = [F.gaussian(q.noisy, s) for q in pairs]
        ps.append(np.mean([M.psnr(q.clean, o) for q, o in zip(pairs, outs)]))
        es.append(np.mean([M.epi(q.clean, o) for q, o in zip(pairs, outs)]))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    a2 = ax[0].twinx()
    ax[0].plot(sigmas, ps, "o-", color=EDGE_COLOR, label="PSNR")
    a2.plot(sigmas, es, "s-", color=DL_COLOR, label="EPI")
    ax[0].set(xlabel="Gaussian sigma (amount of smoothing)", ylabel="PSNR (dB)",
              title="Plain smoothing: more blur removes noise but destroys edges")
    a2.set_ylabel("EPI")
    ax[0].grid(alpha=0.3)
    ax[0].legend(loc="lower left")
    a2.legend(loc="lower right")
    s = summ.drop("Noisy input").sort_values("EPI")
    placed = []
    for m, r in s.iterrows():
        x, y = r["PSNR"], r["EPI"]
        ax[1].scatter(x, y, color=_color(m), s=45, zorder=3, edgecolor="k", lw=0.4)
        dy = 0
        while any(abs(y + dy - py) < 0.012 and abs(x - px) < 0.35 for px, py in placed):
            dy += 0.012
        placed.append((x, y + dy))
        ax[1].annotate(m, (x, y), xytext=(x + 0.04, y + dy), fontsize=7, va="center",
                       arrowprops=dict(arrowstyle="-", lw=0.4, color="#888") if dy else None)
    ax[1].set(xlabel="PSNR (dB)  -> less noise", ylabel="EPI  -> sharper edges",
              title="Noise removal vs edge preservation (mean over test set; top-right is best)")
    ax[1].grid(alpha=0.3)
    fig.subplots_adjust(wspace=0.38)
    _save(fig, path)


def fig_synthetic(ss, path):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    cmap = plt.get_cmap("tab10")
    names = [m for m in dict.fromkeys(ss["method"]) if m != "Noisy input"]
    for a, c in zip(ax, ["PSNR", "SSIM", "EPI"]):
        g = ss[ss.method == "Noisy input"]
        a.plot(g["looks"], g[c], "k--", marker="x", label="Noisy input")
        for i, m in enumerate(names):
            g = ss[ss.method == m]
            base = m in F.METHODS and not F.METHODS[m][2]
            a.plot(g["looks"], g[c], ":" if base else "-", marker="o", ms=3.5, lw=1.4,
                   color=cmap(i % 10), label=m + (" (baseline)" if base else ""))
        a.set(xlabel="number of looks L (higher = weaker speckle)", ylabel=c, title=f"{c} vs speckle strength")
        a.set_xscale("log", base=2)
        a.set_xticks(sorted(ss["looks"].unique()))
        a.set_xticklabels([str(v) for v in sorted(ss["looks"].unique())])
        a.grid(alpha=0.3)
    ax[2].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Synthetic multiplicative speckle added to the averaged images (dotted = non edge-preserving)", y=1.02)
    _save(fig, path)


def save_outputs(outs, folder):
    os.makedirs(folder, exist_ok=True)
    for k, v in outs.items():
        name = "".join(ch if ch.isalnum() else "_" for ch in k).strip("_")
        io.imsave(os.path.join(folder, f"{name}.png"), (np.clip(v, 0, 1) * 255).astype(np.uint8),
                  check_contrast=False)
    print("  saved denoised examples ->", os.path.relpath(folder))
