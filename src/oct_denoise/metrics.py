"""Image quality metrics for OCT denoising.

Full-reference (compare against the registered & averaged image):
    PSNR  - peak signal-to-noise ratio (dB), higher is better
    SSIM  - structural similarity index, higher is better (max 1)
    EPI   - edge preservation index (Sattar et al., 1997), higher is better (max 1)

No-reference (standard in the OCT literature; computed on fixed ROIs):
    SNR   - 10*log10( max(I)^2 / var(background) )                     (dB)
    CNR   - mean over signal ROIs of |mu_s - mu_b| / sqrt(var_s + var_b)
    ENL   - equivalent number of looks, mean over signal ROIs of mu_s^2 / var_s
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref, img):
    return float(peak_signal_noise_ratio(ref, img, data_range=1.0))


def ssim(ref, img):
    return float(structural_similarity(ref, img, data_range=1.0))


def epi(ref, img):
    """Edge Preservation Index.

    Correlation between the Laplacian (high-pass) responses of the reference
    and of the denoised image:
        EPI = sum((dS - mean dS)(dD - mean dD)) /
              sqrt(sum((dS - mean dS)^2) * sum((dD - mean dD)^2))
    1 = all edges of the reference are present in the denoised image.
    Both images get the same light Gaussian (sigma=1) before the Laplacian so
    that pixel-level residual noise of the averaged reference does not dominate
    its edges. Leftover noise AND blurred (over-smoothed) edges both lower EPI.
    """
    ds = ndimage.laplace(ndimage.gaussian_filter(ref.astype(np.float64), 1.0))
    dd = ndimage.laplace(ndimage.gaussian_filter(img.astype(np.float64), 1.0))
    ds -= ds.mean()
    dd -= dd.mean()
    return float(np.sum(ds * dd) / np.sqrt(np.sum(ds ** 2) * np.sum(dd ** 2) + 1e-12))


# --------------------------------------------------------------------------- #
# ROI selection (done ONCE per image on the clean reference, then the same
# ROIs are used for every method so the comparison is fair).
# --------------------------------------------------------------------------- #
def select_rois(clean, bg_size=(40, 120), sig_size=(8, 40), n_signal=3):
    """Pick a homogeneous background ROI and `n_signal` homogeneous bright ROIs.

    Background: the window with the lowest mean intensity (vitreous / deep tissue).
    Signal: bright windows (top 10 % mean) with the lowest relative std, i.e.
    homogeneous parts of retinal layers; non-overlapping.
    Returns lists of (row, col, h, w).
    """
    c = clean.astype(np.float64)
    H, W = c.shape

    def local_stats(h, w):
        m = ndimage.uniform_filter(c, (h, w))
        m2 = ndimage.uniform_filter(c * c, (h, w))
        return m, np.sqrt(np.maximum(m2 - m * m, 0))

    margin = 10
    # background
    bh, bw = bg_size
    m, _ = local_stats(bh, bw)
    valid = np.full_like(m, np.inf)
    valid[bh // 2 + margin:H - bh // 2 - margin, bw // 2 + margin:W - bw // 2 - margin] = \
        m[bh // 2 + margin:H - bh // 2 - margin, bw // 2 + margin:W - bw // 2 - margin]
    r, cc = np.unravel_index(np.argmin(valid), valid.shape)
    bg = [(r - bh // 2, cc - bw // 2, bh, bw)]

    # signal
    sh, sw = sig_size
    m, s = local_stats(sh, sw)
    inner = np.zeros_like(m, bool)
    inner[sh // 2 + margin:H - sh // 2 - margin, sw // 2 + margin:W - sw // 2 - margin] = True
    bright = inner & (m >= np.quantile(m[inner], 0.90))
    score = np.where(bright, s / (m + 1e-6), np.inf)
    taken = np.zeros_like(m, bool)
    sig = []
    for _ in range(n_signal):
        score_t = np.where(taken, np.inf, score)
        if not np.isfinite(score_t).any():
            break
        r, cc = np.unravel_index(np.argmin(score_t), score_t.shape)
        sig.append((r - sh // 2, cc - sw // 2, sh, sw))
        taken[max(0, r - sh):r + sh, max(0, cc - sw):cc + sw] = True
    return bg, sig


def _crop(img, roi):
    r, c, h, w = roi
    return img[r:r + h, c:c + w].astype(np.float64)


def no_reference(img, bg_rois, sig_rois):
    b = np.concatenate([_crop(img, x).ravel() for x in bg_rois])
    mu_b, var_b = b.mean(), b.var() + 1e-12
    snr = 10 * np.log10(img.max() ** 2 / var_b)
    cnr, enl = [], []
    for roi in sig_rois:
        s = _crop(img, roi)
        mu_s, var_s = s.mean(), s.var() + 1e-12
        cnr.append(abs(mu_s - mu_b) / np.sqrt(var_s + var_b))
        enl.append(mu_s ** 2 / var_s)
    return {"SNR": float(snr), "CNR": float(np.mean(cnr)), "ENL": float(np.mean(enl))}


def all_metrics(clean, img, bg_rois, sig_rois):
    out = {"PSNR": psnr(clean, img), "SSIM": ssim(clean, img), "EPI": epi(clean, img)}
    out.update(no_reference(img, bg_rois, sig_rois))
    return out
