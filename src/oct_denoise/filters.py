"""Classical denoising filters for retinal OCT B-scans.

Every function takes a float32 image in [0, 1] and returns a float32 image in [0, 1].

Non edge-preserving baselines : Gaussian, Median, Wiener
Edge-preserving methods       : Bilateral, Guided filter, Total Variation (TV),
                                Perona-Malik anisotropic diffusion,
                                SRAD (speckle reducing anisotropic diffusion),
                                Non-Local Means (NLM)
(The learned DnCNN model lives in ``dncnn.py``.)
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage, signal
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Baselines (NOT edge preserving)
# --------------------------------------------------------------------------- #
def gaussian(img, sigma=1.5):
    """Isotropic Gaussian low-pass filter: smooths noise AND edges equally."""
    return _clip(ndimage.gaussian_filter(img, sigma=sigma))


def median(img, size=5):
    """Median filter: robust to impulsive (salt & pepper like) speckle outliers."""
    return _clip(ndimage.median_filter(img, size=size))


def wiener(img, size=5):
    """Adaptive (local-statistics) Wiener filter."""
    return _clip(signal.wiener(img.astype(np.float64), mysize=size))


# --------------------------------------------------------------------------- #
# Edge-preserving filters
# --------------------------------------------------------------------------- #
def bilateral(img, sigma_color=0.1, sigma_space=3.0):
    """Bilateral filter (Tomasi & Manduchi, 1998).

    Weight = spatial Gaussian x range (intensity) Gaussian, so pixels across a
    strong edge (large intensity difference) barely contribute -> edges survive.
    """
    d = int(2 * round(2 * sigma_space) + 1)
    return _clip(cv2.bilateralFilter(img.astype(np.float32), d, sigma_color, sigma_space))


def guided(img, radius=4, eps=0.01, guide=None):
    """Guided filter (He et al., 2013), self-guided by default.

    Local linear model q = a*I + b inside each window. In flat regions
    var(I) << eps -> a ~ 0 (strong smoothing); at edges var(I) >> eps -> a ~ 1
    (edge kept).
    """
    I = img if guide is None else guide
    p = img
    k = 2 * radius + 1
    box = lambda x: cv2.boxFilter(x, -1, (k, k), borderType=cv2.BORDER_REFLECT)
    mean_I, mean_p = box(I), box(p)
    cov_Ip = box(I * p) - mean_I * mean_p
    var_I = box(I * I) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return _clip(box(a) * I + box(b))


def total_variation(img, weight=0.08):
    """Total-variation (ROF) denoising, Chambolle's algorithm.

    Minimises ||u - f||^2 + weight * TV(u); TV penalises small oscillations
    (noise) but allows sharp jumps (edges) -> piecewise-smooth result.
    """
    return _clip(denoise_tv_chambolle(img, weight=weight))


def perona_malik(img, n_iter=20, kappa=0.1, gamma=0.2, option=1):
    """Perona-Malik anisotropic diffusion (1990).

    dI/dt = div( c(|grad I|) grad I ), with an edge-stopping function
        option 1: c = exp(-(|g|/kappa)^2)       (favours high-contrast edges)
        option 2: c = 1 / (1 + (|g|/kappa)^2)   (favours wide regions)
    Diffusion is strong where the gradient is small (noise) and ~0 across
    edges (large gradient). gamma <= 0.25 for numerical stability.
    """
    u = img.astype(np.float32).copy()
    for _ in range(int(n_iter)):
        # finite differences to the 4 neighbours (replicated border)
        p = np.pad(u, 1, mode="edge")
        dN = p[:-2, 1:-1] - u
        dS = p[2:, 1:-1] - u
        dE = p[1:-1, 2:] - u
        dW = p[1:-1, :-2] - u
        if option == 1:
            c = lambda d: np.exp(-(d / kappa) ** 2)
        else:
            c = lambda d: 1.0 / (1.0 + (d / kappa) ** 2)
        u = u + gamma * (c(dN) * dN + c(dS) * dS + c(dE) * dE + c(dW) * dW)
    return _clip(u)


def srad(img, n_iter=30, dt=0.1, q0_decay=1.0 / 6.0, eps=1e-3):
    """Speckle Reducing Anisotropic Diffusion (Yu & Acton, IEEE TIP 2002).

    Designed for multiplicative speckle (ultrasound / OCT). The diffusion
    coefficient uses the *instantaneous coefficient of variation* q(x,y,t),
    which acts as an edge detector for speckled images:
        c(q) = 1 / (1 + (q^2 - q0^2) / (q0^2 (1 + q0^2)))
    q0(t) = q0(0) * exp(-q0_decay * t) is the speckle scale.
    """
    u = img.astype(np.float64) + eps
    # initial speckle scale estimated from the whole image (coefficient of variation)
    q0_init = np.sqrt(np.var(u)) / np.mean(u)
    for it in range(int(n_iter)):
        t = it * dt
        q0 = q0_init * np.exp(-q0_decay * t)
        p = np.pad(u, 1, mode="edge")
        dN = p[:-2, 1:-1] - u
        dS = p[2:, 1:-1] - u
        dE = p[1:-1, 2:] - u
        dW = p[1:-1, :-2] - u
        g2 = (dN ** 2 + dS ** 2 + dE ** 2 + dW ** 2) / (u ** 2)
        lap = (dN + dS + dE + dW) / u
        num = 0.5 * g2 - (1.0 / 16.0) * lap ** 2
        den = (1.0 + 0.25 * lap) ** 2
        q2 = np.maximum(num / (den + 1e-12), 0.0)
        c = 1.0 / (1.0 + (q2 - q0 ** 2) / (q0 ** 2 * (1.0 + q0 ** 2) + 1e-12))
        c = np.clip(c, 0.0, 1.0)
        cp = np.pad(c, 1, mode="edge")
        cS, cE = cp[2:, 1:-1], cp[1:-1, 2:]
        div = cS * dS + c * dN + cE * dE + c * dW
        u = u + (dt / 4.0) * div
    return _clip(u - eps)


def estimate_noise_sigma(img):
    """Robust noise std estimate (Donoho MAD on the finest diagonal Haar band)."""
    x = img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2].astype(np.float64)
    hh = (x[0::2, 0::2] - x[0::2, 1::2] - x[1::2, 0::2] + x[1::2, 1::2]) / 2.0
    return float(np.median(np.abs(hh)) / 0.6745)


def nlm(img, h_factor=0.8, patch_size=5, patch_distance=6):
    """Non-Local Means (Buades et al., 2005).

    Each pixel = weighted average of pixels whose *surrounding patch* looks
    similar, searched over a large window. Repetitive layered OCT structure is
    a good fit, and edges are kept because patches across an edge are dissimilar.
    """
    sigma = estimate_noise_sigma(img)
    return _clip(denoise_nl_means(img, h=h_factor * sigma, sigma=sigma, fast_mode=True,
                                  patch_size=patch_size, patch_distance=patch_distance))


# --------------------------------------------------------------------------- #
# Registry: name -> (function, hyper-parameter grid, edge preserving?)
# Grids are searched on the validation images (01-04) only.
# --------------------------------------------------------------------------- #
METHODS = {
    "Gaussian": (gaussian, {"sigma": [0.75, 1.0, 1.5, 2.0, 2.5, 3.0]}, False),
    "Median": (median, {"size": [3, 5, 7, 9]}, False),
    "Wiener": (wiener, {"size": [3, 5, 7, 9]}, False),
    "Bilateral": (bilateral, {"sigma_color": [0.2, 0.3, 0.5, 0.8],
                              "sigma_space": [2.0, 3.0, 5.0]}, True),
    "Guided": (guided, {"radius": [4, 6, 8, 12], "eps": [0.02, 0.04, 0.08, 0.16]}, True),
    "TV (Chambolle)": (total_variation, {"weight": [0.05, 0.08, 0.12, 0.16, 0.2]}, True),
    "Perona-Malik AD": (perona_malik, {"n_iter": [15, 30, 50], "kappa": [0.1, 0.2, 0.3],
                                       "option": [1, 2]}, True),
    "SRAD": (srad, {"n_iter": [30, 60, 100], "dt": [0.1, 0.2],
                    "q0_decay": [1.0 / 6.0, 0.05]}, True),
    "NLM": (nlm, {"h_factor": [0.6, 0.8, 1.0, 1.2], "patch_size": [5, 7],
                  "patch_distance": [6, 10]}, True),
}
