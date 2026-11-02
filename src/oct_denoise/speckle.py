"""Synthetic OCT speckle model (for the controlled experiment).

Fully developed speckle is *multiplicative* in the linear intensity domain:
    I_noisy = I_clean * n,    n ~ Gamma(shape=L, scale=1/L)   (mean 1, var 1/L)
where L = number of looks (smaller L = stronger speckle).

OCT B-scans are displayed log-compressed, so we
  1) map the [0,1] display image back to linear intensity with a dynamic range D (dB),
  2) multiply by Gamma speckle,
  3) log-compress again.
In the displayed (log) image the speckle becomes additive and signal-independent,
which is exactly what the Duke images look like.
"""
import numpy as np


def add_speckle(clean, looks=4, rng=None, dynamic_range_db=40.0):
    rng = np.random.default_rng() if rng is None else rng
    lin = 10.0 ** ((clean.astype(np.float64) - 1.0) * dynamic_range_db / 10.0)
    n = rng.gamma(shape=looks, scale=1.0 / looks, size=clean.shape)
    noisy = 1.0 + 10.0 * np.log10(lin * n + 1e-12) / dynamic_range_db
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)
