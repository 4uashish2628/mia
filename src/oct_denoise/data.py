"""Data loading utilities for the Duke SD-OCT denoising dataset.

Dataset: Fang et al., "Sparsity based denoising of spectral domain optical
coherence tomography images", Biomedical Optics Express 3(5), 2012.

Layout expected (already included in this project under ``data/``)::

    data/duke_oct_test/01/noisy.tif     # single noisy B-scan (low SNR)
    data/duke_oct_test/01/average.tif   # registered + averaged B-scan (high SNR "ground truth")
    ...
    data/duke_oct_test/18/...
    data/train_patches/duke_train_patches_15x15.npz   # 4000 noisy/clean 15x15 patch pairs
                                                       # from 10 *separate* training images
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from skimage import io

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEST_DIR = os.path.join(DATA_DIR, "duke_oct_test")
PATCH_FILE = os.path.join(DATA_DIR, "train_patches", "duke_train_patches_15x15.npz")

# Images 01-04 are used ONLY for hyper-parameter tuning (validation),
# images 05-18 are the held-out TEST set on which all numbers are reported.
VAL_IDS = ["01", "02", "03", "04"]
TEST_IDS = [f"{i:02d}" for i in range(5, 19)]


@dataclass
class OCTPair:
    image_id: str
    noisy: np.ndarray   # float32 in [0, 1]
    clean: np.ndarray   # float32 in [0, 1]  (registered & averaged reference)


def _read(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32) / 255.0


def load_pair(image_id: str, root: str = TEST_DIR) -> OCTPair:
    folder = os.path.join(root, image_id)
    return OCTPair(image_id, _read(os.path.join(folder, "noisy.tif")),
                   _read(os.path.join(folder, "average.tif")))


def load_split(ids) -> list[OCTPair]:
    return [load_pair(i) for i in ids]


def load_train_patches(path: str = PATCH_FILE):
    """Returns (noisy, clean) arrays of shape (N, 15, 15), float32 in [0, 1]."""
    d = np.load(path)
    return d["noisy"].astype(np.float32) / 255.0, d["clean"].astype(np.float32) / 255.0
