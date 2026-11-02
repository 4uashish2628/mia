"""A small DnCNN-style residual denoising CNN implemented from scratch in NumPy.

Why NumPy? It keeps the project dependency-light (runs on any laptop, no GPU,
no PyTorch/TensorFlow install) and every step - convolution, backprop, Adam -
is visible and explainable.

Architecture (residual learning, Zhang et al., IEEE TIP 2017):

    x (noisy) -> Conv3x3(1->F) + ReLU
              -> [Conv3x3(F->F) + ReLU] x (D-2)
              -> Conv3x3(F->1)            = predicted noise n_hat
    output    =  x - n_hat

Loss: mean squared error between output and the clean (averaged) patch.
Optimiser: Adam.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# --------------------------------------------------------------------------- #
# 3x3 'same' convolution with im2col (NCHW layout, reflect padding)
# --------------------------------------------------------------------------- #
def _im2col(x):
    """x: (N, C, H, W) -> cols: (N*H*W, C*9) using reflect padding of 1.

    Reflect (mirror) padding avoids the dark-border artefacts that zero padding
    causes, which matters because the training patches are only 15x15.
    """
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect")
    win = sliding_window_view(xp, (3, 3), axis=(2, 3))  # (N, C, H, W, 3, 3)
    return win.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * W, C * 9)


def _col2im(dcols, shape):
    """Adjoint of _im2col: dcols (N*H*W, C*9) -> dx (N, C, H, W)."""
    N, C, H, W = shape
    d = dcols.reshape(N, H, W, C, 3, 3)
    dxp = np.zeros((N, C, H + 2, W + 2), dtype=dcols.dtype)
    for i in range(3):
        for j in range(3):
            dxp[:, :, i:i + H, j:j + W] += d[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    # fold the gradient of the mirrored border back onto the pixels it copied
    dxp[:, :, :, 2] += dxp[:, :, :, 0]
    dxp[:, :, :, -3] += dxp[:, :, :, -1]
    dxp[:, :, 2, :] += dxp[:, :, 0, :]
    dxp[:, :, -3, :] += dxp[:, :, -1, :]
    return dxp[:, :, 1:-1, 1:-1]


class Conv3x3:
    def __init__(self, cin, cout, rng):
        # He initialisation
        self.W = (rng.standard_normal((cin * 9, cout)) * np.sqrt(2.0 / (cin * 9))).astype(np.float32)
        self.b = np.zeros(cout, np.float32)

    def forward(self, x):
        N, C, H, W = x.shape
        self.shape = x.shape
        self.cols = _im2col(x)
        out = self.cols @ self.W + self.b
        return out.reshape(N, H, W, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        N, F, H, W = dout.shape
        d = dout.transpose(0, 2, 3, 1).reshape(-1, F)
        self.dW = self.cols.T @ d
        self.db = d.sum(0)
        return _col2im(d @ self.W.T, self.shape)


class DnCNN:
    def __init__(self, depth=6, features=32, seed=0):
        rng = np.random.default_rng(seed)
        chans = [1] + [features] * (depth - 1) + [1]
        self.layers = [Conv3x3(chans[i], chans[i + 1], rng) for i in range(depth)]
        self.depth, self.features = depth, features

    # ---- forward / backward ------------------------------------------------ #
    def forward(self, x):
        self.masks = []
        h = x
        for k, layer in enumerate(self.layers):
            h = layer.forward(h)
            if k < len(self.layers) - 1:
                m = h > 0
                self.masks.append(m)
                h = h * m
        self.noise = h
        return x - h  # residual learning

    def backward(self, dout):
        g = -dout  # d(out)/d(noise) = -1
        for k in range(len(self.layers) - 1, -1, -1):
            if k < len(self.layers) - 1:
                g = g * self.masks[k]
            g = self.layers[k].backward(g)

    # ---- parameters -------------------------------------------------------- #
    def params(self):
        for L in self.layers:
            yield L, "W"
            yield L, "b"

    def state_dict(self):
        d = {"depth": np.array(self.depth), "features": np.array(self.features)}
        for i, L in enumerate(self.layers):
            d[f"W{i}"], d[f"b{i}"] = L.W, L.b
        return d

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(int(d["depth"]), int(d["features"]))
        for i, L in enumerate(m.layers):
            L.W, L.b = d[f"W{i}"].astype(np.float32), d[f"b{i}"].astype(np.float32)
        return m

    def save(self, path):
        np.savez_compressed(path, **self.state_dict())

    # ---- inference on a full B-scan (processed in overlapping strips) ------ #
    def denoise(self, img, strip=64):
        H, W = img.shape
        pad = self.depth + 1  # receptive-field margin
        out = np.zeros_like(img, dtype=np.float32)
        for r0 in range(0, H, strip):
            a, b = max(0, r0 - pad), min(H, r0 + strip + pad)
            y = self.forward(img[None, None, a:b].astype(np.float32))[0, 0]
            out[r0:min(H, r0 + strip)] = y[r0 - a:r0 - a + min(strip, H - r0)]
        self.masks, self.noise = None, None
        for L in self.layers:
            L.cols = None
        return np.clip(out, 0, 1)


class Adam:
    def __init__(self, model, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.model, self.lr, self.b1, self.b2, self.eps, self.t = model, lr, b1, b2, eps, 0
        self.m = {id(getattr(L, n)): np.zeros_like(getattr(L, n)) for L, n in model.params()}
        self.v = {k: np.zeros_like(v) for k, v in self.m.items()}
        self.keys = [(L, n, id(getattr(L, n))) for L, n in model.params()]

    def step(self):
        self.t += 1
        for L, n, k in self.keys:
            g = getattr(L, "d" + n)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            getattr(L, n)[...] -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(np.float32)


def augment(x, y, rng):
    """Random flips / 90-degree rotations (applied identically to x and y)."""
    k = rng.integers(4)
    x, y = np.rot90(x, k, axes=(2, 3)), np.rot90(y, k, axes=(2, 3))
    if rng.random() < 0.5:
        x, y = x[..., ::-1], y[..., ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)
