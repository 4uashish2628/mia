"""Re-create the data/ folder from the public sources (the data is already included
in this project; use this only if you want to fetch it again).

Source of the 18 test pairs and the training patches: the public GitHub mirror of the
Duke SD-OCT denoising dataset that ships with the MIFCN project
    https://github.com/ashkan-abbasi66/MIFCN   (RESULTS/dt1_Bioptigen_SDOCT, data/)
Original dataset page (Duke University, Fang et al. 2012/2013):
    https://people.duke.edu/~sf59/Fang_BOE_2012.htm
    https://people.duke.edu/~sf59/Fang_TMI_2013.htm

Usage:  python scripts/download_data.py      (needs git)
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = "https://github.com/ashkan-abbasi66/MIFCN"


def main():
    tmp = tempfile.mkdtemp()
    print("cloning", REPO, "...")
    subprocess.check_call(["git", "clone", "--depth", "1", "-q", REPO, os.path.join(tmp, "MIFCN")])
    src = os.path.join(tmp, "MIFCN", "RESULTS", "dt1_Bioptigen_SDOCT")
    dst = os.path.join(ROOT, "data", "duke_oct_test")
    for i in range(1, 19):
        d = os.path.join(dst, f"{i:02d}")
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(src, f"{i:02d}", "test.tif"), os.path.join(d, "noisy.tif"))
        shutil.copy(os.path.join(src, f"{i:02d}", "average.tif"), os.path.join(d, "average.tif"))
    a = np.load(os.path.join(tmp, "MIFCN", "data", "train15_inputs.npy"))
    b = np.load(os.path.join(tmp, "MIFCN", "data", "train15_labels.npy"))
    out = os.path.join(ROOT, "data", "train_patches")
    os.makedirs(out, exist_ok=True)
    # channel 0 = the reference patch (channels 1-4 are non-local similar patches)
    np.savez_compressed(os.path.join(out, "duke_train_patches_15x15.npz"),
                        noisy=np.round(a[:, 0] * 255).astype(np.uint8),
                        clean=np.round(b[:, 0] * 255).astype(np.uint8))
    shutil.rmtree(tmp, ignore_errors=True)
    print("data ready in", os.path.join(ROOT, "data"))


if __name__ == "__main__":
    sys.exit(main())
