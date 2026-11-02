# Data: Duke SD-OCT denoising dataset

The data is **already included** in this folder, so nothing needs to be downloaded.

| Folder / file | Content |
|---|---|
| `duke_oct_test/01 … 18/noisy.tif` | 18 single, noisy retinal B-scans (Bioptigen SD-OCT, 450 × 900 px, 8-bit; image 09 is 401 × 900) |
| `duke_oct_test/01 … 18/average.tif` | The matching **registered and averaged** B-scan (many repeated scans averaged), used as the high-SNR ground truth |
| `train_patches/duke_train_patches_15x15.npz` | 4 000 noisy/clean 15 × 15 patch pairs taken from **10 separate training B-scans** (keys `noisy`, `clean`, uint8). Only used to train the DnCNN. |

How the data is split:

* **Development / validation:** images 01–04. The hyper-parameters of the classical filters are tuned on these. The DnCNN also uses 01–03 for extra patches and 04 to pick the best epoch.
* **Test:** images 05–18 (14 images). Every number in the report comes from these, and they are never used for tuning or training.

The images come from normal and AMD eyes in the A2A SD-OCT multi-centre clinical trial.

## Links

* Original dataset (Duke University, Prof. Sina Farsiu's lab):
  https://people.duke.edu/~sf59/Fang_BOE_2012.htm and https://people.duke.edu/~sf59/Fang_TMI_2013.htm
* Public GitHub mirror that this copy was taken from (`RESULTS/dt1_Bioptigen_SDOCT` and `data/`):
  https://github.com/ashkan-abbasi66/MIFCN
* To fetch the data again: `python scripts/download_data.py`

## Citation (required by the dataset authors)

> L. Fang, S. Li, Q. Nie, J. A. Izatt, C. A. Toth, S. Farsiu, "Sparsity based denoising of spectral domain
> optical coherence tomography images," *Biomedical Optics Express*, 3(5), 927–942, 2012.
> https://doi.org/10.1364/BOE.3.000927

The dataset may be used for academic research and teaching only.
