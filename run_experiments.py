"""Full experimental pipeline for Edge-Preserving Denoising of Retinal OCT scans.

Steps
  1. Hyper-parameter tuning of every classical filter on VALIDATION images 01-04
     (grid search, objective = mean PSNR).
  2. Evaluation of all methods (+ trained DnCNN) on the held-out TEST images 05-18
     with PSNR, SSIM, EPI (full-reference) and SNR, CNR, ENL (no-reference).
  3. Wilcoxon signed-rank significance tests.
  4. Controlled experiment: synthetic multiplicative speckle at 3 noise levels.
  5. All figures for the report.

Usage:  python run_experiments.py            (full run, ~15-25 min on a laptop CPU)
        python run_experiments.py --quick    (small grids / 3 test images, ~2 min)
        python run_experiments.py --no-synthetic
        python run_experiments.py --figures-only   (re-draw figures from saved results)
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from oct_denoise import filters as F  # noqa: E402
from oct_denoise import metrics as M  # noqa: E402
from oct_denoise import visualize as V  # noqa: E402
from oct_denoise.data import PROJECT_ROOT, TEST_IDS, VAL_IDS, load_split  # noqa: E402
from oct_denoise.dncnn import DnCNN  # noqa: E402
from oct_denoise.speckle import add_speckle  # noqa: E402

RES = os.path.join(PROJECT_ROOT, "results")
FIG = os.path.join(RES, "figures")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "dncnn_numpy.npz")
METRICS = ["PSNR", "SSIM", "EPI", "SNR", "CNR", "ENL"]


def grid(param_grid):
    keys = list(param_grid)
    for vals in itertools.product(*[param_grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def tune(method, pairs, quick=False):
    """Grid search on validation pairs; returns best params and the full table."""
    fn, pgrid, _ = F.METHODS[method]
    if quick:  # 2 values per parameter
        pgrid = {k: [v[0], v[len(v) // 2]] if len(v) > 1 else v for k, v in pgrid.items()}
    rows = []
    for p in grid(pgrid):
        score = float(np.mean([M.psnr(q.clean, fn(q.noisy, **p)) for q in pairs]))
        rows.append({**p, "val_PSNR": score})
    best = max(rows, key=lambda r: r["val_PSNR"])
    params = {k: v for k, v in best.items() if k != "val_PSNR"}
    params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
    return params, best["val_PSNR"], pd.DataFrame(rows)


def evaluate(pairs, methods, best_params, dncnn, tag=""):
    """Run every method on every pair. Returns per-image DataFrame + outputs dict."""
    rows, outputs = [], {}
    for q in pairs:
        bg, sig = M.select_rois(q.clean)
        outputs[q.image_id] = {"Noisy": q.noisy, "Averaged (reference)": q.clean}
        rows.append({"image": q.image_id, "method": "Noisy input", "time_s": 0.0,
                     **M.all_metrics(q.clean, q.noisy, bg, sig)})
        for m in methods:
            t = time.perf_counter()
            if m == "DnCNN (ours, NumPy)":
                out = dncnn.denoise(q.noisy)
            else:
                out = F.METHODS[m][0](q.noisy, **best_params[m])
            dt = time.perf_counter() - t
            outputs[q.image_id][m] = out
            rows.append({"image": q.image_id, "method": m, "time_s": dt,
                         **M.all_metrics(q.clean, out, bg, sig)})
        print(f"  {tag} image {q.image_id} done", flush=True)
    return pd.DataFrame(rows), outputs


def summarise(df):
    order = list(dict.fromkeys(df["method"]))
    g = df.groupby("method", sort=False)
    mean, std = g[METRICS + ["time_s"]].mean(), g[METRICS + ["time_s"]].std()
    summ = mean.copy()
    for c in METRICS + ["time_s"]:
        summ[c + "_std"] = std[c]
    return summ.loc[order]


def to_markdown(summ, edge_flags):
    lines = ["| Method | Edge-preserving | PSNR (dB) ↑ | SSIM ↑ | EPI ↑ | SNR (dB) ↑ | CNR ↑ | ENL ↑ | Time (s) |",
             "|---|:---:|---|---|---|---|---|---|---|"]
    best = {c: summ[c].drop("Noisy input", errors="ignore").max() for c in METRICS}
    for m, r in summ.iterrows():
        cells = []
        for c in METRICS:
            fmt = "{:.3f}" if c in ("SSIM", "EPI") else ("{:.1f}" if c == "ENL" else "{:.2f}")
            s = fmt.format(r[c]) + " ± " + fmt.format(r[c + "_std"])
            if m != "Noisy input" and np.isclose(r[c], best[c]):
                s = f"**{s}**"
            cells.append(s)
        ep = edge_flags.get(m, "–")
        lines.append(f"| {m} | {ep} | " + " | ".join(cells) + f" | {r['time_s']:.2f} |")
    return "\n".join(lines)


def significance(df, reference):
    rows = []
    ref = df[df.method == reference].set_index("image")
    for m in df.method.unique():
        if m in (reference, "Noisy input"):
            continue
        other = df[df.method == m].set_index("image").loc[ref.index]
        row = {"compared_with": m}
        for c in ["PSNR", "SSIM", "EPI"]:
            d = ref[c] - other[c]
            p = stats.wilcoxon(ref[c], other[c]).pvalue if np.any(d != 0) else 1.0
            row[f"{c}_diff"] = float(d.mean())
            row[f"{c}_p"] = float(p)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--figures-only", action="store_true",
                    help="re-draw figures from the saved CSV/JSON results (no tuning / evaluation)")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    if args.figures_only:
        return figures_only()
    t_all = time.time()

    val = load_split(VAL_IDS)
    test = load_split(TEST_IDS[:3] if args.quick else TEST_IDS)
    methods = list(F.METHODS)
    edge_flags = {m: ("yes" if F.METHODS[m][2] else "no") for m in methods}

    dncnn = DnCNN.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    if dncnn is None:
        print("!! models/dncnn_numpy.npz not found - run train_dncnn.py first. Skipping DnCNN.")
    else:
        methods.append("DnCNN (ours, NumPy)")
        edge_flags["DnCNN (ours, NumPy)"] = "learned"

    # ---------------- 1. tuning on validation ---------------------------------
    print("== 1. Hyper-parameter tuning on validation images", VAL_IDS)
    best_params, tune_tables = {}, []
    for m in F.METHODS:
        p, s, tab = tune(m, val, args.quick)
        best_params[m] = p
        tab.insert(0, "method", m)
        tune_tables.append(tab)
        print(f"  {m:18s} best {p}  val PSNR {s:.2f}")
    pd.concat(tune_tables).to_csv(os.path.join(RES, "tuning_grid_search.csv"), index=False)
    with open(os.path.join(RES, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    # ---------------- 2. evaluation on test -----------------------------------
    print("== 2. Evaluation on test images", [q.image_id for q in test])
    df, outputs = evaluate(test, methods, best_params, dncnn, "real")
    df.to_csv(os.path.join(RES, "test_metrics_per_image.csv"), index=False)
    summ = summarise(df)
    summ.to_csv(os.path.join(RES, "test_metrics_summary.csv"))
    md = to_markdown(summ, edge_flags)
    with open(os.path.join(RES, "test_metrics_summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)

    # ---------------- 3. significance ------------------------------------------
    ranked = summ.drop("Noisy input")["PSNR"].sort_values(ascending=False)
    winner = ranked.index[0]
    sig = significance(df, winner)
    sig.to_csv(os.path.join(RES, "wilcoxon_vs_best.csv"), index=False)
    print(f"== 3. Wilcoxon signed-rank tests vs best method ({winner})")
    print(sig.round(4).to_string(index=False))

    # ---------------- 4. synthetic speckle --------------------------------------
    synth_rows = []
    if not args.no_synthetic:
        print("== 4. Synthetic speckle experiment")
        for looks in ([4] if args.quick else [2, 4, 8]):
            rng = np.random.default_rng(looks)
            sval = [type(q)(q.image_id, add_speckle(q.clean, looks, rng), q.clean) for q in val]
            stest = [type(q)(q.image_id, add_speckle(q.clean, looks, rng), q.clean) for q in test]
            bp = {m: tune(m, sval, True if args.quick else False)[0] for m in F.METHODS}
            d, _ = evaluate(stest, methods, bp, dncnn, f"L={looks}")
            d.insert(0, "looks", looks)
            synth_rows.append(d)
        synth = pd.concat(synth_rows)
        synth.to_csv(os.path.join(RES, "synthetic_speckle_per_image.csv"), index=False)
        ss = synth.groupby(["looks", "method"], sort=False)[["PSNR", "SSIM", "EPI"]].mean().reset_index()
        ss.to_csv(os.path.join(RES, "synthetic_speckle_summary.csv"), index=False)
        print(ss.pivot(index="method", columns="looks", values="PSNR").round(2).to_string())

    # ---------------- 5. figures ----------------------------------------------
    print("== 5. Figures")
    show = test[0].image_id if args.quick else "11"
    q = next(p for p in test if p.image_id == show) if any(p.image_id == show for p in test) else test[0]
    V.fig_dataset(q, os.path.join(FIG, "01_dataset_example_rois.png"))
    V.fig_noise_analysis(test, os.path.join(FIG, "02_noise_analysis.png"))
    V.fig_qualitative(outputs[q.image_id], os.path.join(FIG, "03_qualitative_all_methods.png"))
    V.fig_zoom(outputs[q.image_id], os.path.join(FIG, "04_zoomed_layers.png"))
    V.fig_profile(outputs[q.image_id], os.path.join(FIG, "05_ascan_edge_profile.png"))
    V.fig_edges(outputs[q.image_id], os.path.join(FIG, "06_edge_maps.png"))
    V.fig_bars(summ, os.path.join(FIG, "07_metric_bars.png"))
    V.fig_tradeoff(test[:4], summ, os.path.join(FIG, "08_noise_vs_edge_tradeoff.png"))
    if synth_rows:
        V.fig_synthetic(ss, os.path.join(FIG, "09_synthetic_speckle.png"))
    # save denoised outputs of the showcase image
    V.save_outputs(outputs[q.image_id], os.path.join(RES, "denoised_examples", f"image_{q.image_id}"))
    print(f"done in {(time.time() - t_all) / 60:.1f} min")


def figures_only():
    summ = pd.read_csv(os.path.join(RES, "test_metrics_summary.csv"), index_col=0)
    with open(os.path.join(RES, "best_params.json")) as f:
        best_params = json.load(f)
    test = load_split(TEST_IDS)
    methods = list(F.METHODS)
    dncnn = DnCNN.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    if dncnn is not None:
        methods.append("DnCNN (ours, NumPy)")
    q = next(p for p in test if p.image_id == "11")
    _, outputs = evaluate([q], methods, best_params, dncnn, "figures")
    V.fig_dataset(q, os.path.join(FIG, "01_dataset_example_rois.png"))
    V.fig_noise_analysis(test, os.path.join(FIG, "02_noise_analysis.png"))
    V.fig_qualitative(outputs["11"], os.path.join(FIG, "03_qualitative_all_methods.png"))
    V.fig_zoom(outputs["11"], os.path.join(FIG, "04_zoomed_layers.png"))
    V.fig_profile(outputs["11"], os.path.join(FIG, "05_ascan_edge_profile.png"))
    V.fig_edges(outputs["11"], os.path.join(FIG, "06_edge_maps.png"))
    V.fig_bars(summ, os.path.join(FIG, "07_metric_bars.png"))
    V.fig_tradeoff(test[:4], summ, os.path.join(FIG, "08_noise_vs_edge_tradeoff.png"))
    sp = os.path.join(RES, "synthetic_speckle_summary.csv")
    if os.path.exists(sp):
        V.fig_synthetic(pd.read_csv(sp), os.path.join(FIG, "09_synthetic_speckle.png"))


if __name__ == "__main__":
    main()
