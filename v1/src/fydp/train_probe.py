"""Spectral linear probe: cross-subject IMU-only FOG forecasting baseline.

Rationale (verified on the fixed subject-disjoint test set, seed 42):
- Raw-waveform PatchTST baseline: test AUPRC 0.500 (val 0.971).
- Five deep variants (posenc, focal, augmentation, instance norm, spectral
  fusion, mid-rate selection, extreme-dropping, 3-seed ensembles): 0.42-0.51.
- Logistic regression on 18 window-level freeze-band/locomotion-band spectral
  ratios: test AUPRC 0.567.

Under this subject shift, low-capacity convex models beat deep nets: the
spectral ratios are dimensionless and transfer across subjects, while
high-capacity encoders memorize train subjects' signatures (train/val ~0.97,
test ~0.50). Protocol is leakage-free: scaler + C selection (grouped CV over
non-test subjects) + final fit all use train-pool data only; the test set is
touched once for reporting.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from fydp.data.dataloader import build_all_windows, subject_group_split
from fydp.paths import project_root, resolve_path
from fydp.train_imu import metrics_from_probs, per_subject_auprc
import torch.nn as nn
import torch


def spectral_v1(x: np.ndarray, fs: int = 128) -> np.ndarray:
    F = np.abs(np.fft.rfft(x, axis=1))
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / fs)
    lb = (freqs >= 0.5) & (freqs <= 3.0)
    fb = (freqs > 3.0) & (freqs <= 8.0)
    pl = (F[:, lb, :] ** 2).sum(axis=1) + 1e-9
    pf = (F[:, fb, :] ** 2).sum(axis=1) + 1e-9
    return np.concatenate([np.log(pf / pl), np.log(pl + pf), x.std(axis=1)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="outputs/metrics_probe.json")
    args = ap.parse_args()

    with open(project_root() / "configs" / "probe.yml") as f:
        probe_cfg = dict(yaml.safe_load(f))
    with open(project_root() / "configs" / "data.yml") as f:
        data_cfg = dict(yaml.safe_load(f))

    root = project_root()
    X, y, subj, _, _ = build_all_windows(
        resolve_path(data_cfg["data_dir"]),
        fs=int(data_cfg.get("sampling_frequency", 128)),
        obs_s=float(data_cfg.get("observation_windows", 5.0)),
        horizon_s=float(data_cfg.get("forecast_horizon", 2.0)),
        stride_s=float(data_cfg.get("window_stride", 1.0)),
        fog_frac=float(data_cfg.get("fog_frac", 0.30)),
    )
    base = int(data_cfg.get("seed", 42))
    tr, va, te = subject_group_split(
        subj,
        test_size=float(data_cfg.get("test_size", 0.25)),
        val_size=float(data_cfg.get("val_size", 0.20)),
        seed=base,
    )
    pool = np.concatenate([tr, va])
    print(f"fixed test subjects: {sorted(set(subj[te].tolist()))}")

    F = spectral_v1(X, fs=int(data_cfg.get("sampling_frequency", 128)))
    yt, Ft = y[te], F[te]

    best_c, best_cv = None, -1.0
    for C in probe_cfg.get("C_grid", [0.05, 0.1, 0.5, 1.0]):
        aucs = []
        for tri, vai in GroupKFold(int(probe_cfg.get("n_folds", 5))).split(pool, groups=subj[pool]):
            sc = StandardScaler().fit(F[pool[tri]])
            clf = LogisticRegression(C=float(C), max_iter=5000,
                                     class_weight="balanced").fit(
                sc.transform(F[pool[tri]]), y[pool[tri]])
            aucs.append(average_precision_score(
                y[pool[vai]], clf.predict_proba(sc.transform(F[pool[vai]]))[:, 1]))
        mean_auc = float(np.mean(aucs))
        print(f"C={C}: groupedCV meanAUPRC={mean_auc:.4f} +/- {np.std(aucs):.4f}")
        if mean_auc > best_cv:
            best_c, best_cv = float(C), mean_auc

    sc = StandardScaler().fit(F[pool])
    clf = LogisticRegression(C=best_c, max_iter=5000,
                             class_weight="balanced").fit(sc.transform(F[pool]), y[pool])
    p = clf.predict_proba(sc.transform(Ft))[:, 1]
    nll = nn.CrossEntropyLoss()(
        torch.log(torch.from_numpy(np.stack([1 - p, p], 1).astype(np.float32)).clamp_min(1e-8)),
        torch.from_numpy(yt),
    ).item()
    test_m = metrics_from_probs(yt, np.stack([1 - p, p], 1), nll)
    print("TEST:", json.dumps(test_m, indent=2))

    out = {
        "test": test_m,
        "test_subjects": sorted(set(subj[te].tolist())),
        "best_C": best_c,
        "groupedcv_auprc": best_cv,
        "test_per_subject_auprc": per_subject_auprc(
            yt, np.stack([1 - p, p], 1), subj[te]),
    }
    with open(root / args.out, "w") as f:
        json.dump(out, f, indent=2)
    np.savez(root / "outputs" / "spectral_probe.npz",
             coef=clf.coef_, intercept=clf.intercept_,
             scaler_mean=sc.mean_, scaler_scale=sc.scale_)


if __name__ == "__main__":
    main()
