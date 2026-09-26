"""Leakage-aware grouped-CV benchmark for IMU FOG forecasting.

Run with:
    PYTHONPATH=src .venv/bin/python -m fydp.benchmark_models

All model and feature selection is done on non-test participants using grouped
CV. The fixed subject-disjoint test labels are accessed only once, after the
winning CV candidate is selected. GPU is used for PyTorch, XGBoost and
CatBoost when available.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning

from fydp.data.dataloader import build_all_windows, subject_group_split
from fydp.paths import project_root, resolve_path

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def feature_sets(x: np.ndarray, fs: int = 128) -> dict[str, np.ndarray]:
    """Window-level features, all calculated from observation windows only."""
    n, t, c = x.shape
    freq = np.fft.rfftfreq(t, 1.0 / fs)
    # FFT-based powers; per-channel detrending removes gravity/offset effects.
    centered = x - x.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    band_edges = [(0.5, 3), (3, 8), (8, 12), (12, 20), (20, 40), (40, 60)]
    bands = []
    band_power = []
    for low, high in band_edges:
        mask = (freq >= low) & (freq < high)
        p = power[:, mask, :].sum(axis=1) + 1e-9
        bands.append(np.log(p))
        band_power.append(p)
    p_low = band_power[0]
    p_freeze = band_power[1]
    p_total = sum(band_power) + 1e-9
    spectral = np.concatenate([
        np.log(p_freeze / p_low),
        *[np.log(p / p_total) for p in band_power],
        np.log(p_total),
        np.log(np.maximum(band_power[1], 1e-9) / np.maximum(band_power[2], 1e-9)),
    ], axis=1)

    # Axis magnitudes help mitigate wearable orientation and placement changes.
    acc_mag = np.linalg.norm(centered[:, :, :3], axis=2)
    gyro_mag = np.linalg.norm(centered[:, :, 3:], axis=2)
    mag_spectral = []
    for sig in (acc_mag, gyro_mag):
        sp = np.abs(np.fft.rfft(sig, axis=1)) ** 2
        bp = [sp[:, (freq >= lo) & (freq < hi)].sum(axis=1) + 1e-9
              for lo, hi in band_edges]
        total = sum(bp) + 1e-9
        mag_spectral.extend([np.log(bp[1] / bp[0])[:, None]])
        mag_spectral.extend([np.log(p / total)[:, None] for p in bp])
        mag_spectral.append(np.log(total)[:, None])
    spectral_mag = np.concatenate(mag_spectral, axis=1)

    # Time-domain distribution features per axis.
    stats = []
    for sig in (x, centered):
        stats.extend([
            sig.mean(axis=1), sig.std(axis=1),
            np.sqrt(np.mean(sig ** 2, axis=1)),
            np.percentile(sig, 10, axis=1), np.percentile(sig, 50, axis=1),
            np.percentile(sig, 90, axis=1),
            np.mean(np.abs(np.diff(sig, axis=1)), axis=1),
        ])
    stats.append(np.stack([acc_mag.mean(1), acc_mag.std(1),
                           gyro_mag.mean(1), gyro_mag.std(1)], axis=1))
    time_stats = np.concatenate(stats, axis=1)

    # Capture short-lived onset-like changes while retaining a fixed-length
    # feature vector: 1-second chunks, centered-channel RMS and freeze ratio.
    chunks = []
    chunk_len = fs
    for start in range(0, t, chunk_len):
        chunk = centered[:, start:min(start + chunk_len, t), :]
        if chunk.shape[1] < chunk_len // 2:
            continue
        chunks.extend([np.sqrt(np.mean(chunk ** 2, axis=1))])
        cfreq = np.fft.rfftfreq(chunk.shape[1], 1.0 / fs)
        cp = np.abs(np.fft.rfft(chunk, axis=1)) ** 2
        lo = cp[:, (cfreq >= 0.5) & (cfreq < 3)].sum(axis=1) + 1e-9
        fr = cp[:, (cfreq >= 3) & (cfreq < 8)].sum(axis=1) + 1e-9
        chunks.append(np.log(fr / lo))
    time_frequency = np.concatenate(chunks, axis=1)

    # Original 18-dim feature set previously established as a useful baseline.
    old = np.concatenate([
        np.log(p_freeze / p_low), np.log(p_low + p_freeze), x.std(axis=1)
    ], axis=1)
    return {
        "spectral18": old.astype(np.float32),
        "spectral_rich": spectral.astype(np.float32),
        "spectral_magnitude": spectral_mag.astype(np.float32),
        "time_stats": time_stats.astype(np.float32),
        "spectral_plus_stats": np.concatenate([spectral, spectral_mag, time_stats], axis=1).astype(np.float32),
        "temporal_chunks": np.concatenate([spectral, time_frequency], axis=1).astype(np.float32),
        "all": np.concatenate([spectral, spectral_mag, time_stats, time_frequency], axis=1).astype(np.float32),
    }


def candidate_models() -> dict[str, object]:
    """Compact, diverse suite of classical tabular model families."""
    models: dict[str, object] = {}
    for c in [0.03, 0.1, 0.3, 1.0, 3.0]:
        models[f"logreg_c{c}"] = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=3000,
                class_weight="balanced", solver="lbfgs"))
    for c in [0.1, 1.0, 10.0]:
        models[f"rbf_svc_c{c}"] = make_pipeline(
            StandardScaler(), SVC(C=c, kernel="rbf", gamma="scale",
                                  class_weight="balanced", probability=True,
                                  random_state=42))
    for k in [5, 15, 31]:
        models[f"knn{k}"] = make_pipeline(RobustScaler(), KNeighborsClassifier(
            n_neighbors=k, weights="distance", p=2))
    for leaf in [2, 5, 10]:
        models[f"extra_trees_l{leaf}"] = ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=leaf, max_features=0.8,
            class_weight="balanced", n_jobs=-1, random_state=42)
        models[f"random_forest_l{leaf}"] = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=leaf, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    for leaf, lr in [(10, 0.05), (20, 0.05), (20, 0.1)]:
        models[f"hist_gb_l{leaf}_lr{lr}"] = HistGradientBoostingClassifier(
            learning_rate=lr, max_iter=200, max_leaf_nodes=leaf,
            l2_regularization=1.0, class_weight="balanced", random_state=42)
    models["lda_shrinkage"] = make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    models["gnb"] = GaussianNB(var_smoothing=1e-8)

    try:
        from xgboost import XGBClassifier
        for depth, reg in [(2, 5), (3, 10), (4, 10)]:
            models[f"xgboost_d{depth}_r{reg}"] = XGBClassifier(
                n_estimators=350, max_depth=depth, learning_rate=0.03,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=reg, objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", device="cuda" if torch.cuda.is_available() else "cpu",
                n_jobs=8, random_state=42)
    except ImportError:
        print("xgboost unavailable; skipping (pip install xgboost)")

    try:
        from catboost import CatBoostClassifier
        for depth, l2 in [(4, 5), (6, 10), (8, 20)]:
            models[f"catboost_d{depth}_l2{l2}"] = CatBoostClassifier(
                iterations=400, depth=depth, learning_rate=0.03, l2_leaf_reg=l2,
                loss_function="Logloss", eval_metric="PRAUC:type=Classic",
                auto_class_weights="Balanced", verbose=False, random_seed=42,
                task_type="GPU" if torch.cuda.is_available() else "CPU",
                devices="0")
    except ImportError:
        print("catboost unavailable; skipping (pip install catboost)")
    return models


def positive_probability(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
        return p[:, 1]
    score = np.asarray(model.decision_function(x))
    return 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))


def score_oof(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> tuple[float, float, int]:
    pooled = float(average_precision_score(y, p))
    per_subject = []
    for subject in np.unique(groups):
        m = groups == subject
        if y[m].min() != y[m].max():
            per_subject.append(float(average_precision_score(y[m], p[m])))
    return pooled, float(np.mean(per_subject)) if per_subject else float("nan"), len(per_subject)


def run() -> None:
    root = project_root()
    with open(root / "configs" / "data.yml") as f:
        cfg = json.loads(json.dumps(__import__("yaml").safe_load(f)))
    X, y, subjects, _, stats = build_all_windows(
        resolve_path(cfg["data_dir"]), fs=int(cfg.get("sampling_frequency", 128)),
        obs_s=float(cfg.get("observation_windows", 5.0)),
        horizon_s=float(cfg.get("forecast_horizon", 2.0)),
        stride_s=float(cfg.get("window_stride", 1.0)), fog_frac=float(cfg.get("fog_frac", 0.3)))
    tr, va, te = subject_group_split(
        subjects, test_size=float(cfg.get("test_size", 0.25)),
        val_size=float(cfg.get("val_size", 0.20)), seed=int(cfg.get("seed", 42)))
    pool = np.concatenate([tr, va])
    print(f"windows={len(y)} pool={len(pool)} held-out subjects={sorted(set(subjects[te]))}")
    print(f"GPU={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    feats = feature_sets(X, int(cfg.get("sampling_frequency", 128)))
    folds = list(GroupKFold(n_splits=5).split(pool, y[pool], groups=subjects[pool]))
    results = []
    oof_cache: dict[str, np.ndarray] = {}
    model_cache: dict[str, list] = {}
    models = candidate_models()
    for feature_name, F in feats.items():
        print(f"\nFEATURES {feature_name}: {F.shape[1]} dims")
        for model_name, template in models.items():
            oof = np.full(len(pool), np.nan, dtype=np.float64)
            fitted = []
            try:
                for fold, (fit_local, val_local) in enumerate(folds):
                    fit_idx, val_idx = pool[fit_local], pool[val_local]
                    model = clone(template)
                    model.fit(F[fit_idx], y[fit_idx])
                    oof[val_local] = positive_probability(model, F[val_idx])
                    fitted.append(model)
                pooled, macro, n_sub = score_oof(y[pool], oof, subjects[pool])
                key = f"{feature_name}::{model_name}"
                result = {"feature": feature_name, "model": model_name,
                          "cv_auprc": pooled, "cv_subject_macro_auprc": macro,
                          "cv_subjects_with_both_classes": n_sub}
                results.append(result)
                oof_cache[key] = oof.copy()
                model_cache[key] = fitted
                print(f"  {model_name:28s} pooled={pooled:.4f} subject_macro={macro:.4f}")
            except Exception as exc:
                print(f"  {model_name:28s} FAILED: {type(exc).__name__}: {exc}")

    if not results:
        raise RuntimeError("All candidates failed")
    results.sort(key=lambda r: (r["cv_auprc"], r["cv_subject_macro_auprc"]), reverse=True)
    print("\nTOP CV CANDIDATES")
    print(pd.DataFrame(results).head(20).to_string(index=False))

    # Check simple probability averages between top candidates using OOF only.
    # Rank-normalized mean ensembles are less sensitive to per-model scale.
    top_keys = [f"{r['feature']}::{r['model']}" for r in results[:10]]
    blend_rows = []
    for k in range(2, min(6, len(top_keys)) + 1):
        from scipy.stats import rankdata
        rank_preds = np.stack([
            rankdata(oof_cache[key], method="average") / len(pool) for key in top_keys[:k]
        ])
        blended = rank_preds.mean(axis=0)
        pooled, macro, n_sub = score_oof(y[pool], blended, subjects[pool])
        blend_rows.append({"models": top_keys[:k], "cv_auprc": pooled,
                           "cv_subject_macro_auprc": macro, "cv_subjects": n_sub})
        print(f"rank blend top-{k}: pooled={pooled:.4f} subject_macro={macro:.4f}")
    results_path = root / "outputs" / "model_benchmark_cv.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"fixed_test_subjects": sorted(set(subjects[te].tolist())),
                   "cv_candidates": results, "rank_blends": blend_rows}, f, indent=2)

    # Select by grouped CV only. Fit the selected candidate on the full pool.
    winner = results[0]
    key = f"{winner['feature']}::{winner['model']}"
    # Refit a clone of the same model template using training-pool observations.
    # The candidate object is identified from the dictionary, not test labels.
    final_model = clone(models[winner["model"]])
    final_model.fit(feats[winner["feature"]][pool], y[pool])
    import joblib
    joblib.dump(final_model, root / "outputs" / "model_benchmark_best.joblib")
    p_test = positive_probability(final_model, feats[winner["feature"]][te])
    y_test = y[te]  # first and only use of held-out labels in this runner
    test_ap = float(average_precision_score(y_test, p_test))
    per_subject = {}
    for subject in sorted(set(subjects[te].tolist())):
        m = subjects[te] == subject
        per_subject[subject] = (None if y_test[m].min() == y_test[m].max()
                                else float(average_precision_score(y_test[m], p_test[m])))
    report = {"winner_by_grouped_cv": winner, "test_auprc": test_ap,
              "test_positive_rate": float(y_test.mean()),
              "test_subjects": sorted(set(subjects[te].tolist())),
              "test_per_subject_auprc": per_subject,
              "test_evaluated_once_within_this_run": True,
              "fixed_split_previously_examined_in_session": True}
    with open(root / "outputs" / "model_benchmark_test.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nFINAL HELD-OUT TEST (one evaluation after CV selection)")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
