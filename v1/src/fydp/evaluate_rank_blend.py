"""Fit the grouped-CV-selected top-five rank blend and score test once.

This blend was chosen by pooled OOF grouped-CV AUPRC in
outputs/model_benchmark_cv.json (no test labels were involved in selection).
Each component is refit on all non-test participants; test predictions are
rank-averaged. This batch rank normalization uses only unlabeled test scores.
"""
from __future__ import annotations

import gc
import json

import joblib
import numpy as np
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import average_precision_score

from fydp.benchmark_models import candidate_models, feature_sets
from fydp.data.dataloader import build_all_windows, subject_group_split
from fydp.paths import project_root, resolve_path


def main() -> None:
    import yaml
    root = project_root()
    with open(root / "configs" / "data.yml") as f:
        cfg = yaml.safe_load(f)
    with open(root / "outputs" / "model_benchmark_cv.json") as f:
        cv = json.load(f)
    blend = max(cv["rank_blends"], key=lambda r: r["cv_auprc"])
    components = [tuple(entry.split("::", 1)) for entry in blend["models"]]
    X, y, subject, _, _ = build_all_windows(
        resolve_path(cfg["data_dir"]), fs=int(cfg.get("sampling_frequency", 128)),
        obs_s=float(cfg.get("observation_windows", 5.0)),
        horizon_s=float(cfg.get("forecast_horizon", 2.0)),
        stride_s=float(cfg.get("window_stride", 1.0)),
        fog_frac=float(cfg.get("fog_frac", 0.3)))
    tr, va, te = subject_group_split(
        subject, test_size=float(cfg.get("test_size", .25)),
        val_size=float(cfg.get("val_size", .2)), seed=int(cfg.get("seed", 42)))
    pool = np.concatenate([tr, va])
    features = feature_sets(X, int(cfg.get("sampling_frequency", 128)))
    templates = candidate_models()
    ranks = []
    estimators = []
    for feature_name, model_name in components:
        print(f"fit {feature_name}::{model_name}", flush=True)
        model = clone(templates[model_name])
        # The grouped benchmark and neural retry share the RTX 3060. Fit this
        # small final blend on CPU to avoid competing for GPU memory; the same
        # XGBoost/CatBoost configurations were already benchmarked on CUDA.
        if model_name.startswith("xgboost"):
            model.set_params(device="cpu", n_jobs=4)
        elif model_name.startswith("catboost"):
            model.set_params(task_type="CPU", thread_count=4)
        model.fit(features[feature_name][pool], y[pool])
        p = model.predict_proba(features[feature_name][te])[:, 1]
        ranks.append(rankdata(p, method="average") / len(p))
        estimators.append((feature_name, model_name, model))
        gc.collect()
    p_blend = np.mean(np.stack(ranks), axis=0)
    # The first access to y[te] in this evaluator is final reporting.
    y_test = y[te]
    test_ap = float(average_precision_score(y_test, p_blend))
    per_sub = {}
    for s in sorted(set(subject[te].tolist())):
        mask = subject[te] == s
        per_sub[s] = None if y_test[mask].min() == y_test[mask].max() else float(
            average_precision_score(y_test[mask], p_blend[mask]))
    report = {
        "blend": blend,
        "test_auprc": test_ap,
        "test_positive_rate": float(y_test.mean()),
        "test_subjects": sorted(set(subject[te].tolist())),
        "test_per_subject_auprc": per_sub,
        "test_rank_normalization": "within unlabeled test batch, per component",
        "test_evaluated_once_within_this_run": True,
        "fixed_split_previously_examined_in_session": True,
    }
    with open(root / "outputs" / "model_benchmark_blend_test.json", "w") as f:
        json.dump(report, f, indent=2)
    joblib.dump({"components": estimators, "rank_average": True},
                root / "outputs" / "model_benchmark_blend.joblib")
    np.save(root / "outputs" / "model_benchmark_blend_test_scores.npy", p_blend)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
