"""Report window-level train/CV/test metrics for the saved spectral probe.

The train result is resubstitution on the final non-test fit pool (train plus
validation subjects). The grouped CV result is a model-selection estimate, and
the fixed test split has been examined during development; neither should be
presented as an untouched final generalization estimate.
"""
from __future__ import annotations

import json

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from fydp.data.dataloader import build_all_windows, subject_group_split
from fydp.paths import project_root, resolve_path
from fydp.train_probe import spectral_v1


def _ece_positive(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Fixed-width ECE for the positive-class (future-FOG) probability."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_id = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    value = 0.0
    for b in range(bins):
        mask = bin_id == b
        if mask.any():
            value += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value)


def _ece_top_label(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Top-label ECE, matching the convention used in the original trainer."""
    confidence = np.maximum(p, 1.0 - p)
    predicted = (p >= 0.5).astype(np.int64)
    correct = predicted == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_id = np.minimum(np.digitize(confidence, edges[1:-1]), bins - 1)
    value = 0.0
    for b in range(bins):
        mask = bin_id == b
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(value)


def _risk_coverage(y: np.ndarray, p: np.ndarray) -> dict:
    """Window-level selective risk, retaining the most confident predictions."""
    order = np.argsort(-np.abs(p - 0.5), kind="stable")
    errors = ((p >= 0.5).astype(np.int64) != y)[order].astype(np.float64)
    n = len(y)
    coverage = np.arange(1, n + 1, dtype=np.float64) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1, dtype=np.float64)
    # Include the origin and compute trapezoid area without a NumPy-version
    # dependency on np.trapezoid.
    x = np.r_[0.0, coverage]
    yy = np.r_[0.0, risk]
    aurc = float(np.sum(np.diff(x) * (yy[1:] + yy[:-1]) / 2.0))
    at_coverage = {}
    for target in (0.5, 0.8, 0.9):
        retained = max(1, int(np.ceil(target * n)))
        at_coverage[f"{int(target * 100)}_pct"] = float(errors[:retained].mean())
    return {
        "aurc_trapezoid_lower_is_better": aurc,
        "error_rate_at_coverage": at_coverage,
        "confidence_definition": "2 * abs(P(FOG) - 0.5); retain highest confidence first",
    }


def _metrics(y: np.ndarray, p: np.ndarray, subjects: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    predicted = (p >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    subject_aps = []
    per_subject = []
    for subject_id in sorted(set(subjects.tolist())):
        mask = subjects == subject_id
        ap = None
        if np.unique(y[mask]).size == 2:
            ap = float(average_precision_score(y[mask], p[mask]))
            subject_aps.append(ap)
        per_subject.append({
            "subject": str(subject_id),
            "n_windows": int(mask.sum()),
            "positive_rate": float(y[mask].mean()),
            "auprc": ap,
        })
    prevalence = float(y.mean())
    brier = float(brier_score_loss(y, p))
    return {
        "n_windows": int(len(y)),
        "n_subjects": int(len(set(subjects.tolist()))),
        "n_positive": int(y.sum()),
        "positive_rate_and_random_auprc_baseline": prevalence,
        "auprc_average_precision": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "threshold_for_classification_metrics": 0.5,
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "positive_precision": float(precision_score(y, predicted, zero_division=0)),
        "positive_recall_sensitivity": float(recall_score(y, predicted, zero_division=0)),
        "positive_f1": float(f1_score(y, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier_score": brier,
        "brier_skill_score_vs_prevalence_baseline": (
            float(1.0 - brier / (prevalence * (1.0 - prevalence)))
            if 0.0 < prevalence < 1.0 else None
        ),
        "ece_positive_probability_10_fixed_bins": _ece_positive(y, p),
        "ece_top_label_10_fixed_bins": _ece_top_label(y, p),
        "risk_coverage": _risk_coverage(y, p),
        "subject_macro_auprc": float(np.mean(subject_aps)) if subject_aps else None,
        "subjects_with_both_classes_for_macro_auprc": int(len(subject_aps)),
        "per_subject": per_subject,
    }


def _participant_bootstrap(
    y: np.ndarray, p: np.ndarray, subjects: np.ndarray, replicates: int = 10_000,
) -> dict:
    """Percentile intervals by resampling whole test participants, not windows."""
    rng = np.random.default_rng(20260926)
    ids = np.asarray(sorted(set(subjects.tolist())))
    rows = {sid: np.flatnonzero(subjects == sid) for sid in ids}
    pooled, macro = [], []
    for _ in range(replicates):
        sampled_ids = rng.choice(ids, size=len(ids), replace=True)
        indices = np.concatenate([rows[sid] for sid in sampled_ids])
        if np.unique(y[indices]).size == 2:
            pooled.append(float(average_precision_score(y[indices], p[indices])))
        aps = [
            float(average_precision_score(y[rows[sid]], p[rows[sid]]))
            for sid in sampled_ids if np.unique(y[rows[sid]]).size == 2
        ]
        if aps:
            macro.append(float(np.mean(aps)))
    return {
        "method": "percentile bootstrap resampling participants with replacement",
        "replicates": replicates,
        "seed": 20260926,
        "pooled_auprc_95pct_ci": [float(x) for x in np.percentile(pooled, [2.5, 97.5])],
        "pooled_valid_replicates": len(pooled),
        "subject_macro_auprc_95pct_ci": [float(x) for x in np.percentile(macro, [2.5, 97.5])],
        "subject_macro_valid_replicates": len(macro),
        "caveat": "only nine test participants; intervals are consequently uncertain",
    }


def main() -> None:
    root = project_root()
    with open(root / "configs" / "data.yml") as f:
        cfg = yaml.safe_load(f)
    with open(root / "outputs" / "metrics_probe.json") as f:
        previous = json.load(f)
    artifact = np.load(root / "outputs" / "spectral_probe.npz")

    X, y, subject, _, _ = build_all_windows(
        resolve_path(cfg["data_dir"]),
        fs=int(cfg.get("sampling_frequency", 128)),
        obs_s=float(cfg.get("observation_windows", 5.0)),
        horizon_s=float(cfg.get("forecast_horizon", 2.0)),
        stride_s=float(cfg.get("window_stride", 1.0)),
        fog_frac=float(cfg.get("fog_frac", 0.30)),
    )
    tr, va, te = subject_group_split(
        subject,
        test_size=float(cfg.get("test_size", 0.25)),
        val_size=float(cfg.get("val_size", 0.20)),
        seed=int(cfg.get("seed", 42)),
    )
    fit_pool = np.concatenate([tr, va])
    features = spectral_v1(X, fs=int(cfg.get("sampling_frequency", 128)))
    scaler_mean = artifact["scaler_mean"]
    scaler_scale = artifact["scaler_scale"]
    coefficients = artifact["coef"].reshape(-1)
    intercept = float(artifact["intercept"].reshape(-1)[0])
    logits = ((features - scaler_mean) / scaler_scale) @ coefficients + intercept
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

    # Report an out-of-fold grouped-CV summary as well as the in-sample fit-pool
    # score. C was selected on these same folds, so CV values are for selection.
    c_value = float(previous["best_C"])
    fold_probabilities = np.full(len(y), np.nan, dtype=np.float64)
    fold_auprc = []
    fold_metrics = []
    n_folds = 5
    for train_rel, valid_rel in GroupKFold(n_splits=n_folds).split(
        fit_pool, groups=subject[fit_pool]
    ):
        train_idx, valid_idx = fit_pool[train_rel], fit_pool[valid_rel]
        scaler = StandardScaler().fit(features[train_idx])
        model = LogisticRegression(
            C=c_value, max_iter=5000, class_weight="balanced"
        ).fit(scaler.transform(features[train_idx]), y[train_idx])
        fold_p = model.predict_proba(scaler.transform(features[valid_idx]))[:, 1]
        fold_probabilities[valid_idx] = fold_p
        fold_result = _metrics(y[valid_idx], fold_p, subject[valid_idx])
        fold_metrics.append(fold_result)
        fold_auprc.append(fold_result["auprc_average_precision"])
    cv_oof = fit_pool[np.isfinite(fold_probabilities[fit_pool])]
    cv_oof_subject_macro = _metrics(
        y[cv_oof], fold_probabilities[cv_oof], subject[cv_oof]
    )
    fold_mean = {}
    for key in (
        "positive_rate_and_random_auprc_baseline",
        "auprc_average_precision",
        "auroc",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "positive_precision",
        "positive_recall_sensitivity",
        "positive_f1",
        "specificity",
        "nll",
        "brier_score",
        "brier_skill_score_vs_prevalence_baseline",
        "ece_positive_probability_10_fixed_bins",
        "ece_top_label_10_fixed_bins",
    ):
        fold_mean[key] = float(np.mean([row[key] for row in fold_metrics]))
    fold_mean["risk_coverage"] = {
        "aurc_trapezoid_lower_is_better": float(np.mean([
            row["risk_coverage"]["aurc_trapezoid_lower_is_better"]
            for row in fold_metrics
        ])),
        "error_rate_at_coverage": {
            coverage: float(np.mean([
                row["risk_coverage"]["error_rate_at_coverage"][coverage]
                for row in fold_metrics
            ]))
            for coverage in ("50_pct", "80_pct", "90_pct")
        },
    }

    report = {
        "model": "logistic regression on 18 spectral features",
        "C": c_value,
        "task": (
            f"{cfg.get('observation_windows', 5.0)}-second IMU observation; "
            f"{cfg.get('forecast_horizon', 2.0)}-second future window; positive if "
            f"at least {100 * cfg.get('fog_frac', 0.30):g}% of future samples have FOG"
        ),
        "split_note": (
            "Fixed subject split previously examined during development; these are "
            "development metrics, not an untouched final test estimate."
        ),
        "train_fit_pool_in_sample": {
            "definition": "final refit on original train plus validation participants; resubstitution score",
            **_metrics(y[fit_pool], probabilities[fit_pool], subject[fit_pool]),
        },
        "grouped_cv_on_fit_pool": {
            "definition": "5-fold subject-grouped CV; C was selected using these same folds",
            "fold_auprc": fold_auprc,
            "fold_mean_auprc": float(np.mean(fold_auprc)),
            "fold_std_auprc_population": float(np.std(fold_auprc)),
            "fold_mean_metrics": fold_mean,
            "out_of_fold_subject_macro_auprc": cv_oof_subject_macro["subject_macro_auprc"],
            "subjects_with_both_classes_for_macro_auprc": (
                cv_oof_subject_macro["subjects_with_both_classes_for_macro_auprc"]
            ),
            "cross_fold_pooled_ranking_note": (
                "AUPRC/AUROC are reported as the mean of fold scores, not by pooling "
                "probabilities from different fold models."
            ),
        },
        "fixed_test": {
            "definition": "held-out participants; metrics computed after model selection",
            **_metrics(y[te], probabilities[te], subject[te]),
        },
        "test_participant_cluster_bootstrap": _participant_bootstrap(
            y[te], probabilities[te], subject[te]
        ),
        "not_evaluated_or_not_supported_by_current_setup": {
            "event_level_auprc_sensitivity_false_alarms_per_hour_and_lead_time": (
                "Not reported: the target is an overlapping future-window label, not a "
                "new-onset event definition with alarm aggregation."
            ),
            "other_forecast_horizons": "Not evaluated; this model was trained for one 2-second horizon.",
            "missing_modality_and_cross_modal_agreement": (
                "Not applicable: this benchmark has an IMU input only and no video branch."
            ),
            "corruption_robustness": "Not evaluated under a prespecified corruption protocol.",
        },
    }
    output_path = root / "outputs" / "probe_train_test_metrics.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {output_path}")
    print(json.dumps({
        "train_fit_pool_in_sample": {
            "auprc": report["train_fit_pool_in_sample"]["auprc_average_precision"],
            "subject_macro_auprc": report["train_fit_pool_in_sample"]["subject_macro_auprc"],
            "auroc": report["train_fit_pool_in_sample"]["auroc"],
        },
        "grouped_cv_fold_mean": {
            "auprc": report["grouped_cv_on_fit_pool"]["fold_mean_auprc"],
            "fold_std": report["grouped_cv_on_fit_pool"]["fold_std_auprc_population"],
            "subject_macro_auprc": report["grouped_cv_on_fit_pool"]["out_of_fold_subject_macro_auprc"],
        },
        "fixed_test": {
            "auprc": report["fixed_test"]["auprc_average_precision"],
            "subject_macro_auprc": report["fixed_test"]["subject_macro_auprc"],
            "auroc": report["fixed_test"]["auroc"],
        },
    }, indent=2))
    print("Test participant-bootstrap AUPRC CIs:", json.dumps(report["test_participant_cluster_bootstrap"]))


if __name__ == "__main__":
    main()
