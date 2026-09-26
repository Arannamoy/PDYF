"""IMU-only research metrics for the spectral logistic probe.

Video, cross-modal disagreement, and missing-video tests are intentionally
absent. Event and lead-time numbers use the specialist FOG flag already stored
on the IMU timeline. They are development results on the previously examined
subject split.
"""
from __future__ import annotations

import json

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from fydp.data.dataloader import (
    build_all_windows,
    list_turning_trials,
    read_trial,
    subject_group_split,
    subject_of,
)
from fydp.evaluate_probe_metrics import _metrics
from fydp.paths import project_root, resolve_path
from fydp.train_probe import spectral_v1

MIN_EPISODE_S = 0.5
FALSE_ALARM_BUDGET_PER_HOUR = 5.0


def _episodes(fog: np.ndarray, fs: int) -> list[tuple[int, int]]:
    """Contiguous FOG runs of at least 0.5 seconds. Returns [start, end) pairs."""
    changes = np.diff(np.r_[0, fog > 0, 0].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    min_samples = int(round(MIN_EPISODE_S * fs))
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_samples]


def _window_rows(root, fs: int, obs_s: float, horizon_s: float, stride_s: float, fog_frac: float):
    """Window metadata in the same order as build_all_windows."""
    obs_len = int(round(obs_s * fs))
    hor_len = int(round(horizon_s * fs))
    stride = int(round(stride_s * fs))
    rows = []
    for fp in list_turning_trials(root):
        df = read_trial(fp)
        if df is None or len(df) < obs_len + hor_len:
            continue
        fog = df["fog_flag"].to_numpy(np.float32)
        t = 0
        local = []
        while t + obs_len + hor_len <= len(fog):
            future = fog[t + obs_len:t + obs_len + hor_len]
            local.append({
                "stem": fp.stem,
                "subject": subject_of(fp),
                "start": t,
                "pred_time": t + obs_len,
                "obs_has_fog": bool(np.max(fog[t:t + obs_len]) > 0),
                "label": int(float(future.mean()) >= fog_frac),
            })
            t += stride
        if local:
            rows.append((fog, local))
    flat = [row for _fog, local in rows for row in local]
    return rows, flat


def _event_report(trials, probabilities, flat_rows, fs: int, horizon_s: float, threshold: float) -> dict:
    """Score new-onset warnings. Ongoing FOG is not counted as early warning."""
    hor_len = int(round(horizon_s * fs))
    cursor = 0
    eligible = detected = 0
    leads = []
    false_alarms = during_fog = 0
    recorded_s = non_fog_s = 0.0
    onset_scores = []
    negative_scores = []
    for fog, local in trials:
        n = len(local)
        probs = probabilities[cursor:cursor + n]
        cursor += n
        recorded_s += len(fog) / fs
        non_fog_s += float(np.sum(fog <= 0)) / fs
        episodes = _episodes(fog, fs)
        alarm_clusters = []
        for row, prob in zip(local, probs):
            if prob < threshold or any(a <= row["pred_time"] < b for a, b in episodes):
                if prob >= threshold and any(a <= row["pred_time"] < b for a, b in episodes):
                    during_fog += 1
                continue
            if not alarm_clusters or row["pred_time"] - alarm_clusters[-1][0] > fs:
                alarm_clusters.append([row["pred_time"], row["obs_has_fog"]])
        matched = set()
        for pred_time, obs_has_fog in alarm_clusters:
            hit = None
            for event_i, (start, _end) in enumerate(episodes):
                if event_i in matched or obs_has_fog:
                    continue
                if pred_time <= start < pred_time + hor_len:
                    hit = event_i
                    break
            if hit is None:
                false_alarms += 1
            else:
                matched.add(hit)
        for event_i, (start, _end) in enumerate(episodes):
            warning = [
                (row, float(prob))
                for row, prob in zip(local, probs)
                if (not row["obs_has_fog"]) and row["pred_time"] <= start < row["pred_time"] + hor_len
            ]
            if not warning:
                continue
            eligible += 1
            onset_scores.append(max(prob for _row, prob in warning))
            fired = [row for row, prob in warning if prob >= threshold]
            if not fired:
                continue
            detected += 1
            earliest = min(row["pred_time"] for row in fired)
            leads.append((start - earliest) / fs)
        for row, prob in zip(local, probs):
            if (not row["obs_has_fog"]) and row["label"] == 0:
                negative_scores.append(float(prob))
    if cursor != len(probabilities):
        raise RuntimeError(f"window alignment failed: {cursor} != {len(probabilities)}")
    event_y = np.r_[np.ones(len(onset_scores)), np.zeros(len(negative_scores))]
    event_p = np.r_[onset_scores, negative_scores]
    event_auprc = None
    event_baseline = None
    if event_y.size and np.unique(event_y).size == 2:
        event_auprc = float(average_precision_score(event_y, event_p))
        event_baseline = float(event_y.mean())
    return {
        "threshold": threshold,
        "eligible_new_onsets": eligible,
        "detected_new_onsets": detected,
        "new_onset_sensitivity": (detected / eligible) if eligible else None,
        "lead_time_s_median": float(np.median(leads)) if leads else None,
        "lead_time_s_mean": float(np.mean(leads)) if leads else None,
        "lead_time_s_min": float(np.min(leads)) if leads else None,
        "lead_time_s_max": float(np.max(leads)) if leads else None,
        "false_alarms": false_alarms,
        "false_alarms_per_recording_hour": (
            false_alarms / (recorded_s / 3600.0) if recorded_s else None
        ),
        "false_alarms_per_non_fog_hour": (
            false_alarms / (non_fog_s / 3600.0) if non_fog_s else None
        ),
        "alarms_during_ongoing_fog_not_counted_as_false_alarms": during_fog,
        "recorded_hours": recorded_s / 3600.0,
        "new_onset_event_auprc": event_auprc,
        "new_onset_event_random_auprc_baseline": event_baseline,
        "new_onset_events_in_auprc": len(onset_scores),
        "fog_free_negative_windows_in_event_auprc": len(negative_scores),
    }


FIXED_THRESHOLDS = (0.2, 0.3, 0.5, 0.7, 0.9)


def _threshold_for_false_alarm_budget(trials, probabilities, fs: int, horizon_s: float) -> dict:
    """Pick from a fixed grid using fit-pool scores only."""
    grid = []
    for threshold in FIXED_THRESHOLDS:
        report = _event_report(trials, probabilities, None, fs, horizon_s, float(threshold))
        grid.append(report)
    feasible = [
        row for row in grid
        if (row["false_alarms_per_recording_hour"] or 0.0) <= FALSE_ALARM_BUDGET_PER_HOUR
    ]
    if feasible:
        selected = max(feasible, key=lambda row: (row["new_onset_sensitivity"] or 0.0, row["threshold"]))
    else:
        selected = min(grid, key=lambda row: row["false_alarms_per_recording_hour"] or 0.0)
    return {"selected_threshold": selected["threshold"], "fit_pool_grid": grid}


def _subset_trials(trials, stems: set[str]):
    return [(fog, local) for fog, local in trials if local and local[0]["stem"] in stems]


def _probs_for_rows(flat_rows, probabilities, stems: set[str]):
    mask = np.array([row["stem"] in stems for row in flat_rows])
    return probabilities[mask]


def _fit_horizon(X, y, subject, pool, c_grid):
    features = spectral_v1(X)
    best_c, best_cv, fold_scores = None, -1.0, None
    for c_value in c_grid:
        scores = []
        for train_rel, valid_rel in GroupKFold(n_splits=5).split(pool, groups=subject[pool]):
            scaler = StandardScaler().fit(features[pool[train_rel]])
            model = LogisticRegression(
                C=float(c_value), max_iter=5000, class_weight="balanced"
            ).fit(scaler.transform(features[pool[train_rel]]), y[pool[train_rel]])
            prob = model.predict_proba(scaler.transform(features[pool[valid_rel]]))[:, 1]
            scores.append(float(average_precision_score(y[pool[valid_rel]], prob)))
        mean_score = float(np.mean(scores))
        if mean_score > best_cv:
            best_c, best_cv, fold_scores = float(c_value), mean_score, scores
    scaler = StandardScaler().fit(features[pool])
    model = LogisticRegression(
        C=best_c, max_iter=5000, class_weight="balanced"
    ).fit(scaler.transform(features[pool]), y[pool])
    probabilities = model.predict_proba(scaler.transform(features))[:, 1]
    return best_c, best_cv, fold_scores, probabilities


def _saved_probabilities(X):
    artifact = np.load(project_root() / "outputs" / "spectral_probe.npz")
    features = spectral_v1(X)
    logits = (
        (features - artifact["scaler_mean"]) / artifact["scaler_scale"]
    ) @ artifact["coef"].reshape(-1) + float(artifact["intercept"].reshape(-1)[0])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def _reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_id = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    rows = []
    for b in range(bins):
        mask = bin_id == b
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "n": int(mask.sum()),
            "mean_predicted_fog_probability": None if not mask.any() else float(p[mask].mean()),
            "observed_fog_rate": None if not mask.any() else float(y[mask].mean()),
        })
    return rows


def _evaluate_split(name, y, p, subjects, trials, flat_rows, stems, fs, horizon_s, threshold):
    chosen = np.array([row["stem"] in stems for row in flat_rows])
    window = _metrics(y[chosen], p[chosen], subjects[chosen])
    clean = chosen & np.array([not row["obs_has_fog"] for row in flat_rows])
    early = None
    if clean.any() and np.unique(y[clean]).size == 2:
        early = {
            "n_windows": int(clean.sum()),
            "positive_rate_and_random_auprc_baseline": float(y[clean].mean()),
            "auprc": float(average_precision_score(y[clean], p[clean])),
            "definition": "windows whose 5-second observation contains no FOG flag",
        }
    event = _event_report(_subset_trials(trials, stems), p[chosen], None, fs, horizon_s, threshold)
    event.pop("new_onset_events_in_auprc", None)
    return {
        "split": name,
        "window_metrics": window,
        "early_warning_windows_with_fog_free_observation": early,
        "new_onset_operating_point": event,
        "reliability_diagram_positive_probability": _reliability(y[chosen], p[chosen]),
    }


def main() -> None:
    root = project_root()
    with open(root / "configs" / "data.yml") as handle:
        data_cfg = yaml.safe_load(handle)
    with open(root / "configs" / "probe.yml") as handle:
        probe_cfg = yaml.safe_load(handle)
    fs = int(data_cfg.get("sampling_frequency", 128))
    obs_s = float(data_cfg.get("observation_windows", 5.0))
    stride_s = float(data_cfg.get("window_stride", 1.0))
    fog_frac = float(data_cfg.get("fog_frac", 0.30))
    data_root = resolve_path(data_cfg["data_dir"])
    fixed_test = {
        "SUB09", "SUB13", "SUB14", "SUB16", "SUB20", "SUB22", "SUB25", "SUB27", "SUB30"
    }
    horizons = {}
    for horizon_s in (0.5, 1.0, 2.0):
        X, y, subject, _files, _stats = build_all_windows(
            data_root, fs=fs, obs_s=obs_s, horizon_s=horizon_s, stride_s=stride_s, fog_frac=fog_frac
        )
        train, val, test = subject_group_split(
            subject,
            test_size=float(data_cfg.get("test_size", 0.25)),
            val_size=float(data_cfg.get("val_size", 0.20)),
            seed=int(data_cfg.get("seed", 42)),
        )
        test_subjects = set(subject[test].tolist())
        if test_subjects != fixed_test:
            raise RuntimeError(f"unexpected test subjects at {horizon_s}s: {sorted(test_subjects)}")
        pool = np.concatenate([train, val])
        trials, flat_rows = _window_rows(data_root, fs, obs_s, horizon_s, stride_s, fog_frac)
        if len(flat_rows) != len(y):
            raise RuntimeError(f"row count {len(flat_rows)} != labels {len(y)} at {horizon_s}s")
        if [row["label"] for row in flat_rows] != y.tolist():
            raise RuntimeError(f"label order mismatch at {horizon_s}s")
        if horizon_s == 2.0:
            c_value, cv_auprc = 0.5, 0.700198644439917
            probabilities = _saved_probabilities(X)
            source = "saved spectral_probe.npz; C previously selected on non-test grouped CV"
        else:
            c_value, cv_auprc, _folds, probabilities = _fit_horizon(
                X, y, subject, pool, probe_cfg.get("C_grid", [0.5])
            )
            source = "new fit for this horizon; C selected by grouped CV on non-test subjects only"
        file_to_subject = {row["stem"]: row["subject"] for row in flat_rows}
        test_stems = {stem for stem, person in file_to_subject.items() if person in test_subjects}
        pool_stems = {stem for stem, person in file_to_subject.items() if person not in test_subjects}
        # Threshold is selected on the fit pool only, then frozen for test.
        pool_probs = _probs_for_rows(flat_rows, probabilities, pool_stems)
        threshold_info = _threshold_for_false_alarm_budget(
            _subset_trials(trials, pool_stems), pool_probs, fs, horizon_s
        )
        threshold = threshold_info["selected_threshold"]
        horizons[str(horizon_s)] = {
            "horizon_s": horizon_s,
            "model_source": source,
            "C": c_value,
            "grouped_cv_mean_auprc": cv_auprc,
            "false_alarm_budget_per_recording_hour": FALSE_ALARM_BUDGET_PER_HOUR,
            "threshold_selected_on_fit_pool": threshold,
            "threshold_grid": list(FIXED_THRESHOLDS),
            "fit_pool_in_sample": _evaluate_split(
                "fit_pool_in_sample", y, probabilities, subject, trials, flat_rows,
                pool_stems, fs, horizon_s, threshold,
            ),
            "fixed_test": _evaluate_split(
                "fixed_test", y, probabilities, subject, trials, flat_rows,
                test_stems, fs, horizon_s, threshold,
            ),
        }
        horizons[str(horizon_s)]["fixed_test"]["new_onset_threshold_grid"] = [
            _event_report(
                _subset_trials(trials, test_stems),
                _probs_for_rows(flat_rows, probabilities, test_stems),
                None, fs, horizon_s, float(threshold_value),
            )
            for threshold_value in FIXED_THRESHOLDS
        ]
        test_metrics = horizons[str(horizon_s)]["fixed_test"]["window_metrics"]
        onset = horizons[str(horizon_s)]["fixed_test"]["new_onset_operating_point"]
        print(
            f"horizon {horizon_s:.1f}s C={c_value} test AUPRC={test_metrics['auprc_average_precision']:.4f} "
            f"onset sensitivity={onset['new_onset_sensitivity']} "
            f"FA/hour={onset['false_alarms_per_recording_hour']}",
            flush=True,
        )
    report = {
        "scope": "IMU only. No video input, video alignment, or cross-modal metric is used.",
        "model": "logistic regression on 18 IMU spectral features",
        "observation_s": obs_s,
        "positive_window_rule": f"at least {100 * fog_frac:.0f}% of the future horizon is FOG",
        "episode_rule": f"contiguous IMU FOG-flag runs lasting at least {MIN_EPISODE_S:.1f} seconds",
        "new_onset_rule": (
            "An onset counts only when a window with no FOG in its 5-second observation "
            "has that onset inside its future horizon. Lead time is onset minus the earliest "
            "such warning at or above the threshold. Alarms during an ongoing episode are not false alarms."
        ),
        "threshold_rule": (
            f"On the fit pool, search thresholds {list(FIXED_THRESHOLDS)} and keep the one with "
            f"the best new-onset sensitivity whose false-alarm rate is at most "
            f"{FALSE_ALARM_BUDGET_PER_HOUR:.0f} per recording hour. Apply that frozen threshold "
            "to test. The test grid is descriptive and was not used to choose the threshold."
        ),
        "development_caveat": (
            "This fixed participant split was examined during model development. "
            "These are development metrics, not an untouched final estimate."
        ),
        "horizons": horizons,
    }
    out = root / "outputs" / "imu_research_metrics.json"
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
