[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arannamoy/PDYF/blob/main/exp.ipynb)

```
uv sync
```

## IMU model benchmarks

The starter trainer is an IMU-only baseline. The benchmark runners compare
window-level spectral/statistical features, classical classifiers, boosted
trees, and GPU neural models using participant-grouped cross-validation.
Install optional tree libraries with `uv sync --extra benchmark`.

```bash
fydp-probe          # compact spectral logistic-regression baseline
fydp-bench          # broad classical + XGBoost/CatBoost grouped-CV sweep
fydp-bench-neural   # GPU MLP / temporal-CNN sweep
fydp-eval-blend     # evaluate the CV-selected tree rank blend
fydp-eval-probe     # detailed train/CV/test metrics for the spectral probe
fydp-eval-imu       # IMU-only research metrics, including onset and horizons
```

Results and model artifacts are written under `outputs/`. The current best
observed result is summarized in `outputs/model_comparison_summary.json`.
The experimental write-up is `docs/whitepaper_imu_fog_forecasting.md`.
The fixed participant split was repeatedly examined during development, so
these metrics are for model development; freeze the pipeline and validate on
a new participant-disjoint split before claiming final generalization.

## Detailed window-level metrics

The table below is for the saved **logistic regression on 18 IMU spectral
features** (`C=0.5`). Each example uses a 5-second observation window to predict
whether at least 30% of the following 2 seconds are annotated as FOG. This is a
future-window label, not specifically a new FOG onset.

### Ranking and subject generalization

| Evaluation | Windows / participants | Positive rate (random AUPRC) | Pooled / fold-mean AUPRC | Subject-macro AUPRC | AUROC |
|---|---:|---:|---:|---:|---:|
| Fit pool, in-sample (train + validation) | 5,814 / 26 | 0.223 | 0.901 | 0.464 (20 subjects) | 0.963 |
| 5-fold subject-grouped CV | fit pool | 0.224 mean | 0.700 ± 0.251 across folds | 0.449 (20 subjects) | 0.932 mean across folds |
| Fixed held-out test | 2,280 / 9 | 0.166 | **0.571** | **0.479 (6 subjects)** | 0.824 |

The fit-pool score is **resubstitution**: the final probe was refit on all
non-test subjects, including the original validation participants. It is
expected to be optimistic. The grouped-CV AUPRC is the mean of the five fold
AP values (the fold standard deviation is 0.251); probabilities from separate
fold models are not pooled to calculate that AUPRC. The fixed test split has
been examined during development, so it is not an untouched final estimate.

### Threshold metrics (`P(FOG) >= 0.5`)

| Metric | Fit pool, in-sample | Grouped-CV fold mean | Fixed test |
|---|---:|---:|---:|
| Accuracy | 0.901 | 0.877 | 0.825 |
| Balanced accuracy | 0.899 | 0.839 | 0.706 |
| Macro-F1 | 0.867 | 0.793 | 0.697 |
| Positive-class precision | 0.723 | 0.591 | 0.476 |
| Positive-class recall / sensitivity (window-level) | 0.897 | 0.802 | 0.526 |
| Positive-class F1 | 0.801 | 0.672 | 0.500 |
| Specificity | 0.902 | 0.876 | 0.885 |

On test, the confusion matrix in `(TN, FP, FN, TP)` order is
`(1683, 219, 179, 199)`. These are window-level threshold metrics, not episode
detection or false alarms per hour.

### Probability quality and selective prediction

| Metric | Fit pool, in-sample | Grouped-CV fold mean | Fixed test |
|---|---:|---:|---:|
| Negative log-likelihood (NLL) | 0.247 | 0.302 | 0.371 |
| Brier score (lower is better) | 0.073 | 0.090 | 0.118 |
| Brier skill vs. prevalence-only predictor | 0.576 | 0.244 | 0.146 |
| ECE of positive FOG probability (10 fixed bins) | 0.084 | 0.091 | 0.080 |
| Top-label ECE (10 fixed bins; matches prior metrics file) | 0.007 | 0.034 | 0.021 |
| Risk–coverage AURC (lower is better) | 0.024 | 0.039 | 0.058 |
| Error rate retaining most-confident 50% / 80% / 90% | 0.015 / 0.042 / 0.064 | 0.033 / 0.064 / 0.092 | 0.044 / 0.104 / 0.141 |

Risk–coverage confidence is `2 * abs(P(FOG) - 0.5)`; the most confident
windows are retained first. This is a window-level selective-prediction
summary, not a validated abstention policy. The positive-probability ECE
measures calibration of the FOG risk itself; top-label ECE measures confidence
in whichever class was predicted. ECE depends on binning and should be read as
supplementary.

### Test results by participant

| Participant | Windows | Positive rate | AUPRC |
|---|---:|---:|---:|
| SUB09 | 228 | 0.101 | 0.253 |
| SUB13 | 114 | 0.000 | N/A* |
| SUB14 | 342 | 0.313 | 0.408 |
| SUB16 | 228 | 0.211 | 0.571 |
| SUB20 | 342 | 0.032 | 0.029 |
| SUB22 | 114 | 0.000 | N/A* |
| SUB25 | 342 | 0.000 | N/A* |
| SUB27 | 342 | 0.389 | 0.798 |
| SUB30 | 228 | 0.246 | 0.813 |

`*` AUPRC is undefined for a participant with only one label class. Subject-
macro AUPRC averages the six test participants who have both positive and
negative windows. A 10,000-replicate **participant-cluster bootstrap** gives a
95% interval of **0.205–0.734** for pooled test AUPRC and **0.232–0.708** for
subject-macro AUPRC. These intervals are very uncertain with only nine test
participants; the 2,280 overlapping windows are not independent observations.

### Metrics not yet validly evaluated

| Requested project metric | Status / what is needed |
|---|---|
| Event-level AUPRC and event sensitivity | Not reported: current labels score overlapping future windows, not distinct FOG episodes. Define episode matching and alarm merging first. |
| False alarms per hour at a fixed event recall | Not reported: requires a continuous-time alarm policy and event annotations. |
| Lead time before FOG onset | Not reported: the positive label includes future-window FOG, including continuing/recurring FOG; it is not restricted to a new onset. |
| Performance at other forecast horizons | Not evaluated; this model is trained only for a 2-second horizon. |
| Missing-video robustness / video-IMU disagreement | Not applicable to this benchmark: it has no video branch or cross-modal predictions. |
| IMU corruption robustness | Not evaluated under a prespecified corruption protocol. |

Recompute the detailed report with `fydp-eval-probe`. The full per-subject
metrics, fold scores, bootstrap interval, and metric definitions are saved in
`outputs/probe_train_test_metrics.json`. Because test subjects have already
informed development, use a new participant-disjoint test cohort (or nested
grouped CV) for a final performance claim.

## IMU-only research metrics

These are the project metrics that can be computed from the IMU files alone.
No video was aligned or scored. The model is logistic regression on 18 spectral
features. The 2-second model is the saved probe (`C=0.5`). The 0.5- and
1-second models were fit with the same subject split; their `C` was selected
only on the non-test participants. Full output is in
`outputs/imu_research_metrics.json`.

### Future-window ranking on the fixed test participants

| Horizon | Positive rate | AUPRC | Subject-macro AUPRC | AUROC | Balanced accuracy | Macro-F1 | NLL | Brier | Positive-probability ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s | 0.148 | 0.606 | 0.500 | 0.848 | 0.736 | 0.724 | 0.331 | 0.103 | 0.079 |
| 1.0 s | 0.159 | 0.594 | 0.503 | 0.842 | 0.727 | 0.715 | 0.348 | 0.110 | 0.077 |
| 2.0 s | 0.166 | 0.571 | 0.479 | 0.824 | 0.706 | 0.697 | 0.371 | 0.118 | 0.080 |

Shorter horizons are easier on this split, but the subject-macro score stays near 0.48–0.50. The 2-second reliability diagram shows the middle probabilities are too high: windows scored 0.5–0.6 were actually FOG only 27% of the time, while scores above 0.9 matched an observed rate of 0.88.

### Early warning, not ongoing FOG

Restricting the test to windows whose 5-second observation contains no FOG flag gives the early-warning result:

| Horizon | FOG-free windows | Positive rate | Early-warning AUPRC |
|---:|---:|---:|---:|
| 0.5 s | 1,646 | 0.0085 | 0.023 |
| 1.0 s | 1,646 | 0.021 | 0.048 |
| 2.0 s | 1,632 | 0.040 | 0.085 |

At 2 seconds this is only about twice the random baseline. The headline window AUPRC of 0.571 is therefore mostly recognition of FOG that is already underway, not warning before it starts.

### New-onset sensitivity, false alarms, and lead time

A new onset is the start of an IMU FOG run lasting at least 0.5 seconds. It counts only when a FOG-free 5-second observation has that onset inside its future horizon. Lead time cannot exceed the horizon, because that is how an eligible onset is defined. Alarms during an ongoing episode are not called false alarms. The test recordings last about 40 minutes.

The threshold was chosen on the fit pool from `{0.2, 0.3, 0.5, 0.7, 0.9}`, keeping the value with the best onset sensitivity at no more than 5 false alarms per recording hour. That rule selects **0.9** at every horizon. On the test participants it detects **no** eligible onset, and it still produces **12 false alarms per hour**.

The other thresholds below were not used for selection. They show the tradeoff on the same test recordings:

| Horizon | Threshold | Eligible onsets | Detected | Sensitivity | Median lead time | False alarms per hour |
|---:|---:|---:|---:|---:|---:|---:|
| 2.0 s | 0.2 | 49 | 29 | 0.59 | 1.20 s | 492 |
| 2.0 s | 0.5 | 49 | 7 | 0.14 | 1.13 s | 194 |
| 2.0 s | 0.9 | 49 | 0 | 0.00 | — | 12 |
| 1.0 s | 0.5 | 49 | 7 | 0.14 | 0.60 s | 191 |
| 0.5 s | 0.5 | 22 | 3 | 0.14 | 0.13 s | 177 |

No operating point gives useful new-onset warning at a tolerable false-alarm rate. Catching about 14% of eligible onsets at threshold 0.5 still produces roughly three false alarms per minute.

### Not computed, because this run is IMU-only

Video-only performance, video–IMU agreement, missing-video robustness, and alignment sensitivity are not in this report. Recompute with `fydp-eval-imu`. These numbers use the participant split already examined during development, so they are not an untouched final estimate.
