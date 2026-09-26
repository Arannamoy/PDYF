# IMU-Only Forecasting of Near-Future Freezing of Gait

**An experimental whitepaper for the FYDP video–IMU study**

Arannamoy Mondal  
26 September 2026

**Status.** This document reports the completed inertial-measurement-unit (IMU) experiments. It is not the finished multimodal thesis. Video encoding, cross-modal agreement, latent rollout, and missing-modality tests were not run. The fixed participant split used below was examined during model development, so the test numbers are development results, not an untouched final estimate.

---

## Abstract

Freezing of gait (FOG) is a sudden interruption of walking or turning in Parkinson’s disease. The parent project asks whether temporally aligned video and wearable IMU can forecast the same near-future motor state, and whether disagreement between those forecasts is a calibrated sign that the forecast should not be trusted. This whitepaper answers the narrower question that can be answered with the IMU files alone: how well a subject-disjoint model can rank a 2-second future FOG window from 5 seconds of shank acceleration and angular velocity.

Seventy-one turning trials from 35 participants were cut into 8,094 windows. A window was labeled positive when at least 30% of the next 2 seconds carried the specialist FOG flag already stored on the IMU timeline. Participants, not windows, were assigned to train, validation, and test. Nine test participants (2,280 windows, 378 positive) were held out with seed 42.

A raw-waveform PatchTST baseline reached a test area under the precision–recall curve (AUPRC) of 0.500. Its validation AUPRC was 0.971, which did not transfer. Logistic regression on 18 freeze-band spectral features, with the penalty chosen by subject-grouped cross-validation, reached a test AUPRC of 0.571 and a subject-macro AUPRC of 0.479. A five-model tree rank blend selected by grouped cross-validation was effectively tied (0.571) but requires cohort-level rank normalization. Gradient-boosted trees and a GPU multilayer perceptron did not beat the spectral logistic model on the held-out people. Temporal convolutional networks were weaker in grouped cross-validation and were not selected.

The absolute AUPRC is easy to misread. The test positive rate is 0.166, which is the AUPRC of a random ranker, so 0.571 is about 3.4 times chance. It is not 57% accuracy. Accuracy is a poor headline: always predicting “no FOG” would be about 79% accurate. The more important finding is that the 0.571 score is mostly recognition of FOG already underway. In 91.9% of positive future windows, the preceding 5-second observation already contained a FOG flag. When the observation was FOG-free, 2-second early-warning AUPRC fell to 0.085, against a random baseline of 0.040. At a pre-specified budget of 5 false alarms per recording hour, the selected threshold detected none of 49 eligible new onsets. At probability 0.5 it detected 7 of 49, with a median lead time of 1.13 seconds, but about 194 false alarms per hour.

Class imbalance explains why the random AUPRC baseline is 0.166 rather than 0.50. It does not explain the cross-person gap. The same imbalance yields an in-sample AUPRC of 0.901 on people the model has already seen. Participant heterogeneity, overlapping windows, and a label that includes continuing freezes are the measured limits.

A separate audit found that the published video and IMU streams do not share a clock. Across 71 paired trials the median duration gap was 0.42 seconds and the largest was 7.92 seconds. That mismatch does not affect the IMU-only scores, because those scores never used video. It does block sample-level fusion until each trial has a verified offset.

The practical IMU baseline is the spectral logistic probe. It is sample-wise, simple, and slightly better calibrated than the deep baseline. It is not an early-warning device, and it is not evidence for the video–IMU agreement claim. That claim still requires alignment, a frozen onset definition, video-only and fusion arms, and a participant split that is not reused for development.

---

## 1. What this whitepaper is, and is not

The parent proposal is a task-specific latent world model for near-future human motor states, using FOG only as the application benchmark. Its intended system has a video branch, an IMU branch, a pre-fusion agreement check, probabilistic latent rollout, and calibration under missing or corrupted input. None of that multimodal system was trained here.

The experiments in this report were designed to stop three failure modes before that larger system is built.

1. **A leaky or uninformative split.** Overlapping windows from the same person must not cross the test boundary. Validation AUPRC must not be treated as evidence of cross-person skill.
2. **A model class that memorizes people.** High-capacity waveform models can look excellent on seen participants and ordinary on unseen ones.
3. **A metric that rewards the wrong event.** Window AUPRC on a label that includes ongoing FOG can look acceptable while new-onset warning fails.

The report therefore covers data construction, the subject split, classical and neural IMU models, window-level ranking and calibration, horizon sensitivity, and an onset-level false-alarm analysis. It also records the video–IMU timing audit, because that audit changes what the next experiment is allowed to claim.

---

## 2. Scientific questions addressed here

Four questions are answered with IMU data.

**Q1.** On participants who were not seen in training, can a 5-second shank-IMU window rank whether the next 2 seconds contain FOG?

**Q2.** Does a higher-capacity model improve that ranking, or does a low-capacity spectral model transfer better?

**Q3.** Is the observed AUPRC limited by class imbalance, or by subject shift and the definition of the label?

**Q4.** If the label is restricted to a new onset after a FOG-free observation, does the same model warn early at a tolerable false-alarm rate?

The parent questions are not answered here: whether video adds information, whether video–IMU disagreement predicts error, and whether a latent rollout improves multi-step forecasts. Those remain open.

---

## 3. Dataset

### 3.1 Source

The recordings are the public turning-in-place dataset of Ribeiro De Souza and colleagues (2022), distributed on Figshare (record 14984667). Thirty-five people with Parkinson’s disease performed alternating 360-degree turns for about two minutes. A Physilog IMU on the shank of the more affected side recorded triaxial acceleration and angular velocity at 128 Hz. A separate Sony camera recorded video at 30 Hz. Movement-disorder specialists identified freezing on the video. Those events were then mapped onto the IMU timeline as a binary flag. This whitepaper uses that flag. It does not re-annotate the video.

Standing trials, recorded for about 10 seconds so that a user can calibrate the sensor, were excluded. They have no turning task and no paired video. Binary `.csv` copies of the IMU files were ignored; the readable tab-separated `.txt` files were used.

### 3.2 What “synchronized” does and does not mean

The dataset paper defines the IMU start as the moment vertical acceleration exceeds 5% of its maximum, and the video start as the first visible foot movement. The published turning IMU files are then cut to exactly 120.00 seconds. The videos are not cut to that same window, and the two devices do not share a hardware clock.

A duration audit of the local files found:

| Comparison | Result |
|---|---|
| Turning IMU files used | 71 |
| Video files | 73 |
| Paired trials | 71 |
| Video without an IMU file | `PDFE10_3`, `PDFE30_3` |
| Standing IMU files without video | 35 |
| Median absolute duration gap | 0.42 s |
| Pairs differing by more than 0.5 s | 33 / 71 |
| Pairs differing by more than 1 s | 14 / 71 |
| Pairs differing by more than 2 s | 7 / 71 |
| Largest gap | 7.92 s (`SUB09_1`: IMU 120.00 s, video 112.08 s) |

This is a start-and-crop mismatch, not evidence that the trials are unrelated. A constant per-trial offset is the right first correction, because a 2-minute recording does not drift by several seconds at these sampling rates. Index alignment is not acceptable: a 2-second forecast horizon is smaller than several of the observed gaps.

The IMU-only results below are unaffected by this mismatch. The FOG flag used as the label is already on the IMU clock. The mismatch becomes a blocker only when a video frame is paired with an IMU sample.

### 3.3 Cohort imbalance

The modeling table has 8,094 windows from 71 turning trials and 35 participants. Of these, 1,673 are positive and 6,421 are negative, a positive rate of 20.7% and about 3.8 negatives per positive.

The imbalance across people is sharper than the imbalance across windows. Nine of 35 participants have no positive window. Others are almost always positive: SUB07 is 338/342 (98.8%) and SUB01 is 224/228 (98.2%). Only 16 participants have a FOG rate between 5% and 95%. There are 173 contiguous FOG runs on the IMU flag. Their median duration is 2.94 seconds; 172 last at least 0.5 seconds. One run lasts the entire 120-second trial. Twenty-six participants contribute at least one run.

---

## 4. Task, windows, and labels

Each example is one observation window of 5.0 seconds (640 samples) followed by a forecast horizon. The primary horizon is 2.0 seconds (256 samples). Windows advance by 1.0 second and never cross a trial boundary. The input is the six IMU channels in the observation only. The future samples are used solely to build the label.

The primary label is positive when the mean FOG flag in the horizon is at least 0.30. This is a future-window label. It is positive if a freeze continues, returns, or begins, provided enough of the horizon is marked. It is not, by itself, a new-onset label.

Two further labels were computed only at evaluation, not used to choose the model class.

- **Early-warning window.** The observation contains no FOG flag, and the horizon is scored with the same 30% rule.
- **New onset.** The start of a contiguous FOG run lasting at least 0.5 seconds, counted only when at least one FOG-free observation has that onset inside its horizon. Lead time is the onset time minus the earliest such warning at or above a threshold. By this definition, lead time cannot exceed the horizon.

The distinction matters. In the full window table, 91.9% of positive future windows already had a FOG flag somewhere in the preceding 5 seconds. A model can therefore obtain a moderate window AUPRC by recognizing an ongoing freeze. That is a legitimate forecasting question — will the next 2 seconds still contain FOG? — but it is not early warning.

---

## 5. Split and leakage controls

The split unit is the participant. `GroupShuffleSplit` with seed 42 assigns 25% of participants to test. A second grouped split, with seed 43, assigns 20% of the remaining participants to validation. No participant appears in more than one part.

| Part | Participants | Windows | Positive windows | Positive rate |
|---|---:|---:|---:|---:|
| Train | 19 | 4,104 | 865 | 21.1% |
| Validation | 7 | 1,710 | 430 | 25.1% |
| Test | 9 | 2,280 | 378 | 16.6% |
| Fit pool (train + validation) | 26 | 5,814 | 1,295 | 22.3% |

The test participants are SUB09, SUB13, SUB14, SUB16, SUB20, SUB22, SUB25, SUB27, and SUB30. Three of them, SUB13, SUB22, and SUB25, have no positive window, so their AUPRC is undefined. They still affect pooled ranking through their negative windows.

For the spectral logistic probe, validation is not an early-stopping set. After the test people are held out, the penalty `C` is chosen by 5-fold subject-grouped cross-validation on the other 26 people. The scaler and the final classifier are then refit on all 5,814 non-test windows. The 0.901 fit-pool AUPRC is therefore a resubstitution score. It is reported only to show memorization, not generalization.

Normalization statistics, penalty selection, and the false-alarm threshold are fit without test labels. The test split itself, however, was inspected more than once while models were compared. A final thesis number needs either a new held-out participant cohort or a nested grouped cross-validation whose outer test folds are not reused for exploration.

---

## 6. Models

All models see the same subject split and the same primary 2-second label, unless a horizon experiment says otherwise.

### 6.1 PatchTST waveform baseline

The starter model is a PatchTST-style encoder on the raw 6-channel waveform, with learned positional embeddings, train-only normalization, and optional per-window instance normalization. Training used AdamW, gradient clipping, and a one-cycle learning rate. Variants included focal loss, jitter and scale augmentation, spectral fusion, dropping extreme-rate training subjects, and a 3-seed probability ensemble. The encoder is high capacity relative to 35 people.

### 6.2 Spectral logistic probe

Each 5-second window is reduced to 18 features, three per channel: the log ratio of 3–8 Hz power to 0.5–3 Hz power, the log total power in those bands, and the channel standard deviation. The ratio is dimensionless, so sensor gain and placement scale it less than raw amplitude. A class-balanced logistic regression is fit after a train-only standard scaler. The candidate penalties are 0.01, 0.05, 0.1, 0.5, and 1.0. Grouped cross-validation selects `C = 0.5`.

This is the preferred reported model. A probability can be computed for one window without seeing the rest of the test cohort.

### 6.3 Classical and boosted-tree benchmark

The same windows were also represented as time-domain statistics, spectral features, spectral-plus-statistics, and the concatenation of those families. Subject-grouped cross-validation compared linear and shrinkage linear models, Gaussian naive Bayes, kernel methods, k-nearest neighbors, random forests, extra trees, histogram gradient boosting, GPU XGBoost, and GPU CatBoost. The single model with the highest pooled out-of-fold AUPRC was CatBoost on time-domain statistics (depth 6, `l2_leaf_reg = 10`).

A rank blend of the five highest pooled cross-validation models was also scored. Component scores on the test cohort were converted to ranks and averaged. That uses unlabeled test scores for normalization. It is not a sample-wise deployed score, and it is reported separately for that reason.

### 6.4 Neural benchmark

A GPU multilayer perceptron was trained on spectral features, spectral-plus-statistics, and all engineered features. A temporal convolutional network was trained on the raw waveform at widths 32 and 64, on an NVIDIA GeForce RTX 3060. Epochs were chosen inside nested subject-grouped cross-validation. An earlier joint run exhausted GPU memory; the reported neural comparison is the completed retry with a smaller batch. The selected neural candidate was the spectral MLP.

### 6.5 Horizon probes

The same spectral logistic family was refit for 0.5-second and 1.0-second horizons. Penalty selection again used only non-test participants. These are sensitivity checks. They are not an ensemble across horizons.

---

## 7. Metrics

AUPRC is the primary window-ranking metric because the positive class is smaller and because precision at the top of the list is the clinically relevant part of the curve. It is always reported next to the positive rate, which is the AUPRC of a random ranking. Subject-macro AUPRC is the unweighted mean of per-person AUPRC among people who have both classes. Pooled AUPRC lets a person with more windows, or a higher FOG rate, dominate.

AUROC is reported because it is less sensitive to prevalence. Balanced accuracy, macro-F1, positive precision, and positive recall are reported at probability 0.5. Accuracy is reported only as a caution.

Probability quality is reported with negative log-likelihood and the Brier score. Calibration is reported two ways: top-label expected calibration error (ECE), matching the original trainer, and ECE of the positive FOG probability itself. The second is the relevant one for a risk score. Both use 10 fixed-width bins and are supplementary, because nine test participants make a bin estimate unstable. A reliability table is included for the 2-second probe.

Selective prediction is summarized by risk–coverage: windows are retained from most confident to least confident, where confidence is twice the distance of the FOG probability from 0.5. The area under the risk–coverage curve is lower when confident windows are more often correct. This is not a validated abstention policy.

Uncertainty of the test AUPRC is reported with a 10,000-replicate percentile bootstrap that resamples participants, not windows. Window-level bootstrap would be misleading because neighboring windows overlap.

Onset metrics use a threshold chosen on the fit pool from the grid {0.2, 0.3, 0.5, 0.7, 0.9}. The rule keeps the threshold with the best new-onset sensitivity whose false-alarm rate is at most 5 per recording hour. Test operating points at the other grid values are descriptive. They were not used to pick the threshold.

---

## 8. Results

### 8.1 The waveform model does not transfer

The PatchTST baseline is the clearest negative result.

| Quantity | Validation | Test |
|---|---:|---:|
| AUPRC | 0.971 | 0.500 |
| Balanced accuracy | 0.926 | 0.661 |
| Macro-F1 | 0.894 | 0.663 |
| NLL | 0.215 | 0.409 |
| Brier | 0.057 | 0.131 |
| Top-label ECE | 0.640 | 0.651 |

Validation looks solved. The test participants are not. A 3-seed focal ensemble with instance normalization scored 0.477 test AUPRC, below the single baseline. Extra waveform capacity and deep ensembling did not buy cross-person ranking. The large ECE also means the baseline confidence is not a usable risk.

### 8.2 Spectral logistic regression is the best practical model

On the fixed test participants the spectral probe, `C = 0.5`, scores:

| Metric | Fit pool, in-sample | Grouped CV | Fixed test |
|---|---:|---:|---:|
| AUPRC | 0.901 | 0.700 ± 0.251 across folds | **0.571** |
| Subject-macro AUPRC | 0.464 (20 people) | 0.449 (20 people) | **0.479 (6 people)** |
| AUROC | 0.963 | 0.932 mean of folds | 0.824 |
| Accuracy at 0.5 | 0.901 | 0.877 | 0.825 |
| Balanced accuracy | 0.899 | 0.839 | 0.706 |
| Macro-F1 | 0.867 | 0.793 | 0.697 |
| Positive precision | 0.723 | 0.591 | 0.476 |
| Positive recall | 0.897 | 0.802 | 0.526 |
| Specificity | 0.902 | 0.876 | 0.885 |
| NLL | 0.247 | 0.302 | 0.371 |
| Brier score | 0.073 | 0.090 | 0.118 |
| Positive-probability ECE | 0.084 | 0.091 | 0.080 |
| Top-label ECE | 0.007 | 0.034 | 0.021 |

The five grouped-CV fold AUPRCs were 0.803, 0.366, 0.967, 0.438, and 0.928. The mean is 0.700 and the population standard deviation is 0.251. That spread is the result, not a nuisance: some held-out groups are easy and some are not.

The test confusion matrix at probability 0.5, in order true negative, false positive, false negative, true positive, is (1683, 219, 179, 199). Positive precision is 0.476 and positive recall is 0.526. Specificity remains 0.885. The model is better than a majority classifier on the rare class, and it is not a precise alarm.

A participant bootstrap gives a 95% interval of 0.205–0.734 for pooled test AUPRC and 0.232–0.708 for subject-macro AUPRC. With nine test people, the point estimate 0.571 is compatible with a wide range of true values. The interval is a warning about sample size, not a license to quote either endpoint as the result.

Risk–coverage on the test probe has area 0.058. The error rate among the most confident 50%, 80%, and 90% of windows is 0.044, 0.104, and 0.141. Confidence is somewhat informative. It does not create a low-error operating region large enough for an unattended alarm.

The 2-second test reliability table shows a systematic mid-range bias:

| Predicted probability | Windows | Mean prediction | Observed FOG rate |
|---|---:|---:|---:|
| 0.0–0.1 | 1,127 | 0.036 | 0.047 |
| 0.1–0.2 | 266 | 0.145 | 0.132 |
| 0.2–0.3 | 190 | 0.248 | 0.158 |
| 0.3–0.4 | 149 | 0.349 | 0.208 |
| 0.4–0.5 | 130 | 0.450 | 0.231 |
| 0.5–0.6 | 129 | 0.548 | 0.271 |
| 0.6–0.7 | 107 | 0.647 | 0.290 |
| 0.7–0.8 | 54 | 0.747 | 0.481 |
| 0.8–0.9 | 42 | 0.856 | 0.738 |
| 0.9–1.0 | 86 | 0.962 | 0.884 |

Low and very high scores are closer to the observed rate. Scores from 0.3 to 0.7 are too high. A stated probability of 0.55 corresponds to an observed FOG rate of 0.27. The score can rank, but it should not be read as a calibrated chance without a validation-only calibrator. No such calibrator was frozen before these test numbers.

### 8.3 Person-level test scores

Subject-macro AUPRC uses only the six test participants with both classes.

| Participant | Windows | Positive rate | Probe AUPRC |
|---|---:|---:|---:|
| SUB09 | 228 | 0.101 | 0.253 |
| SUB13 | 114 | 0.000 | undefined |
| SUB14 | 342 | 0.313 | 0.408 |
| SUB16 | 228 | 0.211 | 0.571 |
| SUB20 | 342 | 0.032 | 0.029 |
| SUB22 | 114 | 0.000 | undefined |
| SUB25 | 342 | 0.000 | undefined |
| SUB27 | 342 | 0.389 | 0.798 |
| SUB30 | 228 | 0.246 | 0.813 |

SUB27 and SUB30 are ranked well. SUB20, with 11 positive windows out of 342, is not ranked better than its own prevalence. The mean predicted risk of a test participant correlates 0.813 with that participant’s FOG rate. Pooled AUPRC therefore partly rewards recognizing who freezes often, not only which moment inside one person will freeze.

### 8.4 Higher capacity does not win the held-out ranking

| Model | Selection score | Test AUPRC |
|---|---:|---:|
| PatchTST waveform baseline | validation AUPRC 0.971, not comparable | 0.500 |
| PatchTST 3-seed focal ensemble | — | 0.477 |
| GPU spectral MLP | grouped-CV AUPRC 0.750; subject-macro 0.455 | 0.526 |
| CatBoost, time-domain statistics | grouped-CV AUPRC 0.886; subject-macro 0.488 | 0.553 |
| Five-model rank blend | grouped-CV AUPRC 0.895; subject-macro 0.477 | 0.571 |
| Spectral logistic regression | grouped-CV mean fold AUPRC 0.700 | **0.571** |

The blend and the logistic probe differ by 0.00007 on this split. That is a tie. The probe is preferred because it does not rank-normalize against other test windows.

The tree models’ pooled cross-validation AUPRC of 0.886–0.895 looked like a large gain. Their subject-macro cross-validation AUPRC was 0.477–0.488, almost the same as the probe’s test subject-macro of 0.479. The apparent gain was a pooled score, not better ranking inside each new person. Temporal convolutional networks were not selected: grouped-CV AUPRC was 0.577 at width 32 and 0.593 at width 64, with subject-macro values of 0.405 and 0.425.

The absolute gain of the probe over the original PatchTST test AUPRC is 0.070. Given the bootstrap interval, that gain should be described as a development improvement on this split, not as a stable effect size.

### 8.5 Shorter horizons are easier, and still not early warning

| Horizon | Test windows | Positive rate | AUPRC | Subject-macro AUPRC | AUROC | NLL | Brier | Positive ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s | 2,300 | 0.148 | 0.606 | 0.500 | 0.848 | 0.331 | 0.103 | 0.079 |
| 1.0 s | 2,300 | 0.159 | 0.594 | 0.503 | 0.842 | 0.348 | 0.110 | 0.077 |
| 2.0 s | 2,280 | 0.166 | 0.571 | 0.479 | 0.824 | 0.371 | 0.118 | 0.080 |

The 0.5- and 1.0-second probes selected `C = 1.0`, with grouped-CV mean AUPRC 0.704 and 0.702. Ranking degrades as the horizon lengthens, which is the expected direction. The subject-macro score remains near one half. Changing the horizon by itself does not remove person shift.

### 8.6 The early-warning subset is close to chance

Windows whose 5-second observation contains no FOG flag are the clean early-warning subset.

| Horizon | FOG-free test windows | Positive rate | AUPRC |
|---:|---:|---:|---:|
| 0.5 s | 1,646 | 0.0085 | 0.023 |
| 1.0 s | 1,646 | 0.021 | 0.048 |
| 2.0 s | 1,632 | 0.040 | 0.085 |

At 2 seconds the model is about twice a random ranker. That is weak evidence of anticipation. The headline 0.571 should not be cited as early-warning performance.

### 8.7 New-onset detection fails the false-alarm budget

The test recordings last about 40 minutes. Forty-nine onsets at the 2-second horizon, and the same count at 1 second, have a FOG-free observation that can legally warn. The 0.5-second horizon has 22 such onsets, because the legal warning window is narrower.

The fit-pool rule, at most 5 false alarms per recording hour, selects threshold 0.9 at every horizon. On the test participants that threshold detects no eligible onset, and it still produces 12 false alarms per hour. The budget is not met once the people change.

Descriptive test points, not used for selection, show the tradeoff at 2 seconds:

| Threshold | Detected / eligible | Sensitivity | Median lead time | False alarms per hour |
|---:|---:|---:|---:|---:|
| 0.2 | 29 / 49 | 0.59 | 1.20 s | 492 |
| 0.3 | 22 / 49 | 0.45 | 1.38 s | 360 |
| 0.5 | 7 / 49 | 0.14 | 1.13 s | 194 |
| 0.7 | 1 / 49 | 0.02 | 0.73 s | 48 |
| 0.9 | 0 / 49 | 0.00 | — | 12 |

At 1.0 and 0.5 seconds, threshold 0.5 also detects 14% of eligible onsets (7/49 and 3/22), with median lead times of 0.60 and 0.13 seconds and about 191 and 177 false alarms per hour. Lead time cannot exceed the horizon under the eligibility rule, so 1.13 seconds of a maximum 2 seconds is a late position inside the forecast window, not a long anticipation.

No grid point is a usable early-warning operating point. A sensitivity of 0.59 costs about eight false alarms per minute. A sensitivity of 0.14 still costs about three per minute.

---

## 9. Interpretation

### 9.1 Why 0.571 is not a low accuracy, and not a solved forecast

AUPRC is a ranking score. On this test set a coin-flip ranking scores about 0.166. The probe scores 0.571, and its AUROC is 0.824. Those numbers say the positive windows tend to sit above the negative windows more often than chance. They do not say that 57% of windows are correct, and they do not say that a new freeze will be announced in time.

The in-sample AUPRC of 0.901, under nearly the same class balance, shows that imbalance does not forbid a high AUPRC. The drop from 0.901 to 0.571 is what happens when the people change. The subject-macro scores make the same point more cleanly: 0.449 in grouped cross-validation and 0.479 on test. The model’s within-person ranking of unseen people is modest, and it was already modest before the test set was opened.

### 9.2 Why pooled cross-validation overstated the trees

CatBoost’s pooled grouped-CV AUPRC was 0.886. Its subject-macro value was 0.488. Comparing 0.886 with the test pooled AUPRC of 0.553 looks like a collapse. Comparing 0.488 with the test subject-macro values near 0.47 does not. Pooled AUPRC weights windows, and therefore weights participants unequally. It also mixes easy all-positive people with rare-event people. A thesis table that quotes only the pooled number will overstate both cross-validation and any model that separates high-FOG people from low-FOG people.

### 9.3 Why the label inflates window AUPRC relative to warning

The 30% future-window rule is a reasonable first operational label, and it matches the implemented trainer. It is a weak label for the proposal’s early-warning sentence. Most positive windows are continuations. Spectral power in the 3–8 Hz band is a known freezing-index family of features. It can mark a shank that is already trembling or arrested. It has much less to say about a shank that is still turning normally. The early-warning and onset tables are the evidence. They should be the tables cited if the claim is anticipation rather than short-horizon continuation.

### 9.4 Why more model capacity failed

Thirty-five participants, and nine in the test set, are not enough independent trajectories for a waveform transformer to learn a transferable freeze precursor. Neighboring windows share almost the same label, so the nominal sample size of 8,094 is not the information size. The PatchTST validation–test gap is the signature of that mismatch. The spectral ratios remove scale and keep a small parameter count. That inductive bias fits the sample size better than it fits the hope that a deeper net will discover a hidden onset pattern. The GPU MLP, given the same spectral features, still lost to logistic regression on the held-out people. Feature choice mattered more than architecture.

### 9.5 What the timing audit implies

The IMU-only chapter can proceed without repairing video alignment. The fusion chapter cannot. Pairing frame `i` with IMU sample `i`, or stretching each video to 120 seconds, would attach the wrong image to a 2-second label whenever the start offset is itself on the order of seconds. The next experiment should estimate one offset per trial by cross-correlating a video motion trace with gyroscope magnitude, accept the offset only when the peak is unique, and hold rejected trials out of sample-level fusion. Those rejected trials are a missing-video condition, not a reason to discard the participant from the IMU arm. The FOG flag should remain the label. Video should be shifted onto that clock, not used to invent a second label.

---

## 10. Limitations

1. **The test split is not untouched.** Model families were compared on the same nine people. The 0.571 figure can guide the next freeze of the pipeline. It cannot be the final generalization claim.
2. **Nine test participants make every pooled metric noisy.** The bootstrap interval is the quantitative form of that limit. Three test participants contribute no positive windows.
3. **Windows overlap.** A 1-second stride and a 2-second horizon mean adjacent labels share samples. Effective sample size is closer to the number of episodes and participants than to 8,094.
4. **The specialist flag was mapped by the dataset authors from video onto IMU time.** Residual mapping error was not measured. It could cap both window and onset scores. This report does not assign a number to that error.
5. **Onset eligibility is strict.** Onsets in the first 5 seconds, onsets already inside an observation, and runs shorter than 0.5 seconds are excluded from the warning denominator. A different onset rule would change the sensitivity. The rule was fixed before the threshold grid was read as a result, but it is still one operational choice.
6. **The false-alarm threshold was selected on in-sample fit-pool scores.** That can make the selected threshold look safer than an out-of-fold threshold. Even under that advantage, the budget did not yield a useful test sensitivity.
7. **No probability calibrator was frozen before test.** ECE and the reliability table are descriptive.
8. **The task is turning in place, with one shank IMU.** Nothing here estimates home walking, dual-task gait, or a waist or foot sensor.
9. **Video was not modeled.** No result in this paper supports, or refutes, the agreement-aware fusion hypothesis.

---

## 11. What the full thesis still has to do

The IMU chapter is sufficient to freeze three decisions.

- Use participant-disjoint evaluation, and report pooled AUPRC, subject-macro AUPRC, and a participant bootstrap together.
- Use the spectral logistic probe as the IMU baseline. Do not cite the PatchTST validation score as performance.
- Do not claim early warning from the 0.571 window AUPRC. The onset table is the current evidence, and it is negative at a tolerable false-alarm rate.

The thesis claim in the proposal still requires, in order:

1. A written onset and phase protocol: normal, pre-onset, FOG, and recovery, derived from the specialist boundaries and frozen before the next training run.
2. A per-trial video–IMU offset, with a rejection rule and a sensitivity check of plus or minus one video frame.
3. Video-only, IMU-only, naive fusion, and agreement-aware fusion on the same subject-disjoint split, after that alignment.
4. The same metrics at 0.5, 1, and 2 seconds, including onset sensitivity at a false-alarm budget chosen without test labels.
5. Calibration learned on validation participants and frozen before test.
6. A missing-video and missing-IMU condition, using unalignable trials as a real missing-video case rather than only as a simulated deletion.
7. A final participant split, or nested grouped cross-validation, that is not the split already used to choose these models.

Until those exist, the defensible sentence is: a low-capacity spectral model modestly ranks 2-second future FOG windows on unseen participants, mainly when freezing is already present, and it does not provide a low-false-alarm warning of a new onset.

---

## 12. Conclusions

1. On this development split, the best practical IMU model is logistic regression on 18 spectral features. Test AUPRC is 0.571, against a random baseline of 0.166. Subject-macro AUPRC is 0.479. AUROC is 0.824.
2. The original PatchTST baseline does not transfer (validation AUPRC 0.971, test 0.500). A deeper ensemble is worse. A GPU spectral MLP and a GPU CatBoost model do not beat the logistic probe on held-out people.
3. The large pooled cross-validation scores of the tree models are not subject-level scores. Subject-macro values stay near 0.45–0.49 for the models that were examined that way.
4. Class imbalance is real and is why accuracy and raw AUPRC are easy to overread. It is not the cause of the cross-person gap.
5. The current positive label is mostly continuation. Early-warning AUPRC is 0.085. New-onset sensitivity is zero at the false-alarm budget selected on the fit pool, and 0.14 at probability 0.5 only with about 194 false alarms per hour.
6. Video and IMU are paired at trial level but are not sample-synchronous. Fusion work has to estimate and accept a per-trial offset before any frame is joined to an IMU window.
7. These statements are strong enough to direct the next experiment. They are not a finished clinical or multimodal result.

---

## 13. Reproducibility

The fixed configuration is `configs/data.yml` and `configs/probe.yml`. Seed 42. Observation 5 seconds, primary horizon 2 seconds, stride 1 second, FOG fraction 0.30. IMU sampling rate 128 Hz.

| Command | Role |
|---|---|
| `fydp-probe` | Fit and score the spectral logistic baseline |
| `fydp-bench` | Subject-grouped classical and boosted-tree sweep |
| `fydp-bench-neural` | GPU MLP and temporal-convolution sweep |
| `fydp-eval-blend` | Score the cross-validation-selected rank blend |
| `fydp-eval-probe` | Window metrics, calibration, and participant bootstrap |
| `fydp-eval-imu` | Horizons, early-warning subset, onset sensitivity, and false alarms |

Principal artifacts:

| File | Content |
|---|---|
| `outputs/metrics_baseline.json` | PatchTST validation and test metrics |
| `outputs/metrics_probe.json` | Spectral probe test metrics and per-person AUPRC |
| `outputs/spectral_probe.npz` | Probe coefficients and scaler |
| `outputs/model_benchmark_cv.json` | Grouped-CV model ranking |
| `outputs/model_benchmark_test.json` | CatBoost test score |
| `outputs/model_benchmark_blend_test.json` | Rank-blend test score |
| `outputs/neural_benchmark.json` | GPU neural comparison |
| `outputs/probe_train_test_metrics.json` | Train, cross-validation, and test window metrics |
| `outputs/imu_research_metrics.json` | Horizon, early-warning, onset, and reliability results |
| `outputs/model_comparison_summary.json` | Short model comparison and evaluation caveat |

Hardware for the GPU runs was an NVIDIA GeForce RTX 3060 with 12 GB of memory. Tree boosting used the GPU during the grouped sweep. The final blend refit used the CPU so that it would not compete with the neural retry. The logistic probe does not need a GPU.

---

## 14. References

Ribeiro De Souza, C., Miao, R., Ávila De Oliveira, J., De Lima-Pardini, A. C., Fragoso De Campos, D., Silva-Batista, C., Teixeira, L., Shokur, S., Mohamed, B., and Coelho, D. B. (2022). A public data set of videos, inertial measurement unit, and clinical scales of freezing of gait in individuals with Parkinson’s disease during a turning-in-place task. *Frontiers in Neuroscience*, 16, 832463.

Figshare dataset record 14984667. A public dataset of video, acceleration, angular velocity, and clinical scales in individuals with Parkinson’s disease during the turning-in-place task. DOI: 10.6084/m9.figshare.14984667.

Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. arXiv:2211.14730. The PatchTST baseline is an adaptation of this architecture, not a reproduction of its published forecasting benchmark.

Internal project documents. *Cross-Modal Agreement-Aware Uncertainty-Calibrated Video-IMU Latent World Model*, proposal version 1.0, 24 August 2026, and the accompanying project-analysis report. Those documents define the multimodal claim. They are not empirical results of the experiments in this whitepaper.

---

## Appendix A. Short Bangla summary

এই রিপোর্ট শুধু IMU নিয়ে করা পরীক্ষার ফল। Video model, fusion, বা cross-modal agreement এখনও চালানো হয়নি।

৫ সেকেন্ডের shank IMU থেকে পরের ২ সেকেন্ডে FOG আছে কি না, সেটি নতুন মানুষের উপর moderateভাবে rank করা যায়। সেরা practical model হলো ১৮টি spectral feature-এর logistic regression। Test AUPRC **0.571**। Random baseline **0.166**, তাই এটি random-এর প্রায় ৩.৪ গুণ। Subject-macro AUPRC **0.479**। PatchTST baseline-এর test AUPRC ছিল **0.500**, যদিও তার validation AUPRC **0.971** ছিল। বেশি বড় neural বা tree model এই test split-এ logistic probe-কে অর্থপূর্ণভাবে হারাতে পারেনি।

**0.571** early warning নয়। Positive future window-এর ৯১.৯% ক্ষেত্রে আগের ৫ সেকেন্ডেই FOG ছিল। Observation-এ FOG না থাকলে ২ সেকেন্ডের early-warning AUPRC মাত্র **0.085**। ঘণ্টায় ৫টির বেশি false alarm না ধরে যে threshold বেছে নেওয়া হয়েছিল, সেটি test-এ কোনো নতুন onset ধরতে পারেনি। Probability 0.5-এ ৪৯টি eligible onset-এর ৭টি ধরা পড়ে, median lead time ১.১৩ সেকেন্ড, কিন্তু false alarm ঘণ্টায় প্রায় ১৯৪টি।

Video এবং IMU একই trial-এর, কিন্তু একই clock-এর নয়। ৭১ জোড়ার মধ্যে median duration gap ০.৪২ সেকেন্ড, সবচেয়ে বড় gap ৭.৯২ সেকেন্ড। IMU-only ফলের জন্য এটি সমস্যা নয়। Fusion-এর আগে প্রতিটি trial-এর offset আলাদা করে বের করতে হবে।

এই fixed test split উন্নয়নের সময় একাধিকবার দেখা হয়েছে। তাই এগুলো development ফল, final generalization estimate নয়।
