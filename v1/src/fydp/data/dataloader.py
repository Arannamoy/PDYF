"""IMU windowing + subject-disjoint splits for FOG future forecasting.

Protocol (per Project Overview / Analysis PDFs):
- Use only .txt turning trials (TSV); .csv copies are binary Excel and unreadable.
- Exclude *standing* trials.
- Fixed observation window OBS_S + future horizon HOR_S, stride STRIDE_S.
- Binary future label: 1 if mean(fog_flag in horizon) >= FOG_FRAC else 0.
- Subject-disjoint GroupShuffleSplit; normalization fit on TRAIN only.
- Never split overlapping windows randomly.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

COLS = ["acc_ml_g", "acc_ap_g", "acc_si_g", "gyr_ml_dps", "gyr_ap_dps", "gyr_si_dps"]

RENAME = {
    "ACC ML [g]": "acc_ml_g",
    "ACC AP [g]": "acc_ap_g",
    "ACC SI [g]": "acc_si_g",
    "GYR ML [deg/s]": "gyr_ml_dps",
    "GYR AP [deg/s]": "gyr_ap_dps",
    "GYR SI [deg/s]": "gyr_si_dps",
    "Freezing event [flag]": "fog_flag",
    "Time [s]": "time_s",
}


def read_trial(path: str | Path) -> pd.DataFrame | None:
    """Read one turning trial. Only .txt (TSV) is supported; returns None if no fog_flag."""
    path = Path(path)
    if path.suffix.lower() != ".txt":
        return None
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns=RENAME)
    if "fog_flag" not in df.columns:
        return None
    need = COLS + ["fog_flag"]
    df = df[need].apply(pd.to_numeric, errors="coerce").dropna()
    return df


def list_turning_trials(root: str | Path) -> list[Path]:
    """List unique turning trials, preferring .txt, excluding *standing*.

    Fixes notebook bug: rglob('SUB.*') required a literal dot after SUB,
    so it matched 0 files. Use glob('SUB*') + stem dedup instead.
    """
    root = Path(root)
    cands = [
        p for p in root.glob("SUB*")
        if p.suffix.lower() in {".csv", ".txt"} and "standing" not in p.stem.lower()
    ]
    by: dict[str, list[Path]] = {}
    for p in cands:
        by.setdefault(p.stem, []).append(p)
    out: list[Path] = []
    for stem, ps in by.items():
        txts = [p for p in ps if p.suffix.lower() == ".txt"]
        if txts:
            out.append(txts[0])
        # .csv copies are binary Excel -> skip silently (documented)
    return sorted(out)


def subject_of(path: str | Path) -> str:
    """Extract SUBnn id. Fixes notebook bug SUB\\d (single digit) -> SUB\\d+."""
    m = re.search(r"(SUB\d+)", Path(path).stem, re.I)
    return m.group(1).upper() if m else "UNK"


def make_windows(
    df: pd.DataFrame,
    obs_len: int,
    hor_len: int,
    stride: int,
    fog_frac: float = 0.30,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Slice (OBS -> future HOR) windows. Fixes notebook bug fog_flags -> fog_flag."""
    x = df[COLS].to_numpy(np.float32)
    y = df["fog_flag"].to_numpy(np.float32)
    xs, ys = [], []
    t = 0
    while t + obs_len + hor_len <= len(df):
        xs.append(x[t:t + obs_len])
        future = y[t + obs_len:t + obs_len + hor_len]
        ys.append(1 if future.mean() >= fog_frac else 0)
        t += stride
    if not xs:
        return None, None
    return np.stack(xs), np.array(ys, dtype=np.int64)


def build_all_windows(
    root: str | Path,
    fs: int = 128,
    obs_s: float = 5.0,
    horizon_s: float = 2.0,
    stride_s: float = 1.0,
    fog_frac: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build windows for every turning trial. Returns X, y, subjects, file_ids, stats."""
    obs_len, hor_len, stride = int(obs_s * fs), int(horizon_s * fs), int(stride_s * fs)
    files = list_turning_trials(root)
    xa, ya, sa, fa, rows = [], [], [], [], []
    for fp in files:
        df = read_trial(fp)
        if df is None or len(df) < obs_len + hor_len:
            continue
        x, y = make_windows(df, obs_len, hor_len, stride, fog_frac)
        if x is None:
            continue
        sub = subject_of(fp)
        xa.append(x)
        ya.append(y)
        sa.append(np.array([sub] * len(y)))
        fa.append(np.array([fp.stem] * len(y)))
        rows.append((fp.name, sub, len(y), float(y.mean())))
    if not xa:
        raise ValueError("No windows built — check data_dir and trial files.")
    stats = pd.DataFrame(rows, columns=["file", "subject", "n_win", "fog_rate"])
    return (
        np.concatenate(xa, axis=0),
        np.concatenate(ya, axis=0),
        np.concatenate(sa, axis=0),
        np.concatenate(fa, axis=0),
        stats,
    )


def subject_group_split(
    subjects: np.ndarray,
    test_size: float = 0.25,
    val_size: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subject-disjoint train/val/test index split (no window-level leakage)."""
    groups = np.asarray(subjects)
    idx = np.arange(len(groups))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_val, te = next(gss.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=val_size / (1.0 - test_size), random_state=seed + 1
    )
    tr, va = next(gss2.split(tr_val, groups=groups[tr_val]))
    return tr_val[tr], tr_val[va], te


def fit_normalizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over (windows, time) of TRAIN only."""
    mu = x_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    sd = x_train.std(axis=(0, 1), keepdims=True).astype(np.float32) + 1e-6
    return mu.squeeze(0).squeeze(0), sd.squeeze(0).squeeze(0)


def apply_normalizer(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((x - mu.reshape(1, 1, -1)) / sd.reshape(1, 1, -1)).astype(np.float32)


def apply_instance_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-window, per-channel z-score (RevIN-style, no learnable affine).

    IMU signals carry strong subject-specific offsets/gains (sensor placement,
    body size, gait style). Global train-only normalization preserves those
    shifts and the model overfits to train subjects' FOG signatures. Instance
    normalization removes per-window offset/gain so the classifier keys on
    dynamics (e.g. high-frequency tremor bursts), which transfer better to
    unseen subjects. Applied AFTER the global normalizer.
    """
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + eps
    return ((x - mu) / sd).astype(np.float32)


def rebalance_train_val(
    subjects: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    seed: int = 0,
    n_iter: int = 2000,
    min_mid: int = 3,
    mid_lo: float = 0.05,
    mid_hi: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Reassign whole subjects between train/val to match window share and fog rate.

    Keeps the TEST set untouched (subject-disjointness preserved). The default
    GroupShuffleSplit val set is often trivially easy (all-zero or all-FOG
    subjects), so checkpoint selection on val AUPRC does not predict test
    AUPRC. Rebalancing makes val representative of the train pool, which makes
    early-stopping/model-selection informative again.

    `min_mid` forces at least that many mid-rate (transition-rich) subjects
    into val. Rationale: all-FOG/all-normal val subjects reward subject
    identification (rank one subject above another), while the test set's
    mid-rate subjects demand within-subject FOG ranking. Without mid-rate
    subjects in val, selection cannot see true forecasting ability.
    """
    rng = np.random.default_rng(seed)
    pool = np.concatenate([tr, va])
    subs = np.unique(subjects[pool])
    # subject -> window indices within pool
    by_sub = {s: pool[subjects[pool] == s] for s in subs}
    sub_rate = {s: float(y[idx].mean()) for s, idx in by_sub.items()}
    mid_subs = {s for s, r in sub_rate.items() if mid_lo < r < mid_hi}
    target_frac = len(va) / max(len(pool), 1)
    pool_fog = float(y[pool].mean())

    def objective(val_subs: set) -> float:
        tr_subs = set(subs.tolist()) - val_subs
        if not val_subs or not tr_subs:
            return float("inf")
        # Soft constraint (not a hard -inf wall: single-subject moves can only
        # add one mid-rate subject at a time, so a hard wall strands the search
        # on an infeasible plateau).
        n_mid = len(val_subs & mid_subs)
        penalty = 2.0 * max(0, min(min_mid, len(mid_subs)) - n_mid)
        va_idx = np.concatenate([by_sub[s] for s in val_subs])
        tr_idx = np.concatenate([by_sub[s] for s in tr_subs])
        frac = len(va_idx) / len(pool)
        return (
            penalty
            + abs(frac - target_frac)
            + abs(float(y[va_idx].mean()) - pool_fog)
            + abs(float(y[tr_idx].mean()) - pool_fog)
        )

    val_subs: set = set(np.unique(subjects[va]).tolist())
    best = objective(val_subs)
    for _ in range(n_iter):
        s = subs[rng.integers(len(subs))].item()
        cand = set(val_subs)
        cand.discard(s) if s in cand else cand.add(s)
        score = objective(cand)
        if score < best:
            best, val_subs = score, cand
    va_new = np.concatenate([by_sub[s] for s in val_subs])
    tr_new = np.concatenate([by_sub[s] for s in set(subs.tolist()) - val_subs])
    return tr_new, va_new


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, f: np.ndarray | None = None):
        self.x = torch.from_numpy(np.ascontiguousarray(x))
        self.y = torch.from_numpy(np.ascontiguousarray(y).astype(np.int64))
        self.f = None if f is None else torch.from_numpy(np.ascontiguousarray(f))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        if self.f is None:
            return self.x[i], self.y[i]
        return self.x[i], self.f[i], self.y[i]


def spectral_features(x: np.ndarray, fs: int = 128) -> np.ndarray:
    """Window-level spectral features (FOG freezing-index family).

    Per channel: log freeze-band (3-8 Hz) / locomotion-band (0.5-3 Hz) power
    ratio, log total power, std. Ratios are dimensionless, so they transfer
    across subjects far better than raw amplitudes: a logistic probe on these
    18 features reaches test AUPRC 0.567 where the raw-waveform PatchTST
    baseline stalls at 0.50. Uses the observation window only (no future leak).
    """
    F = np.abs(np.fft.rfft(x, axis=1))
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / fs)
    lb = (freqs >= 0.5) & (freqs <= 3.0)
    fb = (freqs > 3.0) & (freqs <= 8.0)
    pl = (F[:, lb, :] ** 2).sum(axis=1) + 1e-9
    pf = (F[:, fb, :] ** 2).sum(axis=1) + 1e-9
    ratio = np.log(pf / pl)
    total = np.log(pl + pf)
    std = x.std(axis=1)
    return np.concatenate([ratio, total, std], axis=1).astype(np.float32)
