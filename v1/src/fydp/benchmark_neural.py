"""Nested subject-grouped CV for GPU neural IMU models (TCN / feature MLP).

Outer held-out participants produce out-of-fold scores; early stopping and
normalization use only participants inside each outer training fold. The fixed
test set is scored only after the best neural family is selected by OOF CV.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from fydp.benchmark_models import feature_sets, score_oof
from fydp.data.dataloader import build_all_windows, subject_group_split
from fydp.models.temporal import FeatureMLP, TemporalCNN
from fydp.paths import project_root, resolve_path


def normalize_per_window(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-5
    return ((x - mean) / std).astype(np.float32)


def fit_feature_norm(train: np.ndarray, valid: np.ndarray, test: np.ndarray):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-5
    return (((train - mu) / sd).astype(np.float32),
            ((valid - mu) / sd).astype(np.float32),
            ((test - mu) / sd).astype(np.float32))


def fit_one(model, x_train, y_train, x_stop, y_stop, device, seed,
            max_epochs=40, batch=256, lr=1e-3, patience=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    pos = max(float(y_train.mean()), 1e-3)
    neg = 1.0 - pos
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(min(neg / pos, 5.0), device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=False)
    xv = torch.from_numpy(x_stop).to(device)
    yv = y_stop
    best_ap, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xv)).cpu().numpy()
        ap = (float(average_precision_score(yv, pv))
              if len(np.unique(yv)) > 1 else -float(np.mean((pv - yv) ** 2)))
        if ap > best_ap:
            best_ap, best_epoch, bad = ap, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, best_ap


def predict(model, x, device, batch=512):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.from_numpy(x[start:start + batch]).to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def run():
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"neural benchmark device={device} " +
          (torch.cuda.get_device_name(0) if device == "cuda" else ""))
    root = project_root()
    import yaml
    with open(root / "configs" / "data.yml") as f:
        cfg = yaml.safe_load(f)
    X, y, subjects, _, _ = build_all_windows(
        resolve_path(cfg["data_dir"]), fs=int(cfg.get("sampling_frequency", 128)),
        obs_s=float(cfg.get("observation_windows", 5.0)),
        horizon_s=float(cfg.get("forecast_horizon", 2.0)),
        stride_s=float(cfg.get("window_stride", 1.0)), fog_frac=float(cfg.get("fog_frac", .3)))
    tr, va, te = subject_group_split(
        subjects, test_size=float(cfg.get("test_size", .25)),
        val_size=float(cfg.get("val_size", .2)), seed=int(cfg.get("seed", 42)))
    pool = np.concatenate([tr, va])
    groups = subjects[pool]
    folds = list(GroupKFold(5).split(pool, y[pool], groups=groups))
    F = feature_sets(X, int(cfg.get("sampling_frequency", 128)))
    raw = normalize_per_window(X)
    candidates = [
        ("mlp_spectral18", "spectral18", "feature", 64, 2, .25),
        ("mlp_spectral_plus_stats", "spectral_plus_stats", "feature", 64, 2, .30),
        ("mlp_all", "all", "feature", 128, 2, .35),
        ("tcn_w32", "raw", "temporal", 32, 4, .15),
        ("tcn_w64", "raw", "temporal", 64, 4, .20),
    ]
    results = []
    oof = {}
    epochs_by_name = {}
    for name, fname, kind, width, depth, dropout in candidates:
        print(f"\n{name}")
        pred = np.full(len(pool), np.nan, dtype=np.float64)
        chosen_epochs = []
        for fi, (fit_local, held_local) in enumerate(folds):
            fit_idx, held_idx = pool[fit_local], pool[held_local]
            inner_train_local, stop_local = next(GroupShuffleSplit(
                n_splits=1, test_size=.2, random_state=500 + fi
            ).split(fit_idx, y[fit_idx], groups=subjects[fit_idx]))
            inner_idx, stop_idx = fit_idx[inner_train_local], fit_idx[stop_local]
            if kind == "feature":
                a, b, c = fit_feature_norm(F[fname][inner_idx], F[fname][stop_idx], F[fname][held_idx])
                model = FeatureMLP(a.shape[1], width=width, dropout=dropout, depth=depth)
            else:
                a, b, c = raw[inner_idx], raw[stop_idx], raw[held_idx]
                model = TemporalCNN(width=width, dropout=dropout, n_blocks=depth)
            model, best_epoch, stop_ap = fit_one(
                model, a, y[inner_idx], b, y[stop_idx], device,
                seed=100 + fi, max_epochs=30,
                batch=256 if kind == "feature" else 64, patience=8)
            chosen_epochs.append(best_epoch)
            pred[held_local] = predict(model, c, device)
            print(f"  fold {fi + 1}: stop_ap={stop_ap:.4f} epoch={best_epoch} "
                  f"held_subj={len(set(subjects[held_idx]))}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        pooled, macro, n_sub = score_oof(y[pool], pred, groups)
        results.append({"model": name, "feature": fname, "cv_auprc": pooled,
                        "cv_subject_macro_auprc": macro,
                        "cv_subjects_with_both_classes": n_sub,
                        "fold_epochs": chosen_epochs})
        oof[name] = pred
        epochs_by_name[name] = max(3, int(np.median(chosen_epochs)))
        print(f"  OOF pooled={pooled:.4f} subject_macro={macro:.4f}")

    results.sort(key=lambda r: (r["cv_auprc"], r["cv_subject_macro_auprc"]), reverse=True)
    print("\nCV results:\n", json.dumps(results, indent=2))

    # Only the CV-selected neural model is evaluated on the fixed test set.
    winner = results[0]
    name = winner["model"]
    _, fname, kind, width, depth, dropout = next(c for c in candidates if c[0] == name)
    if kind == "feature":
        a, t, test_x = fit_feature_norm(F[fname][pool], F[fname][pool], F[fname][te])
        final_model = FeatureMLP(a.shape[1], width=width, dropout=dropout, depth=depth)
    else:
        a, t, test_x = raw[pool], raw[pool], raw[te]
        final_model = TemporalCNN(width=width, dropout=dropout, n_blocks=depth)
    # Final fixed epoch count is derived from nested-CV stopping epochs.
    ep = epochs_by_name[name]
    final_model = final_model.to(device)
    pos = max(float(y[pool].mean()), 1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min((1 - pos) / pos, 5.0), device=device))
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(torch.from_numpy(a), torch.from_numpy(y[pool].astype(np.float32))),
                        batch_size=256 if kind == "feature" else 64, shuffle=True)
    torch.manual_seed(42)
    for _ in range(ep):
        final_model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(final_model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
    test_pred = predict(final_model, test_x, device)
    y_test = y[te]
    test_ap = float(average_precision_score(y_test, test_pred))
    per_sub = {}
    for s in sorted(set(subjects[te].tolist())):
        mask = subjects[te] == s
        per_sub[s] = None if y_test[mask].min() == y_test[mask].max() else float(
            average_precision_score(y_test[mask], test_pred[mask]))
    report = {"winner_by_nested_group_cv": winner, "test_auprc": test_ap,
              "test_per_subject_auprc": per_sub,
              "test_subjects": sorted(set(subjects[te].tolist())),
              "device": device, "final_epochs": ep}
    with open(root / "outputs" / "neural_benchmark.json", "w") as f:
        json.dump({"results": results, "test": report}, f, indent=2)
    torch.save(final_model.cpu().state_dict(), root / "outputs" / "neural_best.pt")
    print("\nFINAL", json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
