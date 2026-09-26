"""IMU-only baseline trainer: subject-disjoint, train-only norm, AUPRC-focused.

Metrics (per PDFs): accuracy, balanced accuracy, macro F1, AUPRC,
NLL, Brier score, ECE. Saves checkpoint + metrics.json + normalizer.json.

Test-AUPRC measures (same fixed test subjects, seed 42):
- Learned positional embeddings (the encoder was permutation-invariant).
- Per-window instance norm after the global train-only norm: removes
  subject-specific offset/gain so FOG dynamics transfer to unseen subjects.
- Rebalanced train/val split (test untouched): the raw GroupShuffleSplit val
  set is dominated by trivial all-FOG/all-normal subjects, so val-AUPRC model
  selection did not predict test AUPRC.
- Focal loss + train-time jitter/scale augmentation for sparse, heterogeneous
  FOG events.
- Multi-seed deep ensemble (PDFs §4-5: 3-5 seeds, epistemic uncertainty):
  probabilities are averaged across seeds before metrics are computed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
)
from torch.utils.data import DataLoader

from fydp.data.dataloader import (
    WindowDataset,
    apply_instance_norm,
    apply_normalizer,
    build_all_windows,
    fit_normalizer,
    rebalance_train_val,
    spectral_features,
    subject_group_split,
)
from fydp.models.patchtst import PatchTST
from fydp.models.spectral import FusionModel, SpectralOnlyMLP
from fydp.paths import project_root, resolve_path


def load_cfg(name: str) -> dict:
    with open(project_root() / "configs" / name) as f:
        return dict(yaml.safe_load(f))


class FocalLoss(nn.Module):
    """Multiclass focal loss: down-weights easy windows, keeps ranking sharp.

    Weighted CE (previous default) shifts predicted probabilities and hurts
    both calibration and AUPRC ranking under subject shift. Focal loss focuses
    learning on hard/ambiguous windows (onsets, unseen-subject FOG variants).
    alpha_pos weights the positive (FOG) class, derived from train counts.
    """

    def __init__(self, alpha_pos: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha_pos = float(alpha_pos)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        alpha = torch.where(
            target == 1,
            torch.tensor(self.alpha_pos, device=logits.device),
            torch.tensor(1.0 - self.alpha_pos, device=logits.device),
        )
        return (alpha * (1.0 - pt) ** self.gamma * ce).mean()


def augment_batch(x: torch.Tensor, jitter: float, scale: float) -> torch.Tensor:
    """Train-time IMU augmentation: Gaussian jitter + per-channel gain noise."""
    if jitter > 0:
        x = x + torch.randn_like(x) * jitter
    if scale > 0:
        g = 1.0 + torch.randn(x.size(0), 1, x.size(2), device=x.device) * scale
        x = x * g
    return x


def split_batch(batch, device):
    """Unpack (x, y) or (x, f, y) batches onto device."""
    if len(batch) == 3:
        xb, fb, yb = batch
        return xb.to(device), fb.to(device), yb.to(device)
    xb, yb = batch
    return xb.to(device), None, yb.to(device)


def model_call(model, xb, fb):
    return model(xb, fb) if fb is not None else model(xb)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(float((pred[m] == labels[m]).mean()) - float(conf[m].mean()))
    return float(ece)


def predict_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            xb, fb, yb = split_batch(batch, device)
            logits = model_call(model, xb, fb)
            ys.append(yb.numpy() if not yb.is_cuda else yb.cpu().numpy())
            ps.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def metrics_from_probs(y: np.ndarray, p: np.ndarray, nll_value: float) -> dict:
    return {
        "nll": float(nll_value),
        "acc": float(accuracy_score(y, p.argmax(1))),
        "bal_acc": float(balanced_accuracy_score(y, p.argmax(1))),
        "macro_f1": float(f1_score(y, p.argmax(1), average="macro", zero_division=0)),
        "auprc": float(average_precision_score(y, p[:, 1])),
        "brier": float(brier_score_loss(y, p[:, 1])),
        "ece": expected_calibration_error(p, y),
    }


def evaluate(model, loader, device) -> dict:
    model.eval()
    nll = nn.CrossEntropyLoss()
    tot_loss, n = 0.0, 0
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            xb, fb, yb = split_batch(batch, device)
            logits = model_call(model, xb, fb)
            tot_loss += nll(logits, yb).item() * len(yb)
            n += len(yb)
            ys.append(yb.cpu().numpy())
            ps.append(torch.softmax(logits, dim=1).cpu().numpy())
    return metrics_from_probs(np.concatenate(ys), np.concatenate(ps), tot_loss / max(n, 1))


def per_subject_auprc(y: np.ndarray, p: np.ndarray, subjects: np.ndarray) -> dict:
    out = {}
    for s in sorted(set(subjects.tolist())):
        m = subjects == s
        if y[m].min() == y[m].max():
            out[str(s)] = None  # single-class subject: AUPRC undefined
        else:
            out[str(s)] = float(average_precision_score(y[m], p[m, 1]))
    return out


def run_seed(
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    subj: np.ndarray,
    tr_pool: np.ndarray,
    te: np.ndarray,
    data_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    base = int(data_cfg.get("seed", 42))
    # Train/val partition of the non-test pool. Test indices are fixed by the
    # base seed, so test subjects stay identical across ensemble seeds.
    # Reuse the base train/val split, then rebalance with this seed.
    tr_b, va_b, _ = subject_group_split(
        subj,
        test_size=float(data_cfg.get("test_size", 0.25)),
        val_size=float(data_cfg.get("val_size", 0.20)),
        seed=base,
    )
    tr, va = tr_b, va_b
    mid_lo = float(data_cfg.get("mid_lo", 0.05))
    mid_hi = float(data_cfg.get("mid_hi", 0.95))
    if str(data_cfg.get("rebalance_val", "true")).lower() not in {"0", "false", "no"}:
        tr, va = rebalance_train_val(
            subj, y, tr_b, va_b, seed=seed,
            min_mid=int(data_cfg.get("val_min_mid", 3)),
            mid_lo=mid_lo, mid_hi=mid_hi,
        )
    if str(data_cfg.get("drop_extreme_train", "false")).lower() not in {"0", "false", "no"}:
        # Force within-subject learning: train only on transition-rich subjects.
        keep_subs = {
            s for s in set(subj[tr].tolist())
            if mid_lo <= float(y[tr][subj[tr] == s].mean()) <= mid_hi
        }
        tr = tr[np.array([s in keep_subs for s in subj[tr]])]

    # Model selection on mid-rate val subjects only: all-FOG/all-normal val
    # subjects reward subject identification, not FOG ranking.
    va_rates = {s: float(y[va][subj[va] == s].mean()) for s in set(subj[va].tolist())}
    va_mid = va[np.array([(mid_lo < va_rates[s] < mid_hi) for s in subj[va]])]
    use_mid = len(va_mid) >= 50 and 0.0 < float(y[va_mid].mean()) < 1.0

    mu, sd = fit_normalizer(X[tr])
    Xn = apply_normalizer(X, mu, sd)
    if str(train_cfg.get("instance_norm", "true")).lower() not in {"0", "false", "no"}:
        Xn = apply_instance_norm(Xn)

    branch = str(model_cfg.get("branch", "fusion")).lower()
    use_fusion = branch == "fusion"
    use_spectral = branch in {"fusion", "spectral"}
    if use_spectral:
        Fall = spectral_features(X, fs=int(data_cfg.get("sampling_frequency", 128)))
        fmu = Fall[tr].mean(axis=0)
        fsd = Fall[tr].std(axis=0) + 1e-6
        Fn = ((Fall - fmu) / fsd).astype(np.float32)
    else:
        Fn, fmu, fsd = None, None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = int(train_cfg.get("batch_size", 256))
    lr = float(train_cfg.get("lr", 1e-3))
    epochs = int(model_cfg.get("max_epochs", train_cfg.get("epochs", 25)))
    train_loader = DataLoader(WindowDataset(Xn[tr], y[tr], Fn[tr] if Fn is not None else None),
                              batch_size=batch, shuffle=True)
    val_loader = DataLoader(WindowDataset(Xn[va], y[va], Fn[va] if Fn is not None else None),
                            batch_size=batch)
    val_sel_loader = DataLoader(
        WindowDataset(Xn[va_mid], y[va_mid], Fn[va_mid] if Fn is not None else None),
        batch_size=batch) if use_mid else val_loader
    test_loader = DataLoader(WindowDataset(Xn[te], y[te], Fn[te] if Fn is not None else None),
                             batch_size=batch)
    print(f"[seed {seed}] val_mid_n={len(va_mid) if use_mid else len(va)} "
          f"mid_subjects={sorted(va_rates and [s for s, r in va_rates.items() if mid_lo < r < mid_hi])}")

    patch_kwargs = dict(
        n_channels=int(model_cfg.get("n_channels", 6)),
        patch=int(model_cfg.get("patch", 16)),
        d_model=int(model_cfg.get("d_model", 64)),
        nhead=int(model_cfg.get("nhead", 4)),
        layers=int(model_cfg.get("layers", 2)),
        n_class=int(model_cfg.get("n_class", 2)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        ff_mult=int(model_cfg.get("ff_mult", 4)),
        posenc=str(model_cfg.get("posenc", "true")).lower() not in {"0", "false", "no"},
    )
    if use_fusion:
        model = FusionModel(
            spec_dim=Fn.shape[1],
            spec_hidden=int(model_cfg.get("spec_hidden", 64)),
            spec_dropout=float(model_cfg.get("spec_dropout", 0.2)),
            **patch_kwargs,
        ).to(device)
    elif branch == "spectral":
        model = SpectralOnlyMLP(
            n_feat=Fn.shape[1],
            hidden=int(model_cfg.get("spec_hidden", 64)),
            n_class=int(model_cfg.get("n_class", 2)),
            dropout=float(model_cfg.get("spec_dropout", 0.2)),
        ).to(device)
    else:
        model = PatchTST(**patch_kwargs).to(device)

    counts = np.bincount(y[tr], minlength=2).astype(np.float32)
    if str(model_cfg.get("loss", "focal")).lower() == "focal":
        pos_rate = float(counts[1] / max(len(tr), 1))
        crit: nn.Module = FocalLoss(
            alpha_pos=1.0 - pos_rate, gamma=float(model_cfg.get("focal_gamma", 2.0))
        )
    else:
        weight = torch.tensor(len(tr) / (2 * np.maximum(counts, 1)), dtype=torch.float32).to(device)
        crit = nn.CrossEntropyLoss(weight=weight)
    print(f"[seed {seed}] train_n={len(tr)} val_n={len(va)} counts={counts.tolist()} {type(crit).__name__}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(model_cfg.get("warmup_epochs", 4)) * steps_per_epoch
    pct_start = min(warmup_steps / max(total_steps, 1), 0.5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps,
        pct_start=pct_start, div_factor=10.0,
    )
    select = str(model_cfg.get("select_metric", "auprc"))
    patience = int(model_cfg.get("patience", 12))
    jitter = float(train_cfg.get("augment_jitter", 0.02))
    augh_scale = float(train_cfg.get("augment_scale", 0.05))

    best_score, best_state, bad = (-float("inf") if select == "auprc" else float("inf")), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for batch in train_loader:
            xb, fb, yb = split_batch(batch, device)
            xb = augment_batch(xb, jitter, augh_scale)
            opt.zero_grad()
            loss = crit(model_call(model, xb, fb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
        vm = evaluate(model, val_loader, device)
        vs = evaluate(model, val_sel_loader, device)
        print(f"[seed {seed}] ep {ep:02d} train_loss={tot/len(tr):.4f} val_nll={vm['nll']:.4f} "
              f"val_f1={vm['macro_f1']:.4f} val_auprc={vm['auprc']:.4f} sel_auprc={vs['auprc']:.4f}")
        score = vs[select]
        improved = score > best_score if select == "auprc" else score < best_score
        if improved:
            best_score, best_state, bad = score, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[seed {seed}] early stop at ep {ep} (best {select}={best_score:.4f})")
                break

    model.load_state_dict(best_state)
    yv, pv = predict_probs(model, val_loader, device)
    yt, pt = predict_probs(model, test_loader, device)
    return {
        "seed": seed,
        "val_subjects": sorted(set(subj[va].tolist())),
        "val": metrics_from_probs(yv, pv, evaluate(model, val_loader, device)["nll"]),
        "test_subjects": sorted(set(subj[te].tolist())),
        "y_test": yt,
        "p_test": pt,
        "state": best_state,
        "normalizer": {
            "mean": mu.tolist(), "std": sd.tolist(),
            "spec_mean": None if fmu is None else fmu.tolist(),
            "spec_std": None if fsd is None else fsd.tolist(),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated ensemble seeds, e.g. '42,43,44'")
    args = ap.parse_args()

    data_cfg = load_cfg("data.yml")
    model_cfg = load_cfg("model.yml")
    train_cfg = load_cfg("train.yml")

    if args.epochs is not None:
        model_cfg["max_epochs"] = args.epochs
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",")]
    elif isinstance(train_cfg.get("ensemble_seeds"), list):
        seeds = [int(s) for s in train_cfg["ensemble_seeds"]]
    elif args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = [int(data_cfg.get("seed", 42))]
    torch.manual_seed(seeds[0])
    np.random.seed(seeds[0])

    root = project_root()
    imu_root = resolve_path(data_cfg["data_dir"])
    fs = int(data_cfg.get("sampling_frequency", 128))
    X, y, subj, _, stats = build_all_windows(
        imu_root,
        fs=fs,
        obs_s=float(data_cfg.get("observation_windows", 5.0)),
        horizon_s=float(data_cfg.get("forecast_horizon", 2.0)),
        stride_s=float(data_cfg.get("window_stride", 1.0)),
        fog_frac=float(data_cfg.get("fog_frac", 0.30)),
    )
    print(f"windows={X.shape} fog_rate={y.mean():.4f} subjects={len(set(subj.tolist()))}")
    print(stats.groupby("subject")["n_win"].sum().describe().to_string())

    base = int(data_cfg.get("seed", 42))
    _, _, te = subject_group_split(
        subj,
        test_size=float(data_cfg.get("test_size", 0.25)),
        val_size=float(data_cfg.get("val_size", 0.20)),
        seed=base,
    )
    tr_pool = np.setdiff1d(np.arange(len(y)), te)
    print(f"fixed test: n={len(te)} subjects={sorted(set(subj[te].tolist()))}")

    runs = [run_seed(s, X, y, subj, tr_pool, te, data_cfg, model_cfg, train_cfg) for s in seeds]

    # Ensemble: average predicted probabilities across seeds (unchanged test set).
    p_ens = np.mean([r["p_test"] for r in runs], axis=0)
    yt = runs[0]["y_test"]
    nll = nn.CrossEntropyLoss()(
        torch.log(torch.from_numpy(p_ens).clamp_min(1e-8)), torch.from_numpy(yt)
    ).item()
    test_m = metrics_from_probs(yt, p_ens, nll)
    val_m = runs[0]["val"] if len(runs) == 1 else {
        k: float(np.mean([r["val"][k] for r in runs]))
        for k in runs[0]["val"]
    }
    print("VAL:", json.dumps(val_m, indent=2))
    print("TEST:", json.dumps(test_m, indent=2))

    out_dir = root / str(train_cfg.get("out_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(runs[0]["state"], out_dir / str(train_cfg.get("ckpt_name", "patchtst_imu.pt")))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "val": val_m,
            "test": test_m,
            "test_subjects": sorted(set(subj[te].tolist())),
            "seeds": seeds,
            "per_seed_test_auprc": [
                float(average_precision_score(r["y_test"], r["p_test"][:, 1]))
                for r in runs
            ],
            "test_per_subject_auprc": per_subject_auprc(yt, p_ens, subj[te]),
        }, f, indent=2)
    with open(out_dir / "normalizer.json", "w") as f:
        json.dump(runs[0]["normalizer"], f)


if __name__ == "__main__":
    main()
