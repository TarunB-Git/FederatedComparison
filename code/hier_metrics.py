#!/usr/bin/env python3
"""Evaluation, comprehensive metrics, plotting, and cross-run CSV utilities
for the hierarchical next-action prediction pipeline."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F
from torch.utils.data import DataLoader

_TMP_CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")) / "hier_cache"
_TMP_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_CACHE_ROOT / "matplotlib"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out) or np.isinf(out):
        return None
    return out


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path, title: str = "Confusion Matrix"):
    """Plot a normalized confusion matrix heatmap."""
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)*0.5), max(6, len(class_names)*0.4)))
    
    # Normalize by row (recall)
    row_sums = cm.sum(axis=1)
    cm_norm = np.divide(cm.astype('float'), row_sums[:, np.newaxis], 
                        out=np.zeros_like(cm, dtype=float), 
                        where=row_sums[:, np.newaxis] > 0)
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    # Text annotations
    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm_norm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")
    
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def state_dict_num_bytes(state: dict[str, torch.Tensor]) -> int:
    """Estimate bytes required to transmit a state dict tensor payload."""
    total = 0
    for tensor in state.values():
        if not isinstance(tensor, torch.Tensor):
            continue
        total += int(tensor.numel()) * int(tensor.element_size())
    return int(total)


def _normalized_entropy(counts: np.ndarray) -> float | None:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return None
    probs = counts[counts > 0] / total
    if probs.size <= 1:
        return 0.0
    entropy = float(-(probs * np.log2(probs)).sum())
    max_entropy = float(np.log2(probs.size))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def compute_confusion_metrics(cm: np.ndarray) -> dict[str, float | int | None]:
    """Compute comparable classification metrics from a confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    total = float(cm.sum())
    if total <= 0:
        return {
            "top1": None,
            "f1_macro": None,
            "f1_weighted": None,
            "balanced_accuracy": None,
            "precision_macro": None,
            "recall_macro": None,
            "precision_weighted": None,
            "recall_weighted": None,
            "kappa": None,
            "mcc": None,
            "majority_baseline_top1": None,
            "majority_class_id": -1,
            "majority_class_share": None,
            "active_classes": 0,
            "support_entropy": None,
        }

    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    active_true = support > 0
    active_union = (support + predicted) > 0

    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    weights = support / total

    top1 = float(tp.sum() / total)
    balanced_accuracy = float(recall[active_true].mean()) if bool(active_true.any()) else None
    precision_macro = float(precision[active_union].mean()) if bool(active_union.any()) else None
    recall_macro = float(recall[active_union].mean()) if bool(active_union.any()) else None
    f1_macro = float(f1[active_union].mean()) if bool(active_union.any()) else None
    precision_weighted = float((precision * weights).sum())
    recall_weighted = float((recall * weights).sum())
    f1_weighted = float((f1 * weights).sum())

    expected = float(np.dot(support, predicted) / (total * total))
    denom = 1.0 - expected
    kappa = float((top1 - expected) / denom) if abs(denom) > 1e-12 else None

    cov_ytyp = float(tp.sum() * total - np.dot(support, predicted))
    cov_ypyp = float(total * total - np.dot(predicted, predicted))
    cov_ytyt = float(total * total - np.dot(support, support))
    mcc_denom = float(np.sqrt(max(cov_ypyp, 0.0) * max(cov_ytyt, 0.0)))
    mcc = float(cov_ytyp / mcc_denom) if mcc_denom > 1e-12 else None

    majority_class_id = int(np.argmax(support)) if support.size else -1
    majority_share = float(support[majority_class_id] / total) if support.size else None

    return {
        "top1": top1,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "kappa": kappa,
        "mcc": mcc,
        "majority_baseline_top1": majority_share,
        "majority_class_id": majority_class_id,
        "majority_class_share": majority_share,
        "active_classes": int(active_true.sum()),
        "support_entropy": _normalized_entropy(support),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_hier(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    num_exact_classes: int,
    num_families: int,
    family_fine_to_exact_id: dict[int, dict[int, int]],
    default_exact_id: int,
    coarse_loss_weight: float,
    fine_loss_weight: float,
    max_batches: int = 0,
) -> dict:
    """Evaluate hierarchical model and return comprehensive metrics dict.

    Returns a dict with keys for coarse, fine, exact metrics, per-family
    breakdowns, and confusion matrices stored as numpy arrays under
    special keys `_cm_coarse` and `_cm_exact`.
    """
    model.eval()

    true_coarse_all, pred_coarse_all, coarse_topk_all = [], [], []
    true_fine_all, pred_fine_true_all, pred_fine_pred_all = [], [], []
    true_exact_all, pred_exact_all = [], []
    pred_exact_direct_all: list[np.ndarray] = []   # direct exact-head predictions
    has_exact_head = False

    loss_coarse_sum = 0.0
    loss_fine_sum = 0.0
    loss_total_sum = 0.0
    total = 0

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if max_batches > 0 and bidx >= max_batches:
                break

            if len(batch) >= 6:
                x, y_coarse, y_fine, y_exact, lengths, _player_races = batch
            else:
                x, y_coarse, y_fine, y_exact, lengths = batch

            x = x.to(device)
            y_coarse = y_coarse.to(device)
            y_fine = y_fine.to(device)
            y_exact = y_exact.to(device)
            lengths = lengths.to(device)

            model_out = model(x, lengths)
            exact_logits = None
            if isinstance(model_out, tuple) and len(model_out) == 3:
                coarse_logits, fine_logits, exact_logits = model_out
                has_exact_head = True
            else:
                coarse_logits, fine_logits = model_out

            # losses
            ce_coarse = F.cross_entropy(coarse_logits, y_coarse, reduction="sum")
            ce_fine = torch.tensor(0.0, device=device)
            for fam_id, fam_logits in enumerate(fine_logits):
                mask = y_coarse == fam_id
                if bool(mask.any()):
                    ce_fine = ce_fine + F.cross_entropy(
                        fam_logits[mask], y_fine[mask], reduction="sum",
                    )

            n = int(y_exact.numel())
            total += n
            loss_coarse_sum += float(ce_coarse.item())
            loss_fine_sum += float(ce_fine.item())
            loss_total_sum += float(
                (coarse_loss_weight * ce_coarse + fine_loss_weight * ce_fine).item()
            )

            # predictions
            coarse_pred = coarse_logits.argmax(dim=1)
            topk = torch.topk(
                coarse_logits, k=min(5, num_families), dim=1,
            ).indices

            pred_fine_true = torch.zeros_like(y_fine)
            pred_fine_pred = torch.zeros_like(y_fine)

            for fam_id, fam_logits in enumerate(fine_logits):
                mask_true = y_coarse == fam_id
                if bool(mask_true.any()):
                    pred_fine_true[mask_true] = fam_logits[mask_true].argmax(dim=1)
                mask_pred = coarse_pred == fam_id
                if bool(mask_pred.any()):
                    pred_fine_pred[mask_pred] = fam_logits[mask_pred].argmax(dim=1)

            # exact prediction via coarse→fine→exact mapping (hierarchical path)
            pred_exact = torch.zeros_like(y_exact)
            for i in range(n):
                fam_id = int(coarse_pred[i].item())
                fine_id = int(pred_fine_pred[i].item())
                exact_id = family_fine_to_exact_id.get(fam_id, {}).get(
                    fine_id, default_exact_id,
                )
                if exact_id < 0 or exact_id >= num_exact_classes:
                    exact_id = default_exact_id
                pred_exact[i] = int(exact_id)

            # direct exact prediction via auxiliary exact head (bypasses hierarchy)
            if exact_logits is not None:
                pred_exact_direct_all.append(exact_logits.argmax(dim=1).cpu().numpy())

            true_coarse_all.append(y_coarse.cpu().numpy())
            pred_coarse_all.append(coarse_pred.cpu().numpy())
            coarse_topk_all.append(topk.cpu().numpy())
            true_fine_all.append(y_fine.cpu().numpy())
            pred_fine_true_all.append(pred_fine_true.cpu().numpy())
            pred_fine_pred_all.append(pred_fine_pred.cpu().numpy())
            true_exact_all.append(y_exact.cpu().numpy())
            pred_exact_all.append(pred_exact.cpu().numpy())

    if total == 0:
        return _empty_metrics(num_families, num_exact_classes)

    true_coarse = np.concatenate(true_coarse_all)
    pred_coarse = np.concatenate(pred_coarse_all)
    coarse_topk = np.concatenate(coarse_topk_all)
    true_fine = np.concatenate(true_fine_all)
    pred_fine_true = np.concatenate(pred_fine_true_all)
    pred_fine_pred = np.concatenate(pred_fine_pred_all)
    true_exact = np.concatenate(true_exact_all)
    pred_exact = np.concatenate(pred_exact_all)

    # ── Coarse metrics ───────────────────────────────────────────────────
    coarse_top1 = float((pred_coarse == true_coarse).mean())
    coarse_top5 = float(np.mean([
        int(t in row) for t, row in zip(true_coarse.tolist(), coarse_topk.tolist())
    ]))
    # ── Fine metrics ─────────────────────────────────────────────────────
    fine_top1_true = float((pred_fine_true == true_fine).mean())
    mask_correct_coarse = pred_coarse == true_coarse
    fine_top1_cond = (
        float((pred_fine_pred[mask_correct_coarse] == true_fine[mask_correct_coarse]).mean())
        if bool(mask_correct_coarse.any()) else None
    )

    # ── Exact metrics (hierarchical path) ──────────────────────────────
    # confusion matrices
    cm_coarse = confusion_matrix(true_coarse, pred_coarse, labels=np.arange(num_families))
    cm_exact = confusion_matrix(true_exact, pred_exact, labels=np.arange(num_exact_classes))
    coarse_stats = compute_confusion_metrics(cm_coarse)
    exact_stats = compute_confusion_metrics(cm_exact)

    dominant_id = int(exact_stats["majority_class_id"] or -1)
    dom_mask = true_exact != dominant_id
    exact_top1_ex_dominant = (
        float((pred_exact[dom_mask] == true_exact[dom_mask]).mean())
        if bool(dom_mask.any()) else None
    )

    # ── Direct exact metrics (auxiliary head, bypasses hierarchy) ─────
    direct_exact_stats: dict[str, float | int | None] = {}
    if has_exact_head and pred_exact_direct_all:
        pred_exact_direct = np.concatenate(pred_exact_direct_all)
        cm_exact_direct = confusion_matrix(
            true_exact, pred_exact_direct, labels=np.arange(num_exact_classes),
        )
        direct_exact_stats = compute_confusion_metrics(cm_exact_direct)
    else:
        direct_exact_stats = {
            "top1": None, "f1_macro": None, "balanced_accuracy": None,
            "kappa": None, "mcc": None,
        }

    # per-family breakdown
    per_family = []
    for fam_id in range(num_families):
        mask = true_coarse == fam_id
        support = int(mask.sum())
        if support > 0:
            per_family.append({
                "family_id": fam_id,
                "support": support,
                "coarse_recall": float((pred_coarse[mask] == fam_id).mean()),
                "fine_recall_true_family": float((pred_fine_true[mask] == true_fine[mask]).mean()),
                "exact_top1_within_family": float((pred_exact[mask] == true_exact[mask]).mean()),
            })
        else:
            per_family.append({
                "family_id": fam_id, "support": 0,
                "coarse_recall": None, "fine_recall_true_family": None,
                "exact_top1_within_family": None,
            })

    return {
        "n_examples": int(total),
        "loss": float(loss_total_sum / max(1, total)),
        "coarse_loss": float(loss_coarse_sum / max(1, total)),
        "fine_loss": float(loss_fine_sum / max(1, total)),
        # coarse
        "coarse_top1": coarse_top1,
        "coarse_top5": coarse_top5,
        "coarse_f1_macro": coarse_stats["f1_macro"],
        "coarse_balanced_accuracy": coarse_stats["balanced_accuracy"],
        "coarse_precision_macro": coarse_stats["precision_macro"],
        "coarse_recall_macro": coarse_stats["recall_macro"],
        "coarse_precision_weighted": coarse_stats["precision_weighted"],
        "coarse_recall_weighted": coarse_stats["recall_weighted"],
        "coarse_majority_baseline_top1": coarse_stats["majority_baseline_top1"],
        "coarse_majority_class_id": coarse_stats["majority_class_id"],
        "coarse_majority_class_share": coarse_stats["majority_class_share"],
        "coarse_active_classes": coarse_stats["active_classes"],
        "coarse_support_entropy": coarse_stats["support_entropy"],
        # fine
        "fine_top1_true_family": fine_top1_true,
        "fine_top1_conditional_correct_coarse": fine_top1_cond,
        # exact (hierarchical path: coarse → fine → exact lookup)
        "exact_top1": exact_stats["top1"],
        "exact_f1_macro": exact_stats["f1_macro"],
        "exact_f1_weighted": exact_stats["f1_weighted"],
        "exact_balanced_accuracy": exact_stats["balanced_accuracy"],
        "exact_precision_macro": exact_stats["precision_macro"],
        "exact_recall_macro": exact_stats["recall_macro"],
        "exact_precision_weighted": exact_stats["precision_weighted"],
        "exact_recall_weighted": exact_stats["recall_weighted"],
        "exact_kappa": exact_stats["kappa"],
        "exact_mcc": exact_stats["mcc"],
        "exact_top1_ex_dominant": exact_top1_ex_dominant,
        "dominant_class_id": dominant_id,
        "exact_majority_baseline_top1": exact_stats["majority_baseline_top1"],
        "exact_majority_class_share": exact_stats["majority_class_share"],
        "exact_active_classes": exact_stats["active_classes"],
        "exact_support_entropy": exact_stats["support_entropy"],
        "majority_baseline": exact_stats["majority_baseline_top1"],
        # exact (direct auxiliary head — bypasses hierarchical bottleneck)
        "exact_top1_direct": direct_exact_stats.get("top1"),
        "exact_f1_macro_direct": direct_exact_stats.get("f1_macro"),
        "exact_balanced_accuracy_direct": direct_exact_stats.get("balanced_accuracy"),
        "exact_kappa_direct": direct_exact_stats.get("kappa"),
        "exact_mcc_direct": direct_exact_stats.get("mcc"),
        # internal
        "_cm_coarse": cm_coarse,
        "_cm_exact": cm_exact,
        "_per_family": per_family,
    }


def _empty_metrics(num_families: int, num_exact_classes: int) -> dict:
    return {
        "n_examples": 0, "loss": None,
        "coarse_loss": None, "fine_loss": None,
        "coarse_top1": None, "coarse_top5": None,
        "coarse_f1_macro": None, "coarse_balanced_accuracy": None,
        "coarse_precision_macro": None, "coarse_recall_macro": None,
        "coarse_precision_weighted": None, "coarse_recall_weighted": None,
        "coarse_majority_baseline_top1": None, "coarse_majority_class_id": -1,
        "coarse_majority_class_share": None, "coarse_active_classes": 0,
        "coarse_support_entropy": None,
        "fine_top1_true_family": None, "fine_top1_conditional_correct_coarse": None,
        "exact_top1": None, "exact_f1_macro": None, "exact_f1_weighted": None,
        "exact_balanced_accuracy": None,
        "exact_precision_macro": None, "exact_recall_macro": None,
        "exact_precision_weighted": None, "exact_recall_weighted": None,
        "exact_kappa": None, "exact_mcc": None,
        "exact_top1_ex_dominant": None, "dominant_class_id": -1,
        "exact_majority_baseline_top1": None, "exact_majority_class_share": None,
        "exact_active_classes": 0, "exact_support_entropy": None,
        "majority_baseline": None,
        "exact_top1_direct": None, "exact_f1_macro_direct": None,
        "exact_balanced_accuracy_direct": None,
        "exact_kappa_direct": None, "exact_mcc_direct": None,
        "_cm_coarse": np.zeros((num_families, num_families), dtype=np.int64),
        "_cm_exact": np.zeros((num_exact_classes, num_exact_classes), dtype=np.int64),
        "_per_family": [],
    }


# ---------------------------------------------------------------------------
# Metric comparison helper
# ---------------------------------------------------------------------------

SELECTION_OBJECTIVES: dict[str, dict[str, object]] = {
    "coarse_honest": {
        "primary": "coarse_top1",
        "tiebreakers": ["coarse_balanced_accuracy", "coarse_f1_macro", "exact_top1"],
    },
    "exact_honest": {
        "primary": "exact_top1",
        "tiebreakers": ["exact_f1_macro", "exact_balanced_accuracy", "coarse_top1"],
    },
    "joint_honest": {
        "primary": "exact_top1_direct",
        "tiebreakers": ["coarse_top1", "exact_top1", "exact_f1_macro_direct"],
    },
}

def is_better(
    new_m: dict, best_m: dict | None,
    primary: str = "coarse_top1",
    tiebreakers: list[str] | None = None,
) -> bool:
    if best_m is None:
        return True
    keys = [primary] + (tiebreakers or [])
    for k in keys:
        n = float(new_m.get(k) or -1e9)
        b = float(best_m.get(k) or -1e9)
        if n > b + 1e-12:
            return True
        if n < b - 1e-12:
            return False
    return False


def parse_metric_list(spec: str | None, *, default: list[str]) -> list[str]:
    """Parse a comma-separated metric-key list.

    Empty items are ignored. If parsing yields nothing, returns ``default``.
    """
    if spec is None:
        return list(default)
    items = [x.strip() for x in str(spec).split(",")]
    out = [x for x in items if x]
    return out if out else list(default)


def resolve_selection_objective(
    objective_or_primary: str | None,
    tiebreakers_spec: str | None = None,
) -> tuple[str, list[str]]:
    """Resolve a selection objective preset or raw metric key.

    If ``objective_or_primary`` matches a preset in ``SELECTION_OBJECTIVES``, the
    preset primary/tiebreakers are used (with optional tiebreaker override from
    ``tiebreakers_spec``). Otherwise, the value is treated as a raw primary
    metric key.
    """
    name = str(objective_or_primary or "").strip().lower()
    if not name:
        name = str(SELECTION_OBJECTIVES["coarse_honest"]["primary"])

    if name in SELECTION_OBJECTIVES:
        preset = SELECTION_OBJECTIVES[name]
        primary = str(preset.get("primary", "coarse_top1"))
        default_tiebreakers = [
            str(x) for x in list(preset.get("tiebreakers", []))
        ]
        if tiebreakers_spec is None:
            return primary, default_tiebreakers
        return primary, parse_metric_list(tiebreakers_spec, default=default_tiebreakers)

    # Raw metric key path
    if tiebreakers_spec is None:
        return name, []
    return name, parse_metric_list(tiebreakers_spec, default=[])


# ---------------------------------------------------------------------------
# JSON-safe metrics (strip numpy arrays)
# ---------------------------------------------------------------------------

def metrics_for_json(m: dict) -> dict:
    """Return a copy of metrics dict with numpy arrays removed."""
    return {k: v for k, v in m.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_confusion_csv(cm: np.ndarray, path: Path) -> None:
    idx = list(range(cm.shape[0]))
    pd.DataFrame(cm, index=idx, columns=idx).to_csv(path, index=True)


def save_family_summary(
    per_family: list[dict],
    family_id_to_name: dict[int, str],
    path: Path,
) -> None:
    rows = []
    for r in per_family:
        x = dict(r)
        x["family_name"] = family_id_to_name.get(int(r["family_id"]), str(r["family_id"]))
        rows.append(x)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_per_class_recall(cm: np.ndarray, path: Path, class_prefix: str) -> None:
    support = cm.sum(axis=1)
    recall = np.divide(
        np.diag(cm),
        support,
        out=np.zeros(cm.shape[0], dtype=np.float64),
        where=support > 0,
    )
    pd.DataFrame(
        {
            f"{class_prefix}_id": np.arange(cm.shape[0], dtype=np.int64),
            "support": support.astype(np.int64),
            "recall": recall,
        }
    ).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Cross-run CSV (append-mode)
# ---------------------------------------------------------------------------

CROSS_RUN_COLS = [
    "timestamp", "mode", "dataset_dir",
    "n_train", "n_val", "n_test", "n_clients",
    "epochs_or_rounds", "hidden", "layers", "dropout", "window",
    "architecture", "race", "taxonomy", "split_mode", "effective_split_unit",
    "selection_objective",
    "coarse_top1", "coarse_top5", "coarse_f1_macro", "coarse_balanced_accuracy",
    "coarse_precision_macro", "coarse_recall_macro",
    "coarse_precision_weighted", "coarse_recall_weighted",
    "coarse_majority_baseline_top1", "coarse_majority_class_share",
    "coarse_active_classes", "coarse_support_entropy",
    "fine_top1_true_family", "fine_top1_conditional_correct_coarse",
    "exact_top1", "exact_f1_macro", "exact_f1_weighted",
    "exact_balanced_accuracy", "exact_precision_macro", "exact_recall_macro",
    "exact_precision_weighted", "exact_recall_weighted",
    "exact_kappa", "exact_mcc", "exact_top1_ex_dominant",
    "exact_majority_baseline_top1", "exact_majority_class_share",
    "exact_active_classes", "exact_support_entropy", "majority_baseline",
    "exact_top1_direct", "exact_f1_macro_direct",
    "exact_balanced_accuracy_direct", "exact_kappa_direct", "exact_mcc_direct",
    "train_wall_clock_sec", "validation_wall_clock_sec", "test_wall_clock_sec", "total_wall_clock_sec",
    "avg_epoch_time_sec", "avg_round_time_sec", "avg_client_train_time_sec",
    "train_examples_per_sec", "val_examples_per_sec", "test_examples_per_sec",
    "round_examples_per_sec", "client_updates_per_sec",
    "comm_bytes_down_total", "comm_bytes_up_total", "comm_bytes_total",
    "comm_bytes_per_round_avg", "comm_bytes_per_client_avg",
    "best_epoch_or_round", "outdir",
]


def append_cross_run_row(
    csv_path: Path,
    *,
    mode: str,
    dataset_dir: str,
    n_train: int,
    n_val: int,
    n_test: int,
    n_clients: int,
    epochs_or_rounds: int,
    hidden: int,
    layers: int,
    dropout: float,
    window: int,
    metrics: dict,
    best_epoch_or_round: int | None,
    outdir: str,
) -> None:
    """Append a single row to the cross-run comparison CSV."""
    exists = csv_path.exists()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "dataset_dir": dataset_dir,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_clients": n_clients,
        "epochs_or_rounds": epochs_or_rounds,
        "hidden": hidden,
        "layers": layers,
        "dropout": dropout,
        "window": window,
        "best_epoch_or_round": best_epoch_or_round,
        "outdir": outdir,
    }
    # copy metric values
    for col in CROSS_RUN_COLS:
        if col not in row:
            row[col] = metrics.get(col, None)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CROSS_RUN_COLS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(history_df: pd.DataFrame, outdir: Path) -> None:
    if plt is None or history_df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0, 0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(history_df["epoch"], history_df["val_coarse_top1"], label="coarse_top1")
    if "val_fine_top1_true_family" in history_df.columns:
        axes[0, 1].plot(history_df["epoch"], history_df["val_fine_top1_true_family"], label="fine_top1")
    axes[0, 1].set_title("Hierarchy Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(history_df["epoch"], history_df["val_exact_top1"], label="exact_top1")
    axes[1, 0].plot(history_df["epoch"], history_df["val_exact_f1_macro"], label="macro_f1")
    axes[1, 0].set_title("Exact Metrics")
    axes[1, 0].legend()

    axes[1, 1].plot(history_df["epoch"], history_df["val_exact_balanced_accuracy"], label="bal_acc")
    if "val_exact_kappa" in history_df.columns:
        axes[1, 1].plot(history_df["epoch"], history_df["val_exact_kappa"], label="kappa")
    axes[1, 1].set_title("Balanced Accuracy & Kappa")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "training_curves.png", dpi=140)
    plt.close(fig)


def plot_round_curves(round_df: pd.DataFrame, outdir: Path) -> None:
    if plt is None or round_df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(round_df["round"], round_df["client_loss"], label="client_loss")
    axes[0, 0].set_title("Avg Local Client Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(round_df["round"], round_df["val_coarse_top1"], label="coarse_top1")
    axes[0, 1].plot(round_df["round"], round_df["val_exact_top1"], label="exact_top1")
    axes[0, 1].set_title("Validation Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(round_df["round"], round_df["val_exact_f1_macro"], label="macro_f1")
    axes[1, 0].set_title("Validation Macro-F1")
    axes[1, 0].legend()

    axes[1, 1].plot(round_df["round"], round_df["val_exact_balanced_accuracy"], label="bal_acc")
    if "val_exact_kappa" in round_df.columns:
        axes[1, 1].plot(round_df["round"], round_df["val_exact_kappa"], label="kappa")
    axes[1, 1].set_title("Balanced Accuracy & Kappa")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "round_curves.png", dpi=140)
    plt.close(fig)


def plot_metrics_curves(metrics_df: pd.DataFrame, outdir: Path, prefix: str = "val_") -> None:
    """Plot comprehensive metrics curves from training history or FL rounds.
    
    Args:
        metrics_df: DataFrame with metrics columns
        outdir: Directory to save plots
        prefix: Column prefix to plot (e.g., "val_" for validation, "test_" for test)
    """
    if plt is None or metrics_df.empty:
        return
    
    # Ensure outdir exists
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Determine x-axis column (round or epoch)
    x_col = None
    for col in ["round", "epoch"]:
        if col in metrics_df.columns:
            x_col = col
            break
    
    if x_col is None:
        return
    
    x = metrics_df[x_col]
    
    # 1. Loss and coarse loss
    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        if f"{prefix}loss" in metrics_df.columns:
            ax.plot(x, metrics_df[f"{prefix}loss"], label="loss", marker="o")
        if f"{prefix}coarse_loss" in metrics_df.columns:
            ax.plot(x, metrics_df[f"{prefix}coarse_loss"], label="coarse_loss", marker="s")
        if f"{prefix}fine_loss" in metrics_df.columns:
            ax.plot(x, metrics_df[f"{prefix}fine_loss"], label="fine_loss", marker="^")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"{prefix}loss_curves.png", dpi=140)
        plt.close(fig)
    
    # 2. Coarse accuracy (Top-1 and Top-5)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        if f"{prefix}coarse_top1" in metrics_df.columns:
            ax.plot(x, metrics_df[f"{prefix}coarse_top1"], label="coarse_top1", marker="o")
        if f"{prefix}coarse_top5" in metrics_df.columns:
            ax.plot(x, metrics_df[f"{prefix}coarse_top5"], label="coarse_top5", marker="s")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Accuracy")
        ax.set_title("Coarse Action Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"{prefix}coarse_accuracy.png", dpi=140)
        plt.close(fig)
    
    # 3. Exact action accuracy
    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        if f"{prefix}exact_top1" in metrics_df.columns:
            ax.plot(x, metrics_df[f"{prefix}exact_top1"], label="exact_top1", marker="o")
        if f"{prefix}exact_top5" in metrics_df.columns:
            ax.plot(x, metrics_df[f"{prefix}exact_top5"], label="exact_top5", marker="s")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Accuracy")
        ax.set_title("Exact Action Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"{prefix}exact_accuracy.png", dpi=140)
        plt.close(fig)
    
    # 4. Macro-F1 and balanced accuracy
    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        if f"{prefix}coarse_f1_macro" in metrics_df.columns:
            ax.plot(x, metrics_df[f"{prefix}coarse_f1_macro"], label="coarse_f1_macro", marker="o")
        if f"{prefix}coarse_balanced_accuracy" in metrics_df.columns:
            ax.plot(x, metrics_df[f"{prefix}coarse_balanced_accuracy"], label="coarse_bal_acc", marker="s")
        if f"{prefix}exact_f1_macro" in metrics_df.columns:
            ax.plot(x, metrics_df[f"{prefix}exact_f1_macro"], label="exact_f1_macro", marker="^")
        if f"{prefix}exact_balanced_accuracy" in metrics_df.columns:
            ax.plot(x, metrics_df[f"{prefix}exact_balanced_accuracy"], label="exact_bal_acc", marker="d")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Score")
        ax.set_title("Macro-F1 and Balanced Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"{prefix}f1_balanced_accuracy.png", dpi=140)
        plt.close(fig)
    
    # 5. Per-race metrics (if available)
    race_metrics = [col for col in metrics_df.columns if "race_" in col and prefix in col]
    if race_metrics and plt is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in race_metrics:
            if not metrics_df[col].isna().all():
                ax.plot(x, metrics_df[col], label=col.replace(f"{prefix}", ""), marker="o")
        if ax.get_legend() is not None:
            ax.set_xlabel(x_col)
            ax.set_ylabel("Metric")
            ax.set_title("Per-Race Metrics")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / f"{prefix}race_metrics.png", dpi=140)
        plt.close(fig)
