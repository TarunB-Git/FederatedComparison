#!/usr/bin/env python3
"""Personalized backbone-head FL with race-specific heads for hierarchical GRU.

This implements a FedBSD-style setup with race-specific output heads:
  - Each client keeps private race-specific heads
  - Only backbone parameters are communicated and aggregated
  - Samples are routed to the correct race head based on player_race
  - Default: legacy8 (8-class) coarse taxonomy
  - Supports legacy8, broad5 taxonomies for comparison
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hier_common import (
    add_action_context_features,
    apply_pretrain_coarse_audit,
    assert_hierarchy_targets_usable,
    assert_runtime_coarse_not_inflated,
    build_class_weights,
    build_event_table,
    build_family_maps,
    build_race_maps,
    collate_hier_batch,
    load_and_normalize_events,
    make_loaders,
    prepare_hierarchy_targets,
    resolve_dataset_paths,
)
from hier_metrics import (
    append_cross_run_row,
    compute_confusion_metrics,
    evaluate_hier,
    is_better,
    metrics_for_json,
    parse_metric_list,
    plot_round_curves,
    plot_metrics_curves,
    plot_confusion_matrix,
    resolve_selection_objective,
    save_confusion_csv,
    save_family_summary,
    state_dict_num_bytes,
)
from hier_models import (
    HierGRU,
    HierGRURaceHeads,
    aggregate_state_dicts,
)


def _clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _clone_head_bank(
    head_bank: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {cid: _clone_state_dict(head) for cid, head in head_bank.items()}


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """Save a checkpoint without leaving a half-written target on interruption."""
    tmp_path = path.with_name(path.name + ".tmp")
    prev_path = path.with_name(path.name + ".prev")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(payload, tmp_path)
    if path.exists():
        path.replace(prev_path)
    tmp_path.replace(path)


def _safe_torch_load(path: Path) -> dict | None:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[resume] could not load {path}: {exc}", flush=True)
        return None


def _metrics_from_round_row(row: dict) -> dict:
    metrics: dict = {}
    for key, value in row.items():
        if not str(key).startswith("val_"):
            continue
        metric_key = str(key)[4:]
        if pd.isna(value):
            metrics[metric_key] = None
        else:
            try:
                metrics[metric_key] = float(value)
            except Exception:
                metrics[metric_key] = value
    return metrics


def _advance_round_rng(
    rng: np.random.Generator,
    *,
    completed_rounds: int,
    eligible_clients: list[str],
    clients_per_round: int,
    val_client_loaders: dict[str, DataLoader],
    round_val_clients: int,
) -> None:
    """Advance sampling RNG so resumed runs match uninterrupted runs."""
    sample_k = min(int(clients_per_round), len(eligible_clients))
    val_keys = sorted(list(val_client_loaders.keys()))
    sample_val = int(round_val_clients) > 0 and len(val_keys) > int(round_val_clients)
    for _ in range(max(0, int(completed_rounds))):
        rng.choice(eligible_clients, size=sample_k, replace=False)
        if sample_val:
            rng.choice(val_keys, size=int(round_val_clients), replace=False)


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad = enabled


def _weighted_metric_mean(entries: list[tuple[int, dict, bool]], key: str) -> float | None:
    num = 0.0
    den = 0
    for n, m, _has_personalized in entries:
        value = m.get(key)
        if value is None:
            continue
        num += float(value) * int(n)
        den += int(n)
    if den <= 0:
        return None
    return float(num / den)


def _aggregate_personalized_metrics(
    entries: list[tuple[int, dict, bool]],
    *,
    num_families: int,
    num_exact_classes: int,
) -> dict:
    total_examples = int(sum(int(n) for n, _m, _has in entries))
    n_clients = int(len(entries))
    n_clients_with_head = int(sum(1 for _n, _m, has in entries if has))
    examples_with_head = int(sum(int(n) for n, _m, has in entries if has))

    cm_coarse = np.zeros((num_families, num_families), dtype=np.int64)
    cm_exact = np.zeros((num_exact_classes, num_exact_classes), dtype=np.int64)
    per_family_all: list[dict] = []

    for _n, m, _has in entries:
        cm_coarse = cm_coarse + m.get("_cm_coarse", 0)
        cm_exact = cm_exact + m.get("_cm_exact", 0)
        per_family_all.extend(list(m.get("_per_family", [])))

    coarse_stats = compute_confusion_metrics(cm_coarse)
    exact_stats = compute_confusion_metrics(cm_exact)
    dominant_class_id = int(exact_stats.get("majority_class_id", -1) or -1)
    exact_support = cm_exact.sum(axis=1)
    exact_correct = np.diag(cm_exact)
    keep_mask = np.ones(cm_exact.shape[0], dtype=bool)
    if 0 <= dominant_class_id < keep_mask.size:
        keep_mask[dominant_class_id] = False
    remaining = int(exact_support[keep_mask].sum())
    exact_top1_ex_dominant = (
        float(exact_correct[keep_mask].sum() / remaining)
        if remaining > 0 else None
    )

    return {
        "n_examples": total_examples,
        "loss": _weighted_metric_mean(entries, "loss"),
        "coarse_loss": _weighted_metric_mean(entries, "coarse_loss"),
        "fine_loss": _weighted_metric_mean(entries, "fine_loss"),
        "coarse_top1": coarse_stats["top1"],
        "coarse_top5": _weighted_metric_mean(entries, "coarse_top5"),
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
        "fine_top1_true_family": _weighted_metric_mean(entries, "fine_top1_true_family"),
        "fine_top1_conditional_correct_coarse": _weighted_metric_mean(
            entries,
            "fine_top1_conditional_correct_coarse",
        ),
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
        "dominant_class_id": dominant_class_id,
        "exact_majority_baseline_top1": exact_stats["majority_baseline_top1"],
        "exact_majority_class_share": exact_stats["majority_class_share"],
        "exact_active_classes": exact_stats["active_classes"],
        "exact_support_entropy": exact_stats["support_entropy"],
        "majority_baseline": exact_stats["majority_baseline_top1"],
        "n_eval_clients": n_clients,
        "n_eval_clients_with_personalized_head": n_clients_with_head,
        "personalized_head_coverage_clients": (
            float(n_clients_with_head / n_clients) if n_clients > 0 else None
        ),
        "personalized_head_coverage_examples": (
            float(examples_with_head / total_examples) if total_examples > 0 else None
        ),
        "exact_top1_direct": _weighted_metric_mean(entries, "exact_top1_direct"),
        "exact_f1_macro_direct": _weighted_metric_mean(entries, "exact_f1_macro_direct"),
        "exact_balanced_accuracy_direct": _weighted_metric_mean(entries, "exact_balanced_accuracy_direct"),
        "exact_kappa_direct": _weighted_metric_mean(entries, "exact_kappa_direct"),
        "exact_mcc_direct": _weighted_metric_mean(entries, "exact_mcc_direct"),
        "_cm_coarse": cm_coarse,
        "_cm_exact": cm_exact,
        "_per_family": per_family_all,
    }


def _build_split_client_loaders(
    event_table,
    df: pd.DataFrame,
    *,
    split: str,
    window: int,
    bs: int,
    seed: int,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    split_df = df[df["split"] == split]
    for toon_id in split_df["player_toon_id"].astype(str).unique().tolist():
        ds = event_table.build_dataset(
            split=split,
            player_ids={str(toon_id)},
            window=window,
            max_samples=0,
            shuffle=False,
            seed=seed,
            include_race=True,
        )
        if len(ds) == 0:
            continue
        loaders[str(toon_id)] = make_loaders(ds, bs, shuffle=False)
    return loaders


def _build_client_race_map(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    if "player_toon_id" not in df.columns or "player_race" not in df.columns:
        return out
    for toon_id, g in df.groupby("player_toon_id", observed=True):
        if len(g) <= 0:
            continue
        out[str(toon_id)] = str(g["player_race"].iloc[0])
    return out


def _build_single_race_eval_model(
    *,
    backbone_state: dict[str, torch.Tensor],
    head_state: dict[str, torch.Tensor],
    race: str,
    model_cfg: dict,
    num_families: int,
    fine_dims: list[int],
    num_exact_classes: int,
    device: torch.device,
) -> HierGRU:
    race_model = HierGRURaceHeads(**model_cfg).to(device)
    race_model.load_backbone_state_dict(backbone_state)
    race_model.load_head_state_dict(head_state)

    use_exact = num_exact_classes > 0 and race_model.use_exact_head
    model = HierGRU(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        layers=int(model_cfg["layers"]),
        dropout=float(model_cfg["dropout"]),
        model_name=str(model_cfg.get("model_name", "gru")),
        num_families=num_families,
        fine_dims=fine_dims,
        num_exact_classes=num_exact_classes if use_exact else 0,
        use_exact_head=use_exact,
    ).to(device)

    race_key = race if race in race_model.race_heads else str(model_cfg["races"][0])
    model.encoder.load_state_dict(race_model.encoder.state_dict())
    model.coarse_head.load_state_dict(race_model.race_heads[race_key]["coarse"].state_dict())
    race_fine_heads = race_model.race_heads[race_key]["fine_heads"]
    for fam_id, head in enumerate(model.fine_heads):
        head.load_state_dict(race_fine_heads[fam_id].state_dict())
    # Copy exact head weights if available
    if use_exact and "exact" in race_model.race_heads[race_key] and model.exact_head is not None:
        model.exact_head.load_state_dict(race_model.race_heads[race_key]["exact"].state_dict())

    return model


def evaluate_personalized_heads_race(
    *,
    backbone_state: dict[str, torch.Tensor],
    client_heads: dict[str, dict[str, torch.Tensor]],
    public_head: dict[str, torch.Tensor],
    client_loaders: dict[str, DataLoader],
    client_race_map: dict[str, str],
    model_cfg: dict,
    races: list[str],
    fine_dims: list[int],
    device: torch.device,
    eval_kwargs: dict,
    collect_rows: bool,
) -> tuple[dict, list[dict]]:
    entries: list[tuple[int, dict, bool]] = []
    rows: list[dict] = []

    for client_id, loader in sorted(client_loaders.items(), key=lambda kv: kv[0]):
        has_personalized = client_id in client_heads
        head_state = client_heads.get(client_id, public_head)
        race = client_race_map.get(client_id, races[0])

        model = _build_single_race_eval_model(
            backbone_state=backbone_state,
            head_state=head_state,
            race=race,
            model_cfg=model_cfg,
            num_families=int(eval_kwargs["num_families"]),
            fine_dims=fine_dims,
            num_exact_classes=int(eval_kwargs.get("num_exact_classes", 0)), # <--- ADD THIS LINE
            device=device,
        )


        m = evaluate_hier(model, loader, device, **eval_kwargs)
        n = int(m.get("n_examples", 0) or 0)
        entries.append((n, m, has_personalized))

        if collect_rows:
            rows.append(
                {
                    "player_toon_id": client_id,
                    "client_race": race,
                    "has_personalized_head": bool(has_personalized),
                    "n_examples": n,
                    "coarse_top1": m.get("coarse_top1"),
                    "coarse_top5": m.get("coarse_top5"),
                    "coarse_balanced_accuracy": m.get("coarse_balanced_accuracy"),
                    "exact_top1": m.get("exact_top1"),
                    "exact_f1_macro": m.get("exact_f1_macro"),
                    "exact_recall_macro": m.get("exact_recall_macro"),
                    "exact_kappa": m.get("exact_kappa"),
                }
            )

    aggregated = _aggregate_personalized_metrics(
        entries,
        num_families=int(eval_kwargs["num_families"]),
        num_exact_classes=int(eval_kwargs["num_exact_classes"]),
    )

    if rows:
        for race in races:
            race_rows = [r for r in rows if str(r.get("client_race")) == race and int(r.get("n_examples", 0) or 0) > 0]
            if not race_rows:
                aggregated[f"race_{race.lower()}_exact_top1"] = None
                aggregated[f"race_{race.lower()}_exact_recall_macro"] = None
                continue

            total_n = int(sum(int(r["n_examples"]) for r in race_rows))

            def _wavg(key: str) -> float | None:
                num = 0.0
                den = 0
                for rr in race_rows:
                    v = rr.get(key)
                    if v is None:
                        continue
                    n = int(rr["n_examples"])
                    num += float(v) * n
                    den += n
                if den <= 0:
                    return None
                return float(num / den)

            aggregated[f"race_{race.lower()}_exact_top1"] = _wavg("exact_top1")
            aggregated[f"race_{race.lower()}_exact_recall_macro"] = _wavg("exact_recall_macro")
            aggregated[f"race_{race.lower()}_n_examples"] = total_n

    return aggregated, rows


def save_per_class_recall(cm: np.ndarray, path: Path, class_prefix: str) -> None:
    support = cm.sum(axis=1)
    recall = np.zeros(cm.shape[0], dtype=np.float64)
    nz = support > 0
    recall[nz] = np.diag(cm)[nz] / support[nz]

    df = pd.DataFrame(
        {
            f"{class_prefix}_id": np.arange(cm.shape[0], dtype=np.int64),
            "support": support.astype(np.int64),
            "recall": recall,
        }
    )
    df.to_csv(path, index=False)


def client_update_backbone_head_race(
    global_backbone: dict[str, torch.Tensor],
    local_head: dict[str, torch.Tensor],
    model_cfg: dict,
    client_dataset,
    local_head_epochs: int,
    local_backbone_epochs: int,
    race_to_id: dict[str, int],
    races: list[str],
    bs: int,
    lr: float,
    head_lr_multiplier: float,
    weight_decay: float,
    coarse_loss_weight: float,
    fine_loss_weight: float,
    exact_loss_weight: float,
    label_smoothing: float,
    distill_weight: float,
    coarse_w_t: torch.Tensor | None,
    fine_class_weights: list[torch.Tensor | None],
    device: torch.device,
    max_batches: int = 0,
    workers: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int, float]:
    """Train backbone + head locally, return updated states."""
    # Load data with race labels
    loader_head = make_loaders(client_dataset, bs, shuffle=True, workers=workers)
    loader_backbone = make_loaders(client_dataset, bs, shuffle=True, workers=workers)

    # Initialize model
    model = HierGRURaceHeads(**model_cfg)
    model.load_backbone_state_dict(global_backbone)
    model.load_head_state_dict(local_head)
    model = model.to(device)

    global_backbone_model = HierGRURaceHeads(**model_cfg)
    global_backbone_model.load_backbone_state_dict(global_backbone)
    global_backbone_model = global_backbone_model.to(device)
    _set_requires_grad(global_backbone_model, False)

    # Head-only training
    _set_requires_grad(model.encoder, False)
    for race_heads in model.race_heads.values():
        _set_requires_grad(race_heads, True)

    optimizer_head = torch.optim.Adam(
        list(model.race_heads.parameters()),
        lr=lr * head_lr_multiplier,
        weight_decay=weight_decay,
        foreach=False,
    )

    loss_sum = 0.0
    n_examples = 0

    for epoch in range(local_head_epochs):
        for bidx, batch in enumerate(loader_head):
            if max_batches > 0 and bidx >= max_batches:
                break

            x, y_coarse, y_fine, y_exact, lengths, player_race = batch
            x = x.to(device)
            y_coarse = y_coarse.to(device)
            y_fine = y_fine.to(device)
            y_exact = y_exact.to(device)
            lengths = lengths.to(device)

            # Convert race strings to IDs
            race_ids = torch.tensor([race_to_id.get(r, 0) for r in player_race], device=device)

            # Forward pass with race routing
            outputs = model(x, lengths, race_ids)

            loss = 0.0
            n = int(x.shape[0])

            for race_idx, race in enumerate(races):
                if race not in outputs:
                    continue
                race_output = outputs[race]
                mask = race_output.get("mask", torch.zeros(n, dtype=torch.bool, device=device))
                if not mask.any():
                    continue

                coarse_logits = race_output["coarse"]
                fine_logits = race_output["fine"]
                y_c_masked = y_coarse[mask]
                y_f_masked = y_fine[mask]

                # Coarse loss
                ce_coarse = F.cross_entropy(
                    coarse_logits, y_c_masked,
                    weight=coarse_w_t,
                    reduction="mean",
                    label_smoothing=label_smoothing,
                )
                loss = loss + coarse_loss_weight * ce_coarse

                # Fine loss
                for fam_id, fam_logits in enumerate(fine_logits):
                    fam_mask = y_c_masked == fam_id
                    if not fam_mask.any():
                        continue
                    w = fine_class_weights[fam_id] if fam_id < len(fine_class_weights) else None
                    ce_fine = F.cross_entropy(
                        fam_logits[fam_mask], y_f_masked[fam_mask],
                        weight=w,
                        reduction="mean",
                    )
                    loss = loss + fine_loss_weight * ce_fine

                # Exact direct loss
                if exact_loss_weight > 0 and "exact" in race_output:
                    exact_logits = race_output["exact"]
                    y_e_masked = y_exact[mask]
                    ce_exact = F.cross_entropy(exact_logits, y_e_masked, reduction="mean")
                    loss = loss + exact_loss_weight * ce_exact

            optimizer_head.zero_grad()
            loss.backward()
            optimizer_head.step()

            loss_sum += float(loss.item()) * n
            n_examples += n

    # Backbone training
    _set_requires_grad(model.encoder, True)
    for race_heads in model.race_heads.values():
        _set_requires_grad(race_heads, False)

    optimizer_backbone = torch.optim.Adam(
        list(model.encoder.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        foreach=False,
    )

    for epoch in range(local_backbone_epochs):
        for bidx, batch in enumerate(loader_backbone):
            if max_batches > 0 and bidx >= max_batches:
                break

            x, y_coarse, y_fine, y_exact, lengths, player_race = batch
            x = x.to(device)
            y_coarse = y_coarse.to(device)
            y_fine = y_fine.to(device)
            y_exact = y_exact.to(device)
            lengths = lengths.to(device)

            race_ids = torch.tensor([race_to_id.get(r, 0) for r in player_race], device=device)

            # Forward pass
            z = model.encode(x, lengths)
            z_global = global_backbone_model.encode(x, lengths)

            # Distillation loss
            distill_loss = F.mse_loss(z, z_global, reduction="mean")

            outputs = model(x, lengths, race_ids)

            loss = distill_weight * distill_loss

            # Add task losses
            n = int(x.shape[0])
            for race_idx, race in enumerate(races):
                if race not in outputs:
                    continue
                race_output = outputs[race]
                mask = race_output.get("mask", torch.zeros(n, dtype=torch.bool, device=device))
                if not mask.any():
                    continue

                coarse_logits = race_output["coarse"]
                fine_logits = race_output["fine"]
                y_c_masked = y_coarse[mask]
                y_f_masked = y_fine[mask]

                ce_coarse = F.cross_entropy(
                    coarse_logits, y_c_masked,
                    weight=coarse_w_t,
                    reduction="mean",
                    label_smoothing=label_smoothing,
                )
                loss = loss + coarse_loss_weight * ce_coarse

                for fam_id, fam_logits in enumerate(fine_logits):
                    fam_mask = y_c_masked == fam_id
                    if not fam_mask.any():
                        continue
                    w = fine_class_weights[fam_id] if fam_id < len(fine_class_weights) else None
                    ce_fine = F.cross_entropy(
                        fam_logits[fam_mask], y_f_masked[fam_mask],
                        weight=w,
                        reduction="mean",
                    )
                    loss = loss + fine_loss_weight * ce_fine

                # Exact direct loss
                if exact_loss_weight > 0 and "exact" in race_output:
                    exact_logits = race_output["exact"]
                    y_e_masked = y_exact[mask]
                    ce_exact = F.cross_entropy(exact_logits, y_e_masked, reduction="mean")
                    loss = loss + exact_loss_weight * ce_exact

            optimizer_backbone.zero_grad()
            loss.backward()
            optimizer_backbone.step()

            loss_sum += float(loss.item()) * n
            n_examples += n

    final_backbone = model.backbone_state_dict()
    final_head = model.head_state_dict()

    avg_loss = loss_sum / max(1, n_examples)
    return final_backbone, final_head, n_examples, avg_loss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--coarse-taxonomy",
        default="auto",
        help="auto|dataset|legacy8|broad5|broad6_warp|balanced6|macro_tactical3",
    )
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--clients-per-round", type=int, default=50)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--local-head-epochs", type=int, default=0)
    ap.add_argument("--local-backbone-epochs", type=int, default=0)
    ap.add_argument("--local-lr", type=float, default=0.001)
    ap.add_argument("--head-lr-multiplier", type=float, default=1.0)
    ap.add_argument("--local-weight-decay", type=float, default=0.0)
    ap.add_argument("--local-bs", type=int, default=32)
    ap.add_argument("--eval-bs", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--backbone-agg-every", type=int, default=1)
    ap.add_argument(
        "--aggregate-heads",
        action="store_true",
        help="Ablation: aggregate race heads each round (default: keep all heads local).",
    )
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--coarse-loss-weight", type=float, default=1.0)
    ap.add_argument("--fine-loss-weight", type=float, default=1.0)
    ap.add_argument("--exact-loss-weight", type=float, default=0.3,
                    help="Weight for auxiliary direct exact-action CE loss (0 to disable). "
                         "Not used in backbone-head race model but kept for CLI consistency.")
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="Label smoothing epsilon for coarse cross-entropy (0 to disable).")
    ap.add_argument("--coarse-class-weight-mode", default="inverse_sqrt", help="none|inverse_sqrt|inverse")
    ap.add_argument("--fine-class-weight-mode", default="none")
    ap.add_argument("--max-class-weight", type=float, default=10.0)
    ap.add_argument("--min-client-samples", type=int, default=100)
    ap.add_argument("--max-client-samples", type=int, default=0)
    ap.add_argument("--max-local-batches", type=int, default=0)
    ap.add_argument("--max-eval-batches", type=int, default=0)
    ap.add_argument(
        "--action-context-features",
        choices=["on", "off"],
        default="on",
        help="Include observed current-action context in each input step.",
    )
    ap.add_argument("--selection-objective", default=None)
    ap.add_argument("--selection-primary", dest="selection_primary_legacy", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--selection-tiebreakers", default="")
    ap.add_argument("--cross-run-csv", default="runs/cross_run_results.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="Resume from latest_global_checkpoint.pt if it exists.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-name", default="gru")
    ap.add_argument("--race", choices=["all", "Prot", "Terr", "Zerg"], default="all")
    ap.add_argument(
        "--round-val-clients",
        type=int,
        default=0,
        help="Number of validation clients to sample during rounds (0 for all).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    selection_objective = args.selection_objective or args.selection_primary_legacy or "joint_honest"
    selection_primary, selection_tiebreakers = resolve_selection_objective(
        selection_objective,
        args.selection_tiebreakers,
    )
    local_head_epochs = int(args.local_head_epochs) if int(args.local_head_epochs) > 0 else int(args.local_epochs)
    local_backbone_epochs = (
        int(args.local_backbone_epochs)
        if int(args.local_backbone_epochs) > 0
        else int(args.local_epochs)
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Data
    events_path, preprocessing_path, action_vocab_path = resolve_dataset_paths(args.dataset_dir)
    df, feature_cols, _pre = load_and_normalize_events(
        events_path,
        preprocessing_path,
        race_filter=args.race,
    )
    effective_split_unit = _pre.get("effective_split_unit") or (
        "tournament" if _pre.get("split_mode") == "tournament"
        else "player_toon_id" if _pre.get("split_mode") == "player"
        else "replay_id"
    )
    vocab = json.loads(action_vocab_path.read_text())
    df, hierarchy, hierarchy_meta = prepare_hierarchy_targets(
        df,
        vocab,
        coarse_taxonomy=args.coarse_taxonomy,
    )
    assert_hierarchy_targets_usable(df, race=args.race)
    if args.race != "all":
        df = df[df["player_race"] == args.race].copy()
    df, feature_cols, action_context_meta = add_action_context_features(
        df,
        feature_cols,
        enabled=args.action_context_features == "on",
    )
    args.coarse_class_weight_mode = apply_pretrain_coarse_audit(
        df,
        race=args.race,
        outdir=outdir,
        coarse_class_weight_mode=args.coarse_class_weight_mode,
    )
    action_to_id = {str(k): int(v) for k, v in (vocab.get("action_to_id") or {}).items()}

    family_id_to_name, fine_dims, family_fine_to_exact_id, default_exact_id = build_family_maps(
        hierarchy,
        action_to_id,
        df=df,
    )
    observed_race_to_id, observed_races = build_race_maps(df)
    races = ["Prot", "Terr", "Zerg"] if args.race == "all" else [args.race]
    race_to_id = {race: idx for idx, race in enumerate(races)}

    # Preserve any non-canonical labels from artifacts to avoid routing failures.
    for race in observed_races:
        if race not in race_to_id:
            race_to_id[race] = 0

    num_families = len(fine_dims)
    num_exact_classes = int(df["exact_action_id"].max()) + 1

    train_exact_counts = df.loc[df["split"] == "train", "exact_action_id"].value_counts()
    if len(train_exact_counts):
        default_exact_id = int(train_exact_counts.index[0])

    effective_taxonomy = hierarchy_meta.get("effective_coarse_taxonomy")
    if effective_taxonomy:
        print(
            f"[backbone-head-race] coarse taxonomy request={args.coarse_taxonomy} "
            f"effective={effective_taxonomy}",
            flush=True,
        )

    # Clients
    event_table = build_event_table(df, feature_cols)
    client_sample_counts = event_table.player_sample_counts(split="train")
    eligible_clients = sorted([
        cid for cid, count in client_sample_counts.items()
        if int(count) >= args.min_client_samples
    ])

    if not eligible_clients:
        raise RuntimeError("No eligible clients. Lower --min-client-samples.")

    print(
        f"[backbone-head-race] {len(eligible_clients)} eligible clients, race-specific heads, "
        f"races={races}, observed_races={sorted(observed_races)}, backbone_agg_every={args.backbone_agg_every}",
        flush=True,
    )

    # Val/Test datasets
    ds_val = event_table.build_dataset(
        split="val", window=args.window, max_samples=0,
        shuffle=False, seed=args.seed, include_race=True,
    )
    ds_test = event_table.build_dataset(
        split="test", window=args.window, max_samples=0,
        shuffle=False, seed=args.seed, include_race=True,
    )

    val_client_loaders = _build_split_client_loaders(
        event_table,
        df,
        split="val",
        window=args.window,
        bs=args.eval_bs,
        seed=args.seed,
    )
    test_client_loaders = _build_split_client_loaders(
        event_table,
        df,
        split="test",
        window=args.window,
        bs=args.eval_bs,
        seed=args.seed,
    )
    client_race_map = _build_client_race_map(df)

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"

    if device_name == "dml":
        try:
            import torch_directml
            device = torch_directml.device()
        except ImportError:
            print("[device] torch-directml not installed, falling back to cpu")
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    # Model config for race-specific heads
    race_num_coarse = {race: num_families for race in races}
    race_fine_dims = {race: list(fine_dims) for race in races}

    mcfg = dict(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        model_name=args.model_name,
        race_num_coarse=race_num_coarse,
        race_fine_dims=race_fine_dims,
        races=races,
        num_exact_classes=num_exact_classes if args.exact_loss_weight > 0 else 0,
    )

    # Global backbone + public head template
    init_model = HierGRURaceHeads(**mcfg)
    global_backbone = init_model.backbone_state_dict()
    public_head = init_model.head_state_dict()
    del init_model

    # Each train client stores its own private head over rounds
    client_heads: dict[str, dict[str, torch.Tensor]] = {}

    # Class weights
    train_rows = df[df["split"] == "train"]
    coarse_labels = train_rows["coarse_family_id"].to_numpy(dtype=np.int64)
    coarse_w = build_class_weights(
        coarse_labels,
        num_families,
        args.coarse_class_weight_mode,
        args.max_class_weight,
    )
    coarse_w_t = None if coarse_w is None else torch.tensor(coarse_w, dtype=torch.float32, device=device)

    fine_class_weights: list[torch.Tensor | None] = []
    for fam_id in range(num_families):
        fam_rows = train_rows[train_rows["coarse_family_id"] == fam_id]
        fam_labels = fam_rows["fine_action_id"].to_numpy(dtype=np.int64)
        fam_w = build_class_weights(
            fam_labels,
            max(1, fine_dims[fam_id]),
            args.fine_class_weight_mode,
            args.max_class_weight,
        )
        fine_class_weights.append(None if fam_w is None else torch.tensor(fam_w, dtype=torch.float32, device=device))

    if coarse_w_t is not None:
        print(
            f"[backbone-head-race] coarse class weights ({args.coarse_class_weight_mode}): "
            f"{coarse_w_t.tolist()}",
            flush=True,
        )

    eval_kwargs = dict(
        num_exact_classes=num_exact_classes,
        num_families=num_families,
        family_fine_to_exact_id=family_fine_to_exact_id,
        default_exact_id=default_exact_id,
        coarse_loss_weight=args.coarse_loss_weight,
        fine_loss_weight=args.fine_loss_weight,
        max_batches=args.max_eval_batches,
    )

    config = vars(args).copy()
    config.update(
        {
            "mode": "backbone_head_race_heads",
            "architecture": args.model_name,
            "race": args.race,
            "n_features": len(feature_cols),
            **action_context_meta,
            "n_families": num_families,
            "n_races": len(races),
            "races": races,
            "n_exact_classes": num_exact_classes,
            "fine_dims": fine_dims,
            "n_eligible_clients": len(eligible_clients),
            "n_val": len(ds_val),
            "n_test": len(ds_test),
            "n_val_clients": len(val_client_loaders),
            "n_test_clients": len(test_client_loaders),
            "family_id_to_name": {str(k): v for k, v in family_id_to_name.items()},
            "split_mode": _pre.get("split_mode"),
            "effective_split_unit": effective_split_unit,
            "head_design": "race_specific_heads",
            "federated_aggregation_default": "backbone_only",
            "aggregate_heads_enabled": bool(args.aggregate_heads),
            "local_head_epochs_effective": local_head_epochs,
            "local_backbone_epochs_effective": local_backbone_epochs,
            "selection_objective": selection_objective,
            "selection_primary": selection_primary,
            "selection_tiebreakers_effective": selection_tiebreakers,
            **hierarchy_meta,
        }
    )
    (outdir / "config.json").write_text(json.dumps(config, indent=2))

    # FL rounds
    best_val_metrics: dict | None = None
    best_round: int | None = None
    rounds_rows: list[dict] = []
    per_client_round_rows: list[dict] = []
    best_global_backbone: dict[str, torch.Tensor] | None = None
    best_client_heads: dict[str, dict[str, torch.Tensor]] | None = None
    best_public_head: dict[str, torch.Tensor] | None = None
    total_train_time_sec = 0.0
    total_val_time_sec = 0.0
    comm_bytes_up_total = 0
    comm_bytes_down_total = 0
    run_t0 = time.time()
    start_round = 1

    checkpoint_path = outdir / "latest_global_checkpoint.pt"
    metrics_path = outdir / "round_metrics.csv"
    if args.resume and checkpoint_path.exists() and metrics_path.exists():
        print(f"[resume] loading from {checkpoint_path}", flush=True)
        checkpoint = _safe_torch_load(checkpoint_path)
        if checkpoint is None:
            prev_checkpoint_path = checkpoint_path.with_name(checkpoint_path.name + ".prev")
            if prev_checkpoint_path.exists():
                print(f"[resume] trying previous checkpoint {prev_checkpoint_path}", flush=True)
                checkpoint = _safe_torch_load(prev_checkpoint_path)

        rounds_df = pd.read_csv(metrics_path).fillna(0.0)
        if checkpoint is not None:
            global_backbone = checkpoint["backbone_state"]
            public_head = checkpoint["public_head"]
            client_heads = checkpoint["client_heads"]
            start_round = int(checkpoint["round"]) + 1
            rounds_rows = rounds_df.to_dict("records")
            best_round = int(checkpoint.get("best_round", 0))
            best_val_metrics = checkpoint.get("best_val_metrics")
            comm_bytes_down_total = float(checkpoint.get("comm_bytes_down_total", 0))
            comm_bytes_up_total = float(checkpoint.get("comm_bytes_up_total", 0))
        else:
            best_state_path = outdir / "best_state.pt"
            best_state = _safe_torch_load(best_state_path) if best_state_path.exists() else None
            if best_state is None:
                raise RuntimeError(
                    f"Cannot resume: {checkpoint_path} is unreadable and no usable "
                    f"{best_state_path.name} fallback exists."
                )
            best_round = int(best_state.get("best_round", 0))
            if best_round <= 0:
                raise RuntimeError(f"Cannot resume from {best_state_path}: missing best_round.")
            print(f"[resume] recovering from best_state.pt at round {best_round}", flush=True)
            global_backbone = best_state["global_backbone"]
            public_head = best_state.get("public_head", public_head)
            client_heads = best_state["client_heads"]
            start_round = best_round + 1
            rounds_df = rounds_df[pd.to_numeric(rounds_df["round"], errors="coerce") <= best_round].copy()
            rounds_rows = rounds_df.to_dict("records")
            if not rounds_df.empty:
                comm_bytes_down_total = float(pd.to_numeric(rounds_df.get("comm_bytes_down", 0), errors="coerce").fillna(0).sum())
                comm_bytes_up_total = float(pd.to_numeric(rounds_df.get("comm_bytes_up", 0), errors="coerce").fillna(0).sum())
                best_rows = rounds_df[pd.to_numeric(rounds_df["round"], errors="coerce") == best_round]
                if not best_rows.empty:
                    best_val_metrics = _metrics_from_round_row(best_rows.iloc[-1].to_dict())

        best_state_path = outdir / "best_state.pt"
        best_state = _safe_torch_load(best_state_path) if best_state_path.exists() else None
        if best_state is not None:
            best_global_backbone = _clone_state_dict(best_state["global_backbone"])
            best_client_heads = _clone_head_bank(best_state["client_heads"])
            best_public_head = _clone_state_dict(best_state["public_head"]) if best_state.get("public_head") else None

        _advance_round_rng(
            rng,
            completed_rounds=start_round - 1,
            eligible_clients=eligible_clients,
            clients_per_round=args.clients_per_round,
            val_client_loaders=val_client_loaders,
            round_val_clients=args.round_val_clients,
        )
        print(f"[resume] starting from round {start_round}", flush=True)

    for rnd in range(start_round, args.rounds + 1):
        round_t0 = time.time()
        sample_k = min(args.clients_per_round, len(eligible_clients))
        selected = rng.choice(eligible_clients, size=sample_k, replace=False).tolist()
        backbone_bytes = state_dict_num_bytes(global_backbone)
        head_bytes = state_dict_num_bytes(public_head) if args.aggregate_heads else 0

        backbone_updates: list[tuple[dict[str, torch.Tensor], int]] = []
        local_losses: list[float] = []
        local_sizes: list[int] = []
        client_train_time_sec = 0.0

        for cid in selected:
            client_dataset = event_table.build_dataset(
                split="train",
                player_ids={str(cid)},
                window=args.window,
                max_samples=args.max_client_samples,
                shuffle=True,
                seed=args.seed + rnd,
                include_race=True,
            )
            c_head_init = client_heads.get(cid, public_head)
            client_t0 = time.time()
            c_backbone, c_head_new, c_n, c_loss = client_update_backbone_head_race(
                global_backbone=global_backbone,
                local_head=c_head_init,
                model_cfg=mcfg,
                client_dataset=client_dataset,
                local_head_epochs=local_head_epochs,
                local_backbone_epochs=local_backbone_epochs,
                race_to_id=race_to_id,
                races=races,
                bs=args.local_bs,
                lr=args.local_lr,
                head_lr_multiplier=args.head_lr_multiplier,
                weight_decay=args.local_weight_decay,
                coarse_loss_weight=args.coarse_loss_weight,
                fine_loss_weight=args.fine_loss_weight,
                exact_loss_weight=args.exact_loss_weight,
                label_smoothing=args.label_smoothing,
                distill_weight=args.distill_weight,
                coarse_w_t=coarse_w_t,
                fine_class_weights=fine_class_weights,
                device=device,
                max_batches=args.max_local_batches,
                workers=args.workers,
            )
            client_train_time_sec += float(time.time() - client_t0)

            backbone_updates.append((c_backbone, c_n))
            client_heads[cid] = c_head_new
            local_losses.append(c_loss)
            local_sizes.append(c_n)
            per_client_round_rows.append(
                {
                    "round": rnd,
                    "player_toon_id": cid,
                    "client_race": client_race_map.get(cid),
                    "n_examples": int(c_n),
                    "client_loss": float(c_loss),
                    "client_train_time_sec": float(time.time() - client_t0),
                }
            )

        total_train_time_sec += client_train_time_sec
        if rnd % args.backbone_agg_every == 0 and backbone_updates:
            global_backbone = aggregate_state_dicts(backbone_updates)

        # Optional ablation only. Default behavior keeps all client heads local.
        if args.aggregate_heads and selected:
            public_head = aggregate_state_dicts(
                [(client_heads.get(cid, public_head), 1) for cid in selected]
            )

        val_t0 = time.time()
        
        # Stochastic validation sampling for speed during rounds
        current_val_loaders = val_client_loaders
        if args.round_val_clients > 0 and len(val_client_loaders) > args.round_val_clients:
            val_keys = sorted(list(val_client_loaders.keys()))
            sampled_keys = rng.choice(val_keys, size=args.round_val_clients, replace=False)
            current_val_loaders = {k: val_client_loaders[k] for k in sampled_keys}

        val_m, val_pc_rows = evaluate_personalized_heads_race(
            backbone_state=global_backbone,
            client_heads=client_heads,
            public_head=public_head,
            client_loaders=current_val_loaders,
            client_race_map=client_race_map,
            model_cfg=mcfg,
            races=races,
            fine_dims=fine_dims,
            device=device,
            eval_kwargs=eval_kwargs,
            collect_rows=True,
        )
        val_time_sec = float(time.time() - val_t0)
        assert_runtime_coarse_not_inflated(val_m, outdir=outdir, phase="validation_round", step=rnd)
        total_val_time_sec += val_time_sec
        round_time_sec = float(time.time() - round_t0)
        n_selected_clients = len(selected)
        round_bytes_down = int((backbone_bytes + head_bytes) * n_selected_clients)
        round_bytes_up = int((backbone_bytes + head_bytes) * n_selected_clients)
        round_bytes_total = round_bytes_down + round_bytes_up
        comm_bytes_down_total += round_bytes_down
        comm_bytes_up_total += round_bytes_up

        row = {
            "round": rnd,
            "n_selected_clients": n_selected_clients,
            "n_participating_clients": n_selected_clients,
            "n_train_examples": int(sum(local_sizes)),
            "client_loss": float(np.mean(local_losses)) if local_losses else None,
            "round_time_sec": round_time_sec,
            "client_train_time_sec": client_train_time_sec,
            "val_time_sec": val_time_sec,
            "train_examples_per_sec": float(sum(local_sizes) / client_train_time_sec) if client_train_time_sec > 0 else None,
            "val_examples": int(val_m.get("n_examples") or 0),
            "val_examples_per_sec": float((val_m.get("n_examples") or 0) / val_time_sec) if val_time_sec > 0 else None,
            "client_updates_per_sec": float(n_selected_clients / client_train_time_sec) if client_train_time_sec > 0 else None,
            "comm_bytes_down": round_bytes_down,
            "comm_bytes_up": round_bytes_up,
            "comm_bytes_total": round_bytes_total,
            "comm_bytes_cumulative": comm_bytes_down_total + comm_bytes_up_total,
            "comm_bytes_per_client": float(round_bytes_total / n_selected_clients) if n_selected_clients > 0 else None,
            "val_loss": val_m.get("loss"),
            "val_coarse_loss": val_m.get("coarse_loss"),
            "val_fine_loss": val_m.get("fine_loss"),
            "val_coarse_top1": val_m.get("coarse_top1"),
            "val_coarse_top5": val_m.get("coarse_top5"),
            "val_coarse_f1_macro": val_m.get("coarse_f1_macro"),
            "val_coarse_balanced_accuracy": val_m.get("coarse_balanced_accuracy"),
            "val_coarse_precision_macro": val_m.get("coarse_precision_macro"),
            "val_coarse_recall_macro": val_m.get("coarse_recall_macro"),
            "val_exact_top1": val_m.get("exact_top1"),
            "val_exact_f1_macro": val_m.get("exact_f1_macro"),
            "val_exact_balanced_accuracy": val_m.get("exact_balanced_accuracy"),
            "val_exact_precision_macro": val_m.get("exact_precision_macro"),
            "val_exact_recall_macro": val_m.get("exact_recall_macro"),
            "val_exact_kappa": val_m.get("exact_kappa"),
            "val_exact_top1_direct": val_m.get("exact_top1_direct"),
            "val_exact_f1_macro_direct": val_m.get("exact_f1_macro_direct"),
            "val_exact_balanced_accuracy_direct": val_m.get("exact_balanced_accuracy_direct"),
            "val_personalized_head_coverage_clients": val_m.get("personalized_head_coverage_clients"),
            "val_personalized_head_coverage_examples": val_m.get("personalized_head_coverage_examples"),
        }
        for race in races:
            row[f"val_race_{race.lower()}_exact_top1"] = val_m.get(f"race_{race.lower()}_exact_top1")
            row[f"val_race_{race.lower()}_recall_macro"] = val_m.get(
                f"race_{race.lower()}_exact_recall_macro"
            )
        rounds_rows.append(row)
        pd.DataFrame(rounds_rows).to_csv(outdir / "round_metrics.csv", index=False)
        if per_client_round_rows:
            pd.DataFrame(per_client_round_rows).to_csv(outdir / "per_client_round_metrics.csv", index=False)

        improved = is_better(
            val_m,
            best_val_metrics,
            primary=selection_primary,
            tiebreakers=selection_tiebreakers,
        )
        if improved:
            best_val_metrics = dict(val_m)
            best_round = rnd
            
            # Explicitly free old best state to prevent MemoryError spikes
            if best_global_backbone is not None:
                del best_global_backbone
                del best_client_heads
                del best_public_head
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            best_global_backbone = _clone_state_dict(global_backbone)
            best_client_heads = _clone_head_bank(client_heads)
            best_public_head = _clone_state_dict(public_head) if public_head else None
            _atomic_torch_save(
                {
                    "global_backbone": best_global_backbone,
                    "client_heads": best_client_heads,
                    "public_head": best_public_head,
                    "best_round": best_round,
                },
                outdir / "best_state.pt",
            )

        c1 = row["val_coarse_top1"]
        c5 = row["val_coarse_top5"]
        e1 = row["val_exact_top1"]
        ed = row.get("val_exact_top1_direct")
        c1_txt = "na" if c1 is None else f"{float(c1):.4f}"
        c5_txt = "na" if c5 is None else f"{float(c5):.4f}"
        e1_txt = "na" if e1 is None else f"{float(e1):.4f}"
        ed_txt = "" if ed is None else f" exact_direct={float(ed):.4f}"
        print(
            f"round={rnd} clients={len(selected)} "
            f"loss={float(row['client_loss'] or 0.0):.4f} "
            f"coarse_top1={c1_txt} coarse_top5={c5_txt} exact_top1={e1_txt}"
            + ed_txt
            + (" *" if improved else ""),
            flush=True,
        )

        # Clean up memory before saving
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Checkpoint every round
        _atomic_torch_save({
            "round": rnd,
            "backbone_state": global_backbone,
            "public_head": public_head,
            "client_heads": client_heads,
            "best_round": best_round,
            "best_val_metrics": best_val_metrics,
            "comm_bytes_down_total": comm_bytes_down_total,
            "comm_bytes_up_total": comm_bytes_up_total,
        }, checkpoint_path)

        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rounds_df = pd.DataFrame(rounds_rows)
    rounds_df.to_csv(outdir / "round_metrics.csv", index=False)
    plot_round_curves(rounds_df, outdir)
    plot_metrics_curves(rounds_df, outdir, prefix="val_")

    if per_client_round_rows:
        pd.DataFrame(per_client_round_rows).to_csv(outdir / "per_client_round_metrics.csv", index=False)

    final_backbone = best_global_backbone if best_global_backbone is not None else global_backbone
    final_client_heads = best_client_heads if best_client_heads is not None else client_heads
    final_public_head = best_public_head if best_public_head is not None else public_head

    test_t0 = time.time()
    test_m, test_pc_rows = evaluate_personalized_heads_race(
        backbone_state=final_backbone,
        client_heads=final_client_heads,
        public_head=final_public_head,
        client_loaders=test_client_loaders,
        client_race_map=client_race_map,
        model_cfg=mcfg,
        races=races,
        fine_dims=fine_dims,
        device=device,
        eval_kwargs=eval_kwargs,
        collect_rows=True,
    )
    assert_runtime_coarse_not_inflated(test_m, outdir=outdir, phase="final_test", step=best_round)

    test_json = metrics_for_json(test_m)
    test_time_sec = float(time.time() - test_t0)
    total_wall_clock_sec = float(time.time() - run_t0)
    test_json["best_round"] = best_round
    test_json["mode"] = "backbone_head_race_heads"
    test_json["architecture"] = args.model_name
    test_json["race"] = args.race
    test_json["taxonomy"] = hierarchy_meta.get("effective_coarse_taxonomy")
    test_json["split_mode"] = _pre.get("split_mode")
    test_json["effective_split_unit"] = effective_split_unit
    test_json["backbone_agg_every"] = args.backbone_agg_every
    test_json["aggregate_heads"] = bool(args.aggregate_heads)
    test_json["head_design"] = "race_specific_heads"
    test_json["selection_objective"] = selection_objective
    test_json["selection_primary"] = selection_primary
    test_json["selection_tiebreakers_effective"] = selection_tiebreakers
    test_json.update(action_context_meta)
    test_json["train_wall_clock_sec"] = total_train_time_sec
    test_json["validation_wall_clock_sec"] = total_val_time_sec
    test_json["test_wall_clock_sec"] = test_time_sec
    test_json["total_wall_clock_sec"] = total_wall_clock_sec
    test_json["avg_round_time_sec"] = float(rounds_df["round_time_sec"].mean()) if not rounds_df.empty else None
    test_json["avg_client_train_time_sec"] = float(rounds_df["client_train_time_sec"].mean()) if not rounds_df.empty else None
    test_json["round_examples_per_sec"] = float(rounds_df["train_examples_per_sec"].mean()) if not rounds_df.empty else None
    test_json["client_updates_per_sec"] = float(rounds_df["client_updates_per_sec"].mean()) if not rounds_df.empty else None
    test_json["test_examples_per_sec"] = float((test_m.get("n_examples") or 0) / test_time_sec) if test_time_sec > 0 else None
    test_json["comm_bytes_down_total"] = int(comm_bytes_down_total)
    test_json["comm_bytes_up_total"] = int(comm_bytes_up_total)
    test_json["comm_bytes_total"] = int(comm_bytes_down_total + comm_bytes_up_total)
    test_json["comm_bytes_per_round_avg"] = float((comm_bytes_down_total + comm_bytes_up_total) / args.rounds) if args.rounds > 0 else None
    total_selected_clients = int(rounds_df["n_selected_clients"].sum()) if not rounds_df.empty else 0
    test_json["comm_bytes_per_client_avg"] = float((comm_bytes_down_total + comm_bytes_up_total) / total_selected_clients) if total_selected_clients > 0 else None
    (outdir / "final_test.json").write_text(json.dumps(test_json, indent=2))

    save_confusion_csv(test_m["_cm_coarse"], outdir / "test_coarse_confusion.csv")
    save_confusion_csv(test_m["_cm_exact"], outdir / "test_exact_confusion.csv")
    save_per_class_recall(test_m["_cm_coarse"], outdir / "test_coarse_per_class_recall.csv", "coarse_class")
    save_per_class_recall(test_m["_cm_exact"], outdir / "test_exact_per_class_recall.csv", "exact_class")
    save_family_summary(test_m.get("_per_family", []), family_id_to_name, outdir / "test_per_family.csv")

    if test_pc_rows:
        pc_df = pd.DataFrame(test_pc_rows).sort_values("n_examples", ascending=False)
        pc_df.to_csv(outdir / "per_client_test_metrics.csv", index=False)

        race_summary_rows = []
        for race in races:
            race_df = pc_df[pc_df["client_race"] == race]
            if race_df.empty:
                continue
            weights = race_df["n_examples"].astype(float)
            den = float(weights.sum())

            def _weighted(df: pd.DataFrame, key: str) -> float | None:
                if key not in df.columns:
                    return None
                vals = pd.to_numeric(df[key], errors="coerce")
                mask = vals.notna()
                if not bool(mask.any()):
                    return None
                ww = weights[mask]
                vv = vals[mask]
                d = float(ww.sum())
                if d <= 0:
                    return None
                return float((vv * ww).sum() / d)

            race_summary_rows.append(
                {
                    "race": race,
                    "n_clients": int(race_df.shape[0]),
                    "n_examples": int(race_df["n_examples"].sum()),
                    "exact_top1": _weighted(race_df, "exact_top1"),
                    "exact_recall_macro": _weighted(race_df, "exact_recall_macro"),
                    "coarse_top1": _weighted(race_df, "coarse_top1"),
                    "coarse_balanced_accuracy": _weighted(race_df, "coarse_balanced_accuracy"),
                }
            )

        if race_summary_rows:
            pd.DataFrame(race_summary_rows).to_csv(outdir / "per_race_test_summary.csv", index=False)
            
    # Heatmap visualization
    coarse_names = [family_id_to_name.get(i, f"C{i}") for i in range(num_families)]
    plot_confusion_matrix(test_m["_cm_coarse"], coarse_names, outdir / "test_coarse_confusion.png", "Coarse Action Confusion (Backbone-Head)")

    summary = {
        "best_round": best_round,
        "mode": "backbone_head_race_heads",
        "race": args.race,
        "backbone_agg_every": args.backbone_agg_every,
        "aggregate_heads": bool(args.aggregate_heads),
        "head_design": "race_specific_heads",
        "n_races": len(races),
        "races": races,
        "n_eligible_clients": len(eligible_clients),
        "selection_objective": selection_objective,
        "selection_primary": selection_primary,
        "selection_tiebreakers_effective": selection_tiebreakers,
        "final_test": test_json,
    }
    (outdir / "checkpoint_summary.json").write_text(json.dumps(summary, indent=2))

    cross_csv = Path(args.cross_run_csv)
    cross_csv.parent.mkdir(parents=True, exist_ok=True)
    append_cross_run_row(
        cross_csv,
        mode="backbone_head_race_heads",
        dataset_dir=args.dataset_dir,
        n_train=sum(int(v) for v in client_sample_counts.values()),
        n_val=len(ds_val),
        n_test=len(ds_test),
        n_clients=len(eligible_clients),
        epochs_or_rounds=args.rounds,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        window=args.window,
        metrics=test_json,
        best_epoch_or_round=best_round,
        outdir=str(outdir),
    )

    print(f"[done] backbone-head race-specific. best_round={best_round}")
    print(f"[done] artifacts saved to {outdir}")


if __name__ == "__main__":
    main()
