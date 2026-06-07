#!/usr/bin/env python3
"""Federated Averaging (FedAvg) for hierarchical GRU on SC2EGSet.

Full model (backbone + heads) is aggregated every round.
Clients are real SC2 pro players (player_toon_id).
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
    build_client_hier_sequences,
    build_event_table,
    build_family_maps,
    collate_hier_batch,
    load_and_normalize_events,
    make_loaders,
    prepare_hierarchy_targets,
    resolve_dataset_paths,
)
from hier_metrics import (
    append_cross_run_row,
    evaluate_hier,
    is_better,
    metrics_for_json,
    plot_round_curves,
    plot_confusion_matrix,
    resolve_selection_objective,
    save_confusion_csv,
    save_family_summary,
    save_per_class_recall,
    state_dict_num_bytes,
)
from hier_models import HierGRU, aggregate_state_dicts, family_fine_loss_mean, model_config_dict


def _fmt_metric(value) -> str:
    return "na" if value is None else f"{float(value):.4f}"


def _atomic_torch_save(payload, path: Path) -> None:
    """Save a checkpoint/model without leaving a half-written target."""
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


def _advance_round_rng(
    rng: np.random.Generator,
    *,
    completed_rounds: int,
    eligible_clients: list[str],
    clients_per_round: int,
) -> None:
    """Advance client-sampling RNG so resumed runs match uninterrupted runs."""
    sample_k = min(int(clients_per_round), len(eligible_clients))
    for _ in range(max(0, int(completed_rounds))):
        rng.choice(eligible_clients, size=sample_k, replace=False)


def client_update_fedavg(
    *,
    global_state: dict[str, torch.Tensor],
    model_cfg: dict,
    client_dataset,
    local_epochs: int,
    bs: int,
    lr: float,
    weight_decay: float,
    coarse_loss_weight: float,
    fine_loss_weight: float,
    exact_loss_weight: float,
    label_smoothing: float,
    coarse_w_t: torch.Tensor | None,
    fine_class_weights: list[torch.Tensor | None],
    device: torch.device,
    max_batches: int,
    workers: int = 0,
) -> tuple[dict[str, torch.Tensor], int, float]:
    """Train local model from global init, return updated state dict."""
    model = HierGRU(**model_cfg).to(device)
    model.load_state_dict(global_state)
    use_exact_head = model.use_exact_head

    loader = DataLoader(
        client_dataset, batch_size=bs, shuffle=True,
        collate_fn=collate_hier_batch, num_workers=workers,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    total = 0
    loss_sum = 0.0

    for _ in range(local_epochs):
        for bidx, (x, y_coarse, y_fine, y_exact, lengths) in enumerate(loader):
            if max_batches > 0 and bidx >= max_batches:
                break
            x, y_coarse, y_fine, y_exact, lengths = (
                x.to(device), y_coarse.to(device), y_fine.to(device),
                y_exact.to(device), lengths.to(device),
            )

            model_out = model(x, lengths)
            if use_exact_head:
                coarse_logits, fine_logits, exact_logits = model_out
            else:
                coarse_logits, fine_logits = model_out
                exact_logits = None
            coarse_loss = F.cross_entropy(
                coarse_logits, y_coarse, weight=coarse_w_t,
                label_smoothing=label_smoothing,
            )
            fine_loss = family_fine_loss_mean(fine_logits, y_coarse, y_fine, fine_class_weights)
            loss = coarse_loss_weight * coarse_loss + fine_loss_weight * fine_loss
            if exact_logits is not None and exact_loss_weight > 0:
                loss = loss + exact_loss_weight * F.cross_entropy(exact_logits, y_exact)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            n = int(y_coarse.numel())
            total += n
            loss_sum += float(loss.item()) * n

    state = model.full_state_dict_cpu()
    return state, total, float(loss_sum / max(1, total))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FedAvg hierarchical GRU on SC2EGSet.")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--outdir", default="runs/hier_fedavg")
    ap.add_argument("--cross-run-csv", default="runs/cross_run_results.csv")

    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--clients-per-round", type=int, default=50)
    ap.add_argument("--min-client-samples", type=int, default=64)
    ap.add_argument("--max-client-samples", type=int, default=2000)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--local-bs", type=int, default=128)
    ap.add_argument("--local-lr", type=float, default=1e-3)
    ap.add_argument("--local-weight-decay", type=float, default=1e-5)
    ap.add_argument("--max-local-batches", type=int, default=0)

    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--eval-bs", type=int, default=256)
    ap.add_argument("--max-eval-batches", type=int, default=0)
    ap.add_argument(
        "--action-context-features",
        choices=["on", "off"],
        default="on",
        help="Include observed current-action context in each input step.",
    )

    ap.add_argument("--coarse-loss-weight", type=float, default=1.0)
    ap.add_argument("--fine-loss-weight", type=float, default=1.0)
    ap.add_argument("--exact-loss-weight", type=float, default=0.3,
                    help="Weight for auxiliary direct exact-action CE loss (0 to disable).")
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="Label smoothing epsilon for coarse cross-entropy (0 to disable).")
    ap.add_argument("--coarse-class-weight-mode", choices=["none", "inverse", "inverse_sqrt"], default="inverse_sqrt")
    ap.add_argument("--fine-class-weight-mode", choices=["none", "inverse", "inverse_sqrt"], default="none")
    ap.add_argument("--max-class-weight", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="Resume from latest_global_checkpoint.pt if it exists.")
    ap.add_argument(
        "--coarse-taxonomy",
        default="auto",
        help="Taxonomy name (auto, macro_tactical3, legacy8, etc.)",
    )

    ap.add_argument(
        "--selection-objective",
        default=None,
        help="Named selection objective (coarse_honest, exact_honest, joint_honest) "
             "or a raw primary metric key.",
    )
    ap.add_argument("--selection-primary", dest="selection_primary_legacy", default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--selection-tiebreakers",
        default=None,
        help="Comma-separated tiebreaker metrics (only used when --selection-objective "
             "is a raw metric key, not a named preset).",
    )

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-name", choices=["gru", "lstm", "transformer"], default="gru")
    ap.add_argument("--race", choices=["all", "Prot", "Terr", "Zerg"], default="all")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    selection_objective = args.selection_objective or args.selection_primary_legacy or "joint_honest"
    selection_primary, selection_tiebreakers = resolve_selection_objective(
        selection_objective, args.selection_tiebreakers,
    )

    # ── Data ─────────────────────────────────────────────────────────────
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

    family_id_to_name, fine_dims, family_fine_to_exact_id, default_exact_id = build_family_maps(hierarchy, action_to_id, df=df)
    num_families = len(fine_dims)
    num_exact_classes = int(df["exact_action_id"].max()) + 1

    train_exact_counts = df.loc[df["split"] == "train", "exact_action_id"].value_counts()
    if len(train_exact_counts):
        default_exact_id = int(train_exact_counts.index[0])

    # ── Clients ──────────────────────────────────────────────────────────
    event_table = build_event_table(df, feature_cols)
    client_sample_counts = event_table.player_sample_counts(split="train")
    eligible_clients = sorted([
        cid for cid, count in client_sample_counts.items()
        if int(count) >= args.min_client_samples
    ])

    if not eligible_clients:
        raise RuntimeError("No eligible clients. Lower --min-client-samples.")

    print(f"[fedavg] {len(eligible_clients)} eligible clients (of {len(client_sample_counts)} total)", flush=True)

    # ── Val / Test ───────────────────────────────────────────────────────
    ds_val = event_table.build_dataset(split="val", window=args.window, max_samples=0, shuffle=False, seed=args.seed)
    ds_test = event_table.build_dataset(split="test", window=args.window, max_samples=0, shuffle=False, seed=args.seed)
    loader_val = make_loaders(ds_val, args.eval_bs, shuffle=False)
    loader_test = make_loaders(ds_test, args.eval_bs, shuffle=False)

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

    # ── Model ────────────────────────────────────────────────────────────
    use_exact_head = args.exact_loss_weight > 0
    mcfg = model_config_dict(len(feature_cols), args.hidden, args.layers, args.dropout, args.model_name, num_families, fine_dims)
    mcfg["num_exact_classes"] = num_exact_classes
    mcfg["use_exact_head"] = use_exact_head
    global_model = HierGRU(**mcfg).to(device)

    # ── Class weights ────────────────────────────────────────────────────
    train_rows = df[df["split"] == "train"]
    coarse_labels = train_rows["coarse_family_id"].to_numpy(dtype=np.int64)
    coarse_w = build_class_weights(coarse_labels, num_families, args.coarse_class_weight_mode, args.max_class_weight)
    coarse_w_t = None if coarse_w is None else torch.tensor(coarse_w, dtype=torch.float32, device=device)

    fine_class_weights: list[torch.Tensor | None] = []
    for fam_id in range(num_families):
        fam_rows = train_rows[train_rows["coarse_family_id"] == fam_id]
        fam_labels = fam_rows["fine_action_id"].to_numpy(dtype=np.int64)
        fam_w = build_class_weights(fam_labels, max(1, fine_dims[fam_id]), args.fine_class_weight_mode, args.max_class_weight)
        fine_class_weights.append(None if fam_w is None else torch.tensor(fam_w, dtype=torch.float32, device=device))

    eval_kwargs = dict(
        num_exact_classes=num_exact_classes, num_families=num_families,
        family_fine_to_exact_id=family_fine_to_exact_id,
        default_exact_id=default_exact_id,
        coarse_loss_weight=args.coarse_loss_weight, fine_loss_weight=args.fine_loss_weight,
        max_batches=args.max_eval_batches,
    )

    # ── Config ───────────────────────────────────────────────────────────
    config = vars(args).copy()
    config.update({
        "mode": "fedavg", "architecture": args.model_name, "race": args.race, "n_features": len(feature_cols),
        **action_context_meta,
        "n_families": num_families, "n_exact_classes": num_exact_classes,
        "fine_dims": fine_dims, "n_eligible_clients": len(eligible_clients),
        "n_val": len(ds_val), "n_test": len(ds_test),
        "family_id_to_name": {str(k): v for k, v in family_id_to_name.items()},
        "split_mode": _pre.get("split_mode"),
        "effective_split_unit": effective_split_unit,
        **hierarchy_meta,
    })
    (outdir / "config.json").write_text(json.dumps(config, indent=2))

    # ── FL rounds ────────────────────────────────────────────────────────
    best_val_metrics: dict | None = None
    best_round: int | None = None
    rounds_rows: list[dict] = []
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
        if checkpoint is None:
            raise RuntimeError(f"Cannot resume: no usable checkpoint found for {outdir}")
        global_model.load_state_dict(checkpoint["model_state"])
        start_round = checkpoint["round"] + 1
        
        rounds_df = pd.read_csv(metrics_path).fillna(0.0)
        rounds_rows = rounds_df.to_dict("records")
        best_round = int(checkpoint.get("best_round", 0))
        best_val_metrics = checkpoint.get("best_val_metrics")
        comm_bytes_down_total = float(checkpoint.get("comm_bytes_down_total", 0))
        comm_bytes_up_total = float(checkpoint.get("comm_bytes_up_total", 0))
        _advance_round_rng(
            rng,
            completed_rounds=start_round - 1,
            eligible_clients=eligible_clients,
            clients_per_round=args.clients_per_round,
        )
        print(f"[resume] starting from round {start_round}", flush=True)

    for rnd in range(start_round, args.rounds + 1):
        round_t0 = time.time()
        sample_k = min(args.clients_per_round, len(eligible_clients))
        selected = rng.choice(eligible_clients, size=sample_k, replace=False).tolist()

        global_state = global_model.full_state_dict_cpu()
        state_bytes = state_dict_num_bytes(global_state)
        weighted_states: list[tuple[dict[str, torch.Tensor], int]] = []
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
            )
            client_t0 = time.time()
            c_state, c_n, c_loss = client_update_fedavg(
                global_state=global_state, model_cfg=mcfg,
                client_dataset=client_dataset,
                local_epochs=args.local_epochs, bs=args.local_bs,
                lr=args.local_lr, weight_decay=args.local_weight_decay,
                coarse_loss_weight=args.coarse_loss_weight,
                fine_loss_weight=args.fine_loss_weight,
                exact_loss_weight=args.exact_loss_weight,
                label_smoothing=args.label_smoothing,
                coarse_w_t=coarse_w_t, fine_class_weights=fine_class_weights,
                device=device, max_batches=args.max_local_batches,
                workers=args.workers,
            )
            client_train_time_sec += float(time.time() - client_t0)
            weighted_states.append((c_state, c_n))
            local_losses.append(c_loss)
            local_sizes.append(c_n)

        total_train_time_sec += client_train_time_sec
        agg_state = aggregate_state_dicts(weighted_states)
        global_model.load_state_dict(agg_state)

        val_t0 = time.time()
        val_m = evaluate_hier(global_model, loader_val, device, **eval_kwargs)
        assert_runtime_coarse_not_inflated(val_m, outdir=outdir, phase="validation_round", step=rnd)
        val_time_sec = float(time.time() - val_t0)
        total_val_time_sec += val_time_sec
        round_time_sec = float(time.time() - round_t0)
        n_selected_clients = len(selected)
        round_bytes_down = int(state_bytes * n_selected_clients)
        round_bytes_up = int(state_bytes * n_selected_clients)
        round_bytes_total = round_bytes_down + round_bytes_up
        comm_bytes_down_total += round_bytes_down
        comm_bytes_up_total += round_bytes_up

        row = {
            "round": rnd,
            "n_selected_clients": n_selected_clients,
            "n_participating_clients": n_selected_clients,
            "n_train_examples": sum(local_sizes),
            "client_loss": float(np.mean(local_losses)),
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
        }
        rounds_rows.append(row)
        pd.DataFrame(rounds_rows).to_csv(outdir / "round_metrics.csv", index=False)

        if is_better(
            val_m,
            best_val_metrics,
            primary=selection_primary,
            tiebreakers=selection_tiebreakers,
        ):
            best_val_metrics = dict(val_m)
            best_round = rnd
            _atomic_torch_save(global_model.state_dict(), outdir / "best_model.pt")

        direct_str = f" exact_direct={val_m.get('exact_top1_direct','na')}" if val_m.get("exact_top1_direct") is not None else ""
        print(
            f"round={rnd} clients={len(selected)} loss={row['client_loss']:.4f} "
            f"coarse_top1={row['val_coarse_top1']:.4f} coarse_top5={row['val_coarse_top5']:.4f} "
            f"exact_top1={row['val_exact_top1']:.4f} "
            f"f1={row['val_exact_f1_macro']:.4f} kappa={row['val_exact_kappa']}"
            + direct_str
            + (" *" if rnd == best_round else ""),
            flush=True,
        )

        # Checkpoint every round
        _atomic_torch_save({
            "round": rnd,
            "model_state": global_model.state_dict(),
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

    # ── Final test ───────────────────────────────────────────────────────
    if (outdir / "best_model.pt").exists():
        global_model.load_state_dict(torch.load(outdir / "best_model.pt", map_location="cpu", weights_only=True))

    test_t0 = time.time()
    test_m = evaluate_hier(global_model, loader_test, device, **eval_kwargs)
    assert_runtime_coarse_not_inflated(test_m, outdir=outdir, phase="final_test", step=best_round)
    test_time_sec = float(time.time() - test_t0)
    test_json = metrics_for_json(test_m)
    total_wall_clock_sec = float(time.time() - run_t0)
    test_json["best_round"] = best_round
    test_json["mode"] = "fedavg"
    test_json["architecture"] = args.model_name
    test_json["race"] = args.race
    test_json["taxonomy"] = hierarchy_meta.get("effective_coarse_taxonomy")
    test_json["split_mode"] = _pre.get("split_mode")
    test_json["effective_split_unit"] = effective_split_unit
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

    # Heatmap visualization
    coarse_names = [family_id_to_name.get(i, f"C{i}") for i in range(num_families)]
    plot_confusion_matrix(test_m["_cm_coarse"], coarse_names, outdir / "test_coarse_confusion.png", "Coarse Action Confusion (FedAvg)")

    save_per_class_recall(test_m["_cm_coarse"], outdir / "test_coarse_per_class_recall.csv", "coarse_class")
    save_per_class_recall(test_m["_cm_exact"], outdir / "test_exact_per_class_recall.csv", "exact_class")
    save_family_summary(test_m.get("_per_family", []), family_id_to_name, outdir / "test_per_family.csv")

    # per-client test
    _save_per_client_test(global_model, event_table, df, args, device, eval_kwargs, outdir)

    summary = {
        "best_round": best_round,
        "mode": "fedavg",
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
        cross_csv, mode="fedavg", dataset_dir=args.dataset_dir,
        n_train=sum(int(v) for v in client_sample_counts.values()),
        n_val=len(ds_val), n_test=len(ds_test),
        n_clients=len(eligible_clients),
        epochs_or_rounds=args.rounds,
        hidden=args.hidden, layers=args.layers, dropout=args.dropout, window=args.window,
        metrics=test_json, best_epoch_or_round=best_round, outdir=str(outdir),
    )

    print(f"[done] fedavg. best_round={best_round}")
    print(
        "  exact_top1="
        f"{_fmt_metric(test_json.get('exact_top1'))}  "
        f"f1_macro={_fmt_metric(test_json.get('exact_f1_macro'))}  "
        f"kappa={_fmt_metric(test_json.get('exact_kappa'))}"
    )


def _save_per_client_test(model, event_table, df, args, device, eval_kwargs, outdir):
    """Evaluate global model per player on test set."""
    test_df = df[df["split"] == "test"].copy()
    rows = []
    for toon_id, g in test_df.groupby("player_toon_id", observed=True):
        ds = event_table.build_dataset(
            split="test", player_ids={str(toon_id)}, window=args.window,
            max_samples=0, shuffle=False, seed=0,
        )
        if len(ds) == 0:
            continue
        loader = make_loaders(ds, args.eval_bs, shuffle=False)
        m = evaluate_hier(model, loader, device, **eval_kwargs)
        rows.append({
            "player_toon_id": toon_id,
            "n_examples": m.get("n_examples", 0),
            "exact_top1": m.get("exact_top1"),
            "exact_f1_macro": m.get("exact_f1_macro"),
            "exact_kappa": m.get("exact_kappa"),
            "coarse_top1": m.get("coarse_top1"),
        })
    if rows:
        pc_df = pd.DataFrame(rows).sort_values("n_examples", ascending=False)
        pc_df.to_csv(outdir / "per_client_test_metrics.csv", index=False)


if __name__ == "__main__":
    main()
