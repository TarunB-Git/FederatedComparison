#!/usr/bin/env python3
"""Centralized hierarchical GRU training on SC2EGSet Protoss data."""
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

from hier_common import (
    add_action_context_features,
    apply_pretrain_coarse_audit,
    assert_hierarchy_targets_usable,
    assert_runtime_coarse_not_inflated,
    build_class_weights,
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
    plot_training_curves,
    plot_confusion_matrix,
    resolve_selection_objective,
    save_confusion_csv,
    save_family_summary,
    save_per_class_recall,
)
from hier_models import HierGRU, family_fine_loss_mean


def _fmt_metric(value) -> str:
    return "na" if value is None else f"{float(value):.4f}"


def _decode_exact_predictions(
    coarse_pred: torch.Tensor,
    pred_fine_pred: torch.Tensor,
    *,
    family_fine_to_exact_id: dict[int, dict[int, int]],
    default_exact_id: int,
    num_exact_classes: int,
) -> torch.Tensor:
    pred_exact = torch.zeros_like(pred_fine_pred)
    n = int(pred_fine_pred.numel())
    for i in range(n):
        fam_id = int(coarse_pred[i].item())
        fine_id = int(pred_fine_pred[i].item())
        exact_id = family_fine_to_exact_id.get(fam_id, {}).get(fine_id, default_exact_id)
        if exact_id < 0 or exact_id >= num_exact_classes:
            exact_id = default_exact_id
        pred_exact[i] = int(exact_id)
    return pred_exact


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Centralized hierarchical GRU training on SC2EGSet.",
    )
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--outdir", default="runs/hier_centralized")
    ap.add_argument("--cross-run-csv", default="runs/cross_run_results.csv")

    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-5)

    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument("--max-test-samples", type=int, default=0)
    ap.add_argument("--max-train-batches", type=int, default=0)
    ap.add_argument("--max-eval-batches", type=int, default=0)
    ap.add_argument("--log-every-batches", type=int, default=1000)
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
    ap.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine",
                    help="Learning rate schedule: none (flat) or cosine (with warmup).")
    ap.add_argument("--warmup-fraction", type=float, default=0.05,
                    help="Fraction of total training steps for LR warmup (only used with cosine).")
    ap.add_argument("--coarse-class-weight-mode", choices=["none", "inverse", "inverse_sqrt"], default="inverse_sqrt")
    ap.add_argument("--fine-class-weight-mode", choices=["none", "inverse", "inverse_sqrt"], default="none")
    ap.add_argument("--max-class-weight", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="Resume from latest_checkpoint.pt if it exists.")
    ap.add_argument(
        "--coarse-taxonomy",
        default="auto",
        help="Taxonomy name for coarse labels (auto, macro_tactical3, legacy8, etc.)",
    )

    ap.add_argument(
        "--selection-objective",
        default=None,
        help="Named selection objective (coarse_honest, exact_honest, joint_honest) "
             "or a raw primary metric key. Default: joint_honest.",
    )
    ap.add_argument("--selection-primary", dest="selection_primary_legacy", default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--selection-tiebreakers",
        default=None,
        help="Comma-separated tiebreaker metrics (only used when --selection-objective "
             "is a raw metric key, not a named preset).",
    )

    ap.add_argument("--early-stop-patience", type=int, default=8)
    ap.add_argument("--early-stop-min-epochs", type=int, default=8)

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-name", choices=["gru", "lstm", "transformer"], default="gru")
    ap.add_argument("--race", choices=["all", "Prot", "Terr", "Zerg"], default="all")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    event_table = build_event_table(df, feature_cols)
    ds_train = event_table.build_dataset(
        split="train", window=args.window, max_samples=args.max_train_samples,
        shuffle=True, seed=args.seed,
    )
    ds_val = event_table.build_dataset(
        split="val", window=args.window, max_samples=args.max_val_samples,
        shuffle=False, seed=args.seed,
    )
    ds_test = event_table.build_dataset(
        split="test", window=args.window, max_samples=args.max_test_samples,
        shuffle=False, seed=args.seed,
    )

    loader_train = make_loaders(ds_train, args.bs, shuffle=True, workers=args.workers)
    loader_val = make_loaders(ds_val, args.bs, shuffle=False, workers=args.workers)
    loader_test = make_loaders(ds_test, args.bs, shuffle=False, workers=args.workers)
    total_train_batches = len(loader_train)
    total_val_batches = len(loader_val)
    total_test_batches = len(loader_test)

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
    model = HierGRU(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        model_name=args.model_name,
        num_families=num_families,
        fine_dims=fine_dims,
        num_exact_classes=num_exact_classes,
        use_exact_head=use_exact_head,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── LR Schedule ──────────────────────────────────────────────────────
    scheduler = None
    if args.lr_schedule == "cosine":
        effective_train_batches = total_train_batches
        if args.max_train_batches > 0:
            effective_train_batches = min(total_train_batches, args.max_train_batches)
        total_steps = effective_train_batches * args.epochs
        warmup_steps = max(1, int(args.warmup_fraction * total_steps))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=total_steps,
            pct_start=args.warmup_fraction, anneal_strategy="cos",
            div_factor=25.0, final_div_factor=1e4,
        )

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

    # ── Config ───────────────────────────────────────────────────────────
    config = vars(args).copy()
    config.update({
        "mode": "centralized",
        "architecture": args.model_name,
        "race": args.race,
        "n_features": len(feature_cols),
        **action_context_meta,
        "n_families": num_families,
        "n_exact_classes": num_exact_classes,
        "fine_dims": fine_dims,
        "n_train": len(ds_train), "n_val": len(ds_val), "n_test": len(ds_test),
        "family_id_to_name": {str(k): v for k, v in family_id_to_name.items()},
        "split_mode": _pre.get("split_mode"),
        "effective_split_unit": effective_split_unit,
        **hierarchy_meta,
    })
    (outdir / "config.json").write_text(json.dumps(config, indent=2))

    print(
        "[centralized] "
        f"race={args.race} arch={args.model_name} "
        f"train_samples={len(ds_train)} val_samples={len(ds_val)} test_samples={len(ds_test)} "
        f"train_batches={total_train_batches} val_batches={total_val_batches} test_batches={total_test_batches} "
        f"bs={args.bs} window={args.window}",
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

    # ── Training loop ────────────────────────────────────────────────────
    best_metrics: dict | None = None
    best_epoch: int | None = None
    history = []
    epochs_since_improve = 0
    total_train_time_sec = 0.0
    total_val_time_sec = 0.0
    run_t0 = time.time()
    start_epoch = 1

    checkpoint_path = outdir / "latest_checkpoint.pt"
    metrics_path = outdir / "training_metrics.csv"
    if args.resume and checkpoint_path.exists() and metrics_path.exists():
        print(f"[resume] loading from {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        opt.load_state_dict(checkpoint["opt_state"])
        if scheduler is not None and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        
        history_df = pd.read_csv(metrics_path).fillna(0.0)
        history = history_df.to_dict("records")
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_metrics = checkpoint.get("best_metrics")
        epochs_since_improve = int(checkpoint.get("epochs_since_improve", 0))
        print(f"[resume] starting from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        train_n = 0
        train_coarse_correct = 0
        train_exact_correct = 0
        epoch_train_t0 = time.time()

        for bidx, (x, y_coarse, y_fine, y_exact, lengths) in enumerate(loader_train):
            if args.max_train_batches > 0 and bidx >= args.max_train_batches:
                break
            x, y_coarse, y_fine, y_exact, lengths = (
                x.to(device), y_coarse.to(device), y_fine.to(device), y_exact.to(device), lengths.to(device),
            )

            model_out = model(x, lengths)
            if use_exact_head:
                coarse_logits, fine_logits, exact_logits = model_out
            else:
                coarse_logits, fine_logits = model_out
                exact_logits = None
            coarse_loss = F.cross_entropy(
                coarse_logits, y_coarse, weight=coarse_w_t,
                label_smoothing=args.label_smoothing,
            )
            fine_loss = family_fine_loss_mean(fine_logits, y_coarse, y_fine, fine_class_weights)
            loss = args.coarse_loss_weight * coarse_loss + args.fine_loss_weight * fine_loss
            if exact_logits is not None and args.exact_loss_weight > 0:
                exact_loss = F.cross_entropy(exact_logits, y_exact)
                loss = loss + args.exact_loss_weight * exact_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            if scheduler is not None:
                scheduler.step()

            n = int(y_coarse.numel())
            train_n += n
            train_loss_sum += float(loss.item()) * n
            coarse_pred = coarse_logits.argmax(dim=1)
            pred_fine_pred = torch.zeros_like(y_fine)
            for fam_id, fam_logits in enumerate(fine_logits):
                mask_pred = coarse_pred == fam_id
                if bool(mask_pred.any()):
                    pred_fine_pred[mask_pred] = fam_logits[mask_pred].argmax(dim=1)
            pred_exact = _decode_exact_predictions(
                coarse_pred,
                pred_fine_pred,
                family_fine_to_exact_id=family_fine_to_exact_id,
                default_exact_id=default_exact_id,
                num_exact_classes=num_exact_classes,
            )
            batch_coarse_correct = int((coarse_pred == y_coarse).sum().item())
            batch_exact_correct = int((pred_exact == y_exact).sum().item())
            train_coarse_correct += batch_coarse_correct
            train_exact_correct += batch_exact_correct

            if args.log_every_batches > 0 and ((bidx + 1) % args.log_every_batches == 0):
                elapsed = float(time.time() - epoch_train_t0)
                batches_done = int(bidx + 1)
                samples_per_sec = float(train_n / elapsed) if elapsed > 0 else None
                batch_coarse_top1 = float(batch_coarse_correct / max(1, n))
                batch_exact_top1 = float(batch_exact_correct / max(1, n))
                train_coarse_top1 = float(train_coarse_correct / max(1, train_n))
                train_exact_top1 = float(train_exact_correct / max(1, train_n))
                print(
                    f"[centralized] epoch={epoch} batch={batches_done}/{total_train_batches} "
                    f"train_loss={train_loss_sum / max(1, train_n):.4f} "
                    f"batch_coarse_top1={batch_coarse_top1:.4f} "
                    f"batch_exact_top1={batch_exact_top1:.4f} "
                    f"train_coarse_top1={train_coarse_top1:.4f} "
                    f"train_exact_top1={train_exact_top1:.4f} "
                    f"samples={train_n} "
                    f"samples_per_sec={_fmt_metric(samples_per_sec)}",
                    flush=True,
                )

        train_time_sec = float(time.time() - t0)
        total_train_time_sec += train_time_sec
        val_t0 = time.time()
        val_m = evaluate_hier(model, loader_val, device, **eval_kwargs)
        assert_runtime_coarse_not_inflated(val_m, outdir=outdir, phase="validation_epoch", step=epoch)
        val_time_sec = float(time.time() - val_t0)
        total_val_time_sec += val_time_sec

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss_sum / max(1, train_n)),
            "train_coarse_top1": float(train_coarse_correct / max(1, train_n)),
            "train_exact_top1": float(train_exact_correct / max(1, train_n)),
            "epoch_time_sec": float(train_time_sec + val_time_sec),
            "train_time_sec": train_time_sec,
            "val_time_sec": val_time_sec,
            "train_examples": int(train_n),
            "val_examples": int(val_m.get("n_examples") or 0),
            "train_examples_per_sec": float(train_n / train_time_sec) if train_time_sec > 0 else None,
            "val_examples_per_sec": float((val_m.get("n_examples") or 0) / val_time_sec) if val_time_sec > 0 else None,
            "val_loss": val_m.get("loss"),
            "val_coarse_top1": val_m.get("coarse_top1"),
            "val_coarse_top5": val_m.get("coarse_top5"),
            "val_coarse_f1_macro": val_m.get("coarse_f1_macro"),
            "val_coarse_balanced_accuracy": val_m.get("coarse_balanced_accuracy"),
            "val_coarse_precision_macro": val_m.get("coarse_precision_macro"),
            "val_coarse_recall_macro": val_m.get("coarse_recall_macro"),
            "val_fine_top1_true_family": val_m.get("fine_top1_true_family"),
            "val_exact_top1": val_m.get("exact_top1"),
            "val_exact_f1_macro": val_m.get("exact_f1_macro"),
            "val_exact_balanced_accuracy": val_m.get("exact_balanced_accuracy"),
            "val_exact_precision_macro": val_m.get("exact_precision_macro"),
            "val_exact_recall_macro": val_m.get("exact_recall_macro"),
            "val_exact_kappa": val_m.get("exact_kappa"),
            "val_exact_mcc": val_m.get("exact_mcc"),
            "val_exact_top1_direct": val_m.get("exact_top1_direct"),
            "val_exact_f1_macro_direct": val_m.get("exact_f1_macro_direct"),
            "val_exact_balanced_accuracy_direct": val_m.get("exact_balanced_accuracy_direct"),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(outdir / "training_metrics.csv", index=False)

        improved = is_better(
            val_m,
            best_metrics,
            primary=selection_primary,
            tiebreakers=selection_tiebreakers,
        )
        if improved:
            best_metrics = dict(val_m)
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(model.state_dict(), outdir / "best_model.pt")
        else:
            epochs_since_improve += 1

        direct_str = f" val_exact_direct={_fmt_metric(row.get('val_exact_top1_direct'))}" if use_exact_head else ""
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"train_coarse_top1={row['train_coarse_top1']:.4f} "
            f"train_exact_top1={row['train_exact_top1']:.4f} "
            f"val_coarse_top1={row['val_coarse_top1']:.4f} "
            f"val_coarse_top5={row['val_coarse_top5']:.4f} "
            f"val_exact_top1={row['val_exact_top1']:.4f} "
            f"val_f1={row['val_exact_f1_macro']:.4f} "
            f"val_kappa={row['val_exact_kappa']}"
            + direct_str
            + (" *" if improved else ""),
            flush=True,
        )

        # Checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "epochs_since_improve": epochs_since_improve,
        }, checkpoint_path)

        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (
            args.early_stop_patience > 0
            and epoch >= args.early_stop_min_epochs
            and epochs_since_improve >= args.early_stop_patience
        ):
            print(f"[early-stop] no improvement for {epochs_since_improve} epochs.", flush=True)
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(outdir / "training_metrics.csv", index=False)
    plot_training_curves(history_df, outdir)

    # ── Final test ───────────────────────────────────────────────────────
    if (outdir / "best_model.pt").exists():
        model.load_state_dict(torch.load(outdir / "best_model.pt", map_location="cpu", weights_only=True))

    test_t0 = time.time()
    test_m = evaluate_hier(model, loader_test, device, **eval_kwargs)
    assert_runtime_coarse_not_inflated(test_m, outdir=outdir, phase="final_test", step=best_epoch)
    test_time_sec = float(time.time() - test_t0)
    test_json = metrics_for_json(test_m)
    total_wall_clock_sec = float(time.time() - run_t0)
    test_json["best_epoch"] = best_epoch
    test_json["mode"] = "centralized"
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
    test_json["avg_epoch_time_sec"] = float(history_df["epoch_time_sec"].mean()) if not history_df.empty else None
    train_examples_total = int(history_df["train_examples"].sum()) if not history_df.empty else 0
    val_examples_total = int(history_df["val_examples"].sum()) if not history_df.empty else 0
    test_json["train_examples_per_sec"] = float(train_examples_total / total_train_time_sec) if total_train_time_sec > 0 else None
    test_json["val_examples_per_sec"] = float(val_examples_total / total_val_time_sec) if total_val_time_sec > 0 else None
    test_json["test_examples_per_sec"] = float((test_m.get("n_examples") or 0) / test_time_sec) if test_time_sec > 0 else None
    (outdir / "final_test.json").write_text(json.dumps(test_json, indent=2))

    save_confusion_csv(test_m["_cm_coarse"], outdir / "test_coarse_confusion.csv")
    save_confusion_csv(test_m["_cm_exact"], outdir / "test_exact_confusion.csv")
    
    # Heatmap visualization
    coarse_names = [family_id_to_name.get(i, f"C{i}") for i in range(num_families)]
    plot_confusion_matrix(test_m["_cm_coarse"], coarse_names, outdir / "test_coarse_confusion.png", "Coarse Action Confusion")
    
    save_per_class_recall(test_m["_cm_coarse"], outdir / "test_coarse_per_class_recall.csv", "coarse_class")
    save_per_class_recall(test_m["_cm_exact"], outdir / "test_exact_per_class_recall.csv", "exact_class")
    save_family_summary(test_m.get("_per_family", []), family_id_to_name, outdir / "test_per_family.csv")

    summary = {
        "best_epoch": best_epoch,
        "mode": "centralized",
        "selection_objective": selection_objective,
        "selection_primary": selection_primary,
        "selection_tiebreakers_effective": selection_tiebreakers,
        "final_test": test_json,
    }
    (outdir / "checkpoint_summary.json").write_text(json.dumps(summary, indent=2))

    # ── Cross-run CSV ────────────────────────────────────────────────────
    cross_csv = Path(args.cross_run_csv)
    cross_csv.parent.mkdir(parents=True, exist_ok=True)
    append_cross_run_row(
        cross_csv,
        mode="centralized",
        dataset_dir=args.dataset_dir,
        n_train=len(ds_train), n_val=len(ds_val), n_test=len(ds_test),
        n_clients=0,
        epochs_or_rounds=args.epochs,
        hidden=args.hidden, layers=args.layers, dropout=args.dropout, window=args.window,
        metrics=test_json,
        best_epoch_or_round=best_epoch,
        outdir=str(outdir),
    )

    print(f"[done] centralized hierarchical GRU. best_epoch={best_epoch}")
    print(
        "  exact_top1="
        f"{_fmt_metric(test_json.get('exact_top1'))}  "
        f"f1_macro={_fmt_metric(test_json.get('exact_f1_macro'))}  "
        f"kappa={_fmt_metric(test_json.get('exact_kappa'))}"
    )


if __name__ == "__main__":
    main()
