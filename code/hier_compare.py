#!/usr/bin/env python3
"""Cross-run comparison: reads final_test.json from multiple run directories
and produces a summary table (CSV + formatted stdout)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compile comparison table from hier pipeline run directories.",
    )
    ap.add_argument(
        "--run-dirs", nargs="+", default=[],
        help="Directories containing final_test.json output.",
    )
    ap.add_argument(
        "--cross-run-csv", default="",
        help="Optionally, path to cross_run_results.csv to display directly.",
    )
    ap.add_argument("--output", default="", help="Path to save comparison CSV.")
    ap.add_argument("--verbose", action="store_true", help="Show per-family recall summary.")
    return ap.parse_args()


DISPLAY_COLS = [
    "mode", "architecture", "race", "taxonomy", "split_mode",
    "n_examples",
    "coarse_top1", "coarse_top5",
    "coarse_precision_macro", "coarse_recall_macro",
    "coarse_majority_baseline_top1", "coarse_active_classes",
    "fine_top1_true_family", "fine_top1_conditional_correct_coarse",
    "exact_top1", "exact_f1_macro", "exact_f1_weighted",
    "exact_balanced_accuracy", "exact_kappa", "exact_mcc",
    "exact_precision_macro", "exact_recall_macro",
    "exact_top1_ex_dominant", "exact_majority_baseline_top1", "exact_active_classes",
    "train_wall_clock_sec", "test_wall_clock_sec", "total_wall_clock_sec",
    "avg_epoch_time_sec", "avg_round_time_sec",
    "comm_bytes_total", "comm_bytes_per_round_avg",
    "majority_baseline",
    "selection_objective",
    "best_epoch", "best_round",
]


def _enrich_from_config(data: dict, rd_path: Path) -> None:
    """Try to read config.json and add architecture/taxonomy columns."""
    cfg_path = rd_path / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            if "architecture" not in data:
                data["architecture"] = cfg.get("architecture") or cfg.get("mode", "")
            if "race" not in data:
                data["race"] = cfg.get("race", "all")
            if "taxonomy" not in data:
                data["taxonomy"] = cfg.get("effective_coarse_taxonomy") or cfg.get("coarse_taxonomy", "")
            if "split_mode" not in data:
                data["split_mode"] = cfg.get("split_mode")
        except Exception:
            pass


def main() -> None:
    args = parse_args()

    # ── From cross_run_results.csv ───────────────────────────────────────
    if args.cross_run_csv and Path(args.cross_run_csv).exists():
        df = pd.read_csv(args.cross_run_csv)
        print("\n=== Cross-Run Results ===")
        print(df.to_string(index=False))
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\n[saved] {args.output}")
        return

    # ── From individual run directories ──────────────────────────────────
    rows = []
    for rd in args.run_dirs:
        rd_path = Path(rd)
        test_path = rd_path / "final_test.json"
        if not test_path.exists():
            print(f"[skip] {rd}: no final_test.json")
            continue
        data = json.loads(test_path.read_text())
        data["outdir"] = str(rd_path)

        # Enrich with config metadata
        _enrich_from_config(data, rd_path)

        rows.append(data)

    if not rows:
        print("No valid runs found.")
        return

    df = pd.DataFrame(rows)
    # select display columns that exist
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    display_df = df[cols]

    print("\n=== Hierarchical Pipeline Comparison ===")
    print(display_df.to_string(index=False))

    if args.verbose:
        # Show per-family info from each run
        for rd in args.run_dirs:
            rd_path = Path(rd)
            pf_path = rd_path / "test_per_family.csv"
            if pf_path.exists():
                pf_df = pd.read_csv(pf_path)
                print(f"\n--- Per-family: {rd_path.name} ---")
                print(pf_df.to_string(index=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
