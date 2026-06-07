#!/usr/bin/env python3
"""Run the full hierarchical thesis pipeline end to end.

Stages:
1) prepare.py (if the processed dataset is missing)
2) centralized learning: GRU/LSTM/Transformer × Prot/Terr/Zerg
3) FedAvg: GRU/LSTM/Transformer × Prot/Terr/Zerg
4) FedProx: GRU/LSTM/Transformer × Prot/Terr/Zerg
5) backbone-head race-specific FL: GRU/LSTM/Transformer
6) comparison report
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the full hierarchical SC2EGSet pipeline.")
    ap.add_argument("--root", default=os.environ.get("ROOT", str(Path(__file__).resolve().parent.parent)), help="Repository / dataset root.")
    ap.add_argument("--dataset-dir", default=os.environ.get("DATASET_DIR", ""), help="Directory containing processed_events.parquet etc.")
    ap.add_argument("--outroot", default=os.environ.get("OUTROOT", ""), help="Output root for run artifacts.")
    ap.add_argument("--profile", choices=["smoke", "full"], default=os.environ.get("PROFILE", "full"))
    ap.add_argument("--modes", default=os.environ.get("MODES", "centralized,fedavg,fedprox,backbone_head"), help="Comma-separated list of modes to run.")
    ap.add_argument("--split-mode", choices=["replay", "tournament", "player"], default=os.environ.get("SPLIT_MODE", "replay"))
    ap.add_argument("--coarse-taxonomy", default=os.environ.get("COARSE_TAXONOMY", "auto"))
    ap.add_argument("--coarse-class-weight-mode", choices=["none", "inverse", "inverse_sqrt"], default=os.environ.get("COARSE_CLASS_WEIGHT_MODE", "inverse_sqrt"))
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "123")))
    ap.add_argument("--hidden", type=int, default=int(os.environ.get("HIDDEN", "256")))
    ap.add_argument("--layers", type=int, default=int(os.environ.get("LAYERS", "2")))
    ap.add_argument("--dropout", type=float, default=float(os.environ.get("DROPOUT", "0.2")))
    ap.add_argument("--window", type=int, default=int(os.environ.get("WINDOW", "8")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("LR", "0.001")))
    ap.add_argument("--bs", type=int, default=int(os.environ.get("BS", "128")))
    ap.add_argument("--mu", type=float, default=float(os.environ.get("MU", "0.01")))
    ap.add_argument("--clients-per-round", type=int, default=int(os.environ.get("CLIENTS_PER_ROUND", "50")))
    ap.add_argument("--rounds", type=int, default=int(os.environ.get("ROUNDS", "50")))
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "20")))
    ap.add_argument("--selection-objective", default=os.environ.get("SELECTION_OBJECTIVE", "joint_honest"))
    ap.add_argument("--selection-tiebreakers", default=os.environ.get("SELECTION_TIEBREAKERS", "coarse_balanced_accuracy,coarse_f1_macro,exact_top1"))
    ap.add_argument("--action-context-features", choices=["on", "off"], default=os.environ.get("ACTION_CONTEXT_FEATURES", "on"))
    ap.add_argument("--window-candidates", default=os.environ.get("WINDOW_CANDIDATES", "4,8,16"))
    ap.add_argument("--max-train", type=int, default=int(os.environ.get("MAX_TRAIN", "0")))
    ap.add_argument("--max-val", type=int, default=int(os.environ.get("MAX_VAL", "0")))
    ap.add_argument("--max-test", type=int, default=int(os.environ.get("MAX_TEST", "0")))
    ap.add_argument("--max-local-batches", type=int, default=int(os.environ.get("MAX_LOCAL_BATCHES", "0")))
    ap.add_argument("--max-eval-batches", type=int, default=int(os.environ.get("MAX_EVAL_BATCHES", "0")))
    ap.add_argument("--auto-prepare", action="store_true", default=True, help="Prepare dataset if missing.")
    ap.add_argument("--skip-prepare", action="store_true", help="Do not run preprocessing even if dataset is missing.")
    ap.add_argument("--archs", default=os.environ.get("ARCHS", "gru"), help="Comma-separated list of architectures (gru, lstm, transformer).")
    ap.add_argument("--races", default=os.environ.get("RACES", "Prot"), help="Comma-separated list of races (Prot, Terr, Zerg).")
    ap.add_argument("--backbone-races", default=os.environ.get("BACKBONE_RACES", "all"), help="Backbone-Head only: comma-separated races (all, Prot, Terr, Zerg).")
    ap.add_argument("--workers", type=int, default=4, help="Number of dataloader workers (default: 4)")
    ap.add_argument("--no-resume", action="store_true", help="Do not resume from latest checkpoint (default: false)")
    ap.add_argument("--early-stop-patience", type=int, default=int(os.environ.get("EARLY_STOP_PATIENCE", "8")), help="Patience for early stopping in centralized runs.")
    ap.add_argument("--round-val-clients", type=int, default=0, help="Backbone-Head only: number of val clients to sample during rounds.")
    return ap.parse_args()


def run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def run_if_needed(cmd: list[str], *, cwd: Path, env: dict[str, str], outdir: Path) -> None:
    final_json = outdir / "final_test.json"
    if final_json.exists():
        print(f"\n[skip] completed run exists: {outdir}", flush=True)
        return
    run(cmd, cwd=cwd, env=env)


def dataset_ready(dataset_dir: Path) -> bool:
    required = [
        dataset_dir / "processed_events.parquet",
        dataset_dir / "preprocessing.json",
        dataset_dir / "action_vocab.json",
    ]
    return all(p.exists() for p in required)


def artifact_taxonomy_for_request(taxonomy: str) -> str:
    requested = str(taxonomy or "").strip().lower()
    if requested in {"", "auto", "legacy", "legacy8"}:
        return "legacy8"
    return requested


def dataset_matches_primary_config(dataset_dir: Path, *, split_mode: str, taxonomy: str) -> tuple[bool, list[str]]:
    if not dataset_ready(dataset_dir):
        return False, ["dataset files missing"]
    reasons: list[str] = []
    expected_taxonomy = artifact_taxonomy_for_request(taxonomy)
    try:
        preprocessing = json.loads((dataset_dir / "preprocessing.json").read_text())
    except Exception as exc:
        return False, [f"cannot read preprocessing.json: {exc}"]
    try:
        vocab = json.loads((dataset_dir / "action_vocab.json").read_text())
    except Exception as exc:
        return False, [f"cannot read action_vocab.json: {exc}"]

    actual_split = str(preprocessing.get("split_mode", "")).strip().lower()
    if actual_split != str(split_mode).strip().lower():
        reasons.append(f"split_mode={actual_split or '<missing>'}, expected={split_mode}")

    hierarchy = vocab.get("hierarchy") or {}
    actual_taxonomy = str(
        hierarchy.get("hierarchy_taxonomy") or preprocessing.get("hierarchy_taxonomy") or ""
    ).strip().lower()
    if actual_taxonomy != expected_taxonomy:
        reasons.append(f"hierarchy_taxonomy={actual_taxonomy or '<missing>'}, expected={expected_taxonomy}")

    return not reasons, reasons


def preflight(dataset_dir: Path) -> None:
    required_modules = ["numpy", "pandas", "pyarrow", "torch", "sklearn", "matplotlib"]
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"Missing Python modules: {', '.join(missing)}")
    if not dataset_ready(dataset_dir):
        return
    required = ["processed_events.parquet", "preprocessing.json", "action_vocab.json"]
    missing_files = [name for name in required if not (dataset_dir / name).exists()]
    if missing_files:
        raise SystemExit(f"Dataset is missing required files in {dataset_dir}: {', '.join(missing_files)}")


def write_tuning_summary(outroot: Path, args: argparse.Namespace, profile_defaults: dict) -> None:
    try:
        window_candidates = [int(x.strip()) for x in str(args.window_candidates).split(",") if x.strip()]
    except Exception:
        window_candidates = [int(args.window)]
    summary = {
        "candidate_windows": window_candidates,
        "selected_window": int(args.window),
        "selection_objective": str(args.selection_objective),
        "selection_tiebreakers": str(args.selection_tiebreakers),
        "action_context_features": str(args.action_context_features),
        "profile": str(args.profile),
        "selected_epochs": int(profile_defaults["epochs"]),
        "selected_rounds": int(profile_defaults["rounds"]),
        "selected_batch_size": int(args.bs),
        "rationale": (
            "Use one shared window and one comparable training budget across centralized and "
            "federated approaches so thesis comparisons stay aligned."
        ),
    }
    (outroot / "tuning_summary.json").write_text(json.dumps(summary, indent=2))


def prepare_dataset(code_dir: Path, repo_root: Path, dataset_dir: Path, taxonomy: str, profile: str, split_mode: str) -> None:
    artifact_taxonomy = artifact_taxonomy_for_request(taxonomy)
    cmd = [
        sys.executable,
        str(code_dir / "prepare.py"),
        "--root", str(repo_root),
        "--outdir", str(dataset_dir),
        "--hierarchy-taxonomy", artifact_taxonomy,
        "--split-mode", str(split_mode),
    ]
    if profile == "smoke":
        cmd.append("--smoke")
    run(cmd, cwd=repo_root, env=os.environ.copy())


def main() -> None:
    args = parse_args()
    repo_root = Path(args.root).resolve()
    code_dir = repo_root / "code" if (repo_root / "code").exists() else repo_root
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else (repo_root / ("artifacts_smoke" if args.profile == "smoke" else "artifacts")).resolve()
    outroot = Path(args.outroot).resolve() if args.outroot else (repo_root / "runs" / f"hier_pipeline_{args.profile}").resolve()
    outroot.mkdir(parents=True, exist_ok=True)
    cross_csv = outroot / "cross_run_results.csv"

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(code_dir))
    env.setdefault("XDG_CACHE_HOME", str(repo_root / ".cache"))
    env.setdefault("MPLCONFIGDIR", str(repo_root / ".cache" / "matplotlib"))
    env.setdefault("OMP_NUM_THREADS", env.get("OMP_NUM_THREADS", "1"))
    env.setdefault("MKL_NUM_THREADS", env.get("MKL_NUM_THREADS", "1"))
    env.setdefault("STRICT_COARSE_AUDIT", os.environ.get("STRICT_COARSE_AUDIT", "1"))
    env.setdefault("COARSE_INFLATION_RECTIFY", os.environ.get("COARSE_INFLATION_RECTIFY", "1"))

    preflight(dataset_dir)

    if not dataset_ready(dataset_dir):
        if args.skip_prepare:
            raise SystemExit(f"Dataset is not ready: {dataset_dir}")
        if not args.auto_prepare:
            raise SystemExit(f"Dataset is not ready: {dataset_dir}; rerun with preprocessing enabled.")
        prepare_dataset(code_dir, repo_root, dataset_dir, args.coarse_taxonomy, args.profile, args.split_mode)
    else:
        matches, mismatch_reasons = dataset_matches_primary_config(
            dataset_dir,
            split_mode=args.split_mode,
            taxonomy=args.coarse_taxonomy,
        )
        if not matches:
            reason_text = "; ".join(mismatch_reasons)
            if args.skip_prepare:
                raise SystemExit(
                    f"Dataset at {dataset_dir} does not match the primary thesis config: {reason_text}"
                )
            if not args.auto_prepare:
                raise SystemExit(
                    f"Dataset at {dataset_dir} does not match the primary thesis config: {reason_text}; "
                    "rerun with preprocessing enabled."
                )
            print(f"[preflight] dataset config mismatch ({reason_text}); rebuilding artifacts...", flush=True)
            prepare_dataset(code_dir, repo_root, dataset_dir, args.coarse_taxonomy, args.profile, args.split_mode)

    profile_defaults = {
        "smoke": {"epochs": 3, "rounds": 5, "max_train": 500, "max_val": 200, "max_test": 200, "max_local_batches": 5, "max_eval_batches": 5},
        "full": {"epochs": args.epochs, "rounds": args.rounds, "max_train": args.max_train, "max_val": args.max_val, "max_test": args.max_test, "max_local_batches": args.max_local_batches, "max_eval_batches": args.max_eval_batches},
    }[args.profile]
    write_tuning_summary(outroot, args, profile_defaults)
    (outroot / "benchmark_config.json").write_text(json.dumps({
        "profile": str(args.profile),
        "split_mode": str(args.split_mode),
        "coarse_taxonomy": str(args.coarse_taxonomy),
        "artifact_taxonomy": artifact_taxonomy_for_request(args.coarse_taxonomy),
        "selection_objective": str(args.selection_objective),
        "selection_tiebreakers": str(args.selection_tiebreakers),
        "coarse_class_weight_mode": str(args.coarse_class_weight_mode),
        "action_context_features": str(args.action_context_features),
        "batch_size": int(args.bs),
        "window": int(args.window),
        "epochs": int(profile_defaults["epochs"]),
        "rounds": int(profile_defaults["rounds"]),
        "strict_coarse_audit": str(env.get("STRICT_COARSE_AUDIT", "1")),
        "coarse_inflation_rectify": str(env.get("COARSE_INFLATION_RECTIFY", "1")),
    }, indent=2))

    common = [
        sys.executable,
        str(code_dir / "hier_centralized.py"),
        "--dataset-dir", str(dataset_dir),
        "--cross-run-csv", str(cross_csv),
        "--window", str(args.window),
        "--hidden", str(args.hidden),
        "--layers", str(args.layers),
        "--dropout", str(args.dropout),
        "--seed", str(args.seed),
        "--device", args.device,
        "--max-eval-batches", str(profile_defaults["max_eval_batches"]),
        "--coarse-taxonomy", args.coarse_taxonomy,
        "--coarse-class-weight-mode", args.coarse_class_weight_mode,
        "--action-context-features", args.action_context_features,
        "--selection-objective", args.selection_objective,
        "--selection-tiebreakers", args.selection_tiebreakers,
        "--workers", str(args.workers),
    ]
    if not args.no_resume:
        common.append("--resume")
    common.extend(["--early-stop-patience", str(args.early_stop_patience)])

    central_common = [
        "--epochs", str(profile_defaults["epochs"]),
        "--bs", str(args.bs),
        "--lr", str(args.lr),
        "--max-train-samples", str(profile_defaults["max_train"]),
        "--max-val-samples", str(profile_defaults["max_val"]),
        "--max-test-samples", str(profile_defaults["max_test"]),
        "--max-train-batches", str(profile_defaults["max_local_batches"]),
    ]

    federated_common = [
        "--rounds", str(profile_defaults["rounds"]),
        "--clients-per-round", str(args.clients_per_round),
        "--local-bs", str(args.bs),
        "--local-lr", str(args.lr),
        "--local-epochs", "1",
        "--max-local-batches", str(profile_defaults["max_local_batches"]),
        "--max-eval-batches", str(profile_defaults["max_eval_batches"]),
        "--coarse-taxonomy", args.coarse_taxonomy,
        "--coarse-class-weight-mode", args.coarse_class_weight_mode,
        "--action-context-features", args.action_context_features,
        "--selection-objective", args.selection_objective,
        "--selection-tiebreakers", args.selection_tiebreakers,
        "--workers", str(args.workers),
    ]
    if not args.no_resume:
        federated_common.append("--resume")

    races = [r.strip() for r in args.races.split(",") if r.strip()]
    backbone_races = [r.strip() for r in args.backbone_races.split(",") if r.strip()]
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    modes_to_run = [m.strip().lower() for m in args.modes.split(",") if m.strip()]

    if "centralized" in modes_to_run:
        for arch in archs:
            for race in races:
                outdir = outroot / f"centralized_{arch}_{race.lower()}"
                run_if_needed([
                    sys.executable, str(code_dir / "hier_centralized.py"),
                    *common[2:],
                    *central_common,
                    "--outdir", str(outdir),
                    "--model-name", arch,
                    "--race", race,
                ], cwd=repo_root, env=env, outdir=outdir)

    if "fedavg" in modes_to_run:
        for arch in archs:
            for race in races:
                outdir = outroot / f"fedavg_{arch}_{race.lower()}"
                run_if_needed([
                    sys.executable, str(code_dir / "hier_fedavg.py"),
                    "--dataset-dir", str(dataset_dir),
                    "--cross-run-csv", str(cross_csv),
                    "--window", str(args.window),
                    "--hidden", str(args.hidden),
                    "--layers", str(args.layers),
                    "--dropout", str(args.dropout),
                    "--seed", str(args.seed),
                    "--device", args.device,
                    "--model-name", arch,
                    "--race", race,
                    *federated_common,
                    "--outdir", str(outdir),
                ], cwd=repo_root, env=env, outdir=outdir)

    if "fedprox" in modes_to_run:
        for arch in archs:
            for race in races:
                outdir = outroot / f"fedprox_{arch}_{race.lower()}"
                run_if_needed([
                    sys.executable, str(code_dir / "hier_fedprox.py"),
                    "--dataset-dir", str(dataset_dir),
                    "--cross-run-csv", str(cross_csv),
                    "--window", str(args.window),
                    "--hidden", str(args.hidden),
                    "--layers", str(args.layers),
                    "--dropout", str(args.dropout),
                    "--seed", str(args.seed),
                    "--device", args.device,
                    "--model-name", arch,
                    "--race", race,
                    *federated_common,
                    "--mu", str(args.mu),
                    "--outdir", str(outdir),
                ], cwd=repo_root, env=env, outdir=outdir)

    if "backbone_head" in modes_to_run:
        for arch in archs:
            for race in backbone_races:
                race_suffix = "" if race == "all" else f"_{race.lower()}"
                outdir = outroot / f"backbone_head_{arch}{race_suffix}"
                run_if_needed([
                    sys.executable, str(code_dir / "hier_backbone_head_race.py"),
                    "--dataset-dir", str(dataset_dir),
                    "--cross-run-csv", str(cross_csv),
                    "--window", str(args.window),
                    "--hidden", str(args.hidden),
                    "--layers", str(args.layers),
                    "--dropout", str(args.dropout),
                    "--seed", str(args.seed),
                    "--device", args.device,
                    "--model-name", arch,
                    "--race", race,
                    *federated_common,
                    "--round-val-clients", str(args.round_val_clients),
                    "--outdir", str(outdir),
                ], cwd=repo_root, env=env, outdir=outdir)

    run([
        sys.executable, str(code_dir / "hier_compare.py"),
        "--cross-run-csv", str(cross_csv),
    ], cwd=repo_root, env=env)

    print(f"\n[done] results written to {outroot}")


if __name__ == "__main__":
    main()
