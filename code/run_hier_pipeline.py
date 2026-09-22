#!/usr/bin/env python3
"""Run the full hierarchical thesis pipeline end to end.

Stages:
1) prepare.py (if the processed dataset is missing)
2) centralized learning: GRU/LSTM/Transformer × Prot/Terr/Zerg
3) FedAvg: GRU/LSTM/Transformer × Prot/Terr/Zerg
4) FedProx: GRU/LSTM/Transformer × Prot/Terr/Zerg
5) backbone-head FL: GRU/LSTM/Transformer (all races by default)
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
    ap.add_argument("--bs", type=int, default=int(os.environ["BS"]) if "BS" in os.environ else None,
                    help="Batch size. Full profile default: 512; smoke default: 128.")
    ap.add_argument("--mu", type=float, default=float(os.environ.get("MU", "0.01")))
    ap.add_argument("--clients-per-round", type=int, default=int(os.environ["CLIENTS_PER_ROUND"]) if "CLIENTS_PER_ROUND" in os.environ else None,
                    help="Clients sampled per round. Full profile default: 25; smoke default: 50.")
    ap.add_argument("--rounds", type=int, default=int(os.environ["ROUNDS"]) if "ROUNDS" in os.environ else None,
                    help="FedAvg/FedProx rounds. Full profile default: 50; smoke default: 5.")
    ap.add_argument("--backbone-rounds", type=int, default=int(os.environ["BACKBONE_ROUNDS"]) if "BACKBONE_ROUNDS" in os.environ else None,
                    help="Backbone-head rounds. Full profile default: 150; smoke default: 5.")
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "20")))
    ap.add_argument("--selection-objective", default=os.environ.get("SELECTION_OBJECTIVE", "joint_honest"))
    ap.add_argument("--selection-tiebreakers", default=os.environ.get("SELECTION_TIEBREAKERS", "coarse_balanced_accuracy,coarse_f1_macro,exact_top1"))
    ap.add_argument("--action-context-features", choices=["on", "off"], default=os.environ.get("ACTION_CONTEXT_FEATURES", "on"))
    ap.add_argument("--window-candidates", default=os.environ.get("WINDOW_CANDIDATES", "4,8,16"))
    ap.add_argument("--max-train", type=int, default=int(os.environ.get("MAX_TRAIN", "0")))
    ap.add_argument("--max-val", type=int, default=int(os.environ.get("MAX_VAL", "0")))
    ap.add_argument("--max-test", type=int, default=int(os.environ.get("MAX_TEST", "0")))
    ap.add_argument("--max-local-batches", type=int, default=int(os.environ["MAX_LOCAL_BATCHES"]) if "MAX_LOCAL_BATCHES" in os.environ else None,
                    help="Maximum local batches per selected client. Full profile default: 75; smoke default: 5.")
    ap.add_argument("--max-eval-batches", type=int, default=int(os.environ.get("MAX_EVAL_BATCHES", "0")))
    ap.add_argument("--max-client-samples", type=int, default=int(os.environ.get("MAX_CLIENT_SAMPLES", "2000")),
                    help="FedAvg/FedProx sample cap per client (paper protocol: 2000).")
    ap.add_argument("--local-weight-decay", type=float, default=float(os.environ.get("LOCAL_WEIGHT_DECAY", "1e-5")),
                    help="FedAvg/FedProx local weight decay (paper protocol: 1e-5).")
    ap.add_argument("--backbone-max-client-samples", type=int, default=int(os.environ.get("BACKBONE_MAX_CLIENT_SAMPLES", "0")),
                    help="Backbone-head sample cap (saved runs: 0, meaning no sample cap).")
    ap.add_argument("--backbone-local-weight-decay", type=float, default=float(os.environ.get("BACKBONE_LOCAL_WEIGHT_DECAY", "0")),
                    help="Backbone-head local weight decay (saved runs: 0).")
    ap.add_argument("--central-weight-decay", type=float, default=float(os.environ.get("CENTRAL_WEIGHT_DECAY", "1e-5")))
    ap.add_argument("--eval-bs", type=int, default=int(os.environ.get("EVAL_BS", "256")))
    ap.add_argument("--auto-prepare", action="store_true", default=True, help="Prepare dataset if missing.")
    ap.add_argument("--skip-prepare", action="store_true", help="Do not run preprocessing even if dataset is missing.")
    ap.add_argument("--archs", default=os.environ.get("ARCHS", "gru"), help="Comma-separated list of architectures (gru, lstm, transformer).")
    ap.add_argument("--races", default=os.environ.get("RACES", "Prot"), help="Comma-separated list of races (Prot, Terr, Zerg).")
    ap.add_argument("--backbone-races", default=os.environ.get("BACKBONE_RACES", "all"), help="Backbone-Head only: comma-separated races (all, Prot, Terr, Zerg).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Override data-loader workers for every mode.")
    ap.add_argument("--central-workers", type=int, default=None,
                    help="Centralized data-loader workers. Full profile default: 8.")
    ap.add_argument("--federated-workers", type=int, default=None,
                    help="Federated data-loader workers. Full profile default: 0.")
    ap.add_argument("--no-resume", action="store_true", help="Do not resume from latest checkpoint (default: false)")
    ap.add_argument("--early-stop-patience", type=int, default=int(os.environ["EARLY_STOP_PATIENCE"]) if "EARLY_STOP_PATIENCE" in os.environ else None,
                    help="Centralized patience. Saved full runs used 100; smoke uses 8.")
    ap.add_argument("--round-val-clients", type=int, default=int(os.environ["ROUND_VAL_CLIENTS"]) if "ROUND_VAL_CLIENTS" in os.environ else None,
                    help="Backbone-head validation clients per round. Full profile default: 50; smoke default: all.")
    return ap.parse_args()


def resolve_protocol(args: argparse.Namespace) -> dict:
    """Resolve profile defaults to the settings used by the saved paper runs."""
    full = args.profile == "full"
    worker_override = args.workers
    central_workers = worker_override if worker_override is not None else (
        args.central_workers if args.central_workers is not None else (8 if full else 0)
    )
    federated_workers = worker_override if worker_override is not None else (
        args.federated_workers if args.federated_workers is not None else 0
    )
    return {
        "epochs": int(args.epochs if full else 3),
        "fed_rounds": int(args.rounds if args.rounds is not None else (50 if full else 5)),
        "backbone_rounds": int(args.backbone_rounds if args.backbone_rounds is not None else (150 if full else 5)),
        "batch_size": int(args.bs if args.bs is not None else (512 if full else 128)),
        "clients_per_round": int(args.clients_per_round if args.clients_per_round is not None else (25 if full else 50)),
        "max_train": int(args.max_train if full else 500),
        "max_val": int(args.max_val if full else 200),
        "max_test": int(args.max_test if full else 200),
        "max_local_batches": int(args.max_local_batches if args.max_local_batches is not None else (75 if full else 5)),
        "max_eval_batches": int(args.max_eval_batches if full else 5),
        "standard_max_client_samples": int(args.max_client_samples),
        "standard_local_weight_decay": float(args.local_weight_decay),
        "backbone_max_client_samples": int(args.backbone_max_client_samples),
        "backbone_local_weight_decay": float(args.backbone_local_weight_decay),
        "central_weight_decay": float(args.central_weight_decay),
        "eval_bs": int(args.eval_bs),
        "central_workers": int(central_workers),
        "federated_workers": int(federated_workers),
        "early_stop_patience": int(args.early_stop_patience if args.early_stop_patience is not None else (100 if full else 8)),
        "round_val_clients": int(args.round_val_clients if args.round_val_clients is not None else (50 if full else 0)),
    }


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


def write_tuning_summary(outroot: Path, args: argparse.Namespace, protocol: dict) -> None:
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
        "selected_epochs": int(protocol["epochs"]),
        "fedavg_fedprox_rounds": int(protocol["fed_rounds"]),
        "backbone_head_rounds": int(protocol["backbone_rounds"]),
        "selected_batch_size": int(protocol["batch_size"]),
        "rationale": (
            "The full profile reproduces the saved paper-run configurations. Backbone-head "
            "uses its recorded 150-round ablation protocol and recorded local-training exceptions."
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

    protocol = resolve_protocol(args)
    write_tuning_summary(outroot, args, protocol)
    (outroot / "benchmark_config.json").write_text(json.dumps({
        "profile": str(args.profile),
        "split_mode": str(args.split_mode),
        "coarse_taxonomy": str(args.coarse_taxonomy),
        "artifact_taxonomy": artifact_taxonomy_for_request(args.coarse_taxonomy),
        "selection_objective": str(args.selection_objective),
        "selection_tiebreakers": str(args.selection_tiebreakers),
        "coarse_class_weight_mode": str(args.coarse_class_weight_mode),
        "action_context_features": str(args.action_context_features),
        "device": str(args.device),
        "batch_size": int(protocol["batch_size"]),
        "window": int(args.window),
        "centralized": {
            "epochs": int(protocol["epochs"]),
            "weight_decay": float(protocol["central_weight_decay"]),
            "early_stop_patience": int(protocol["early_stop_patience"]),
            "workers": int(protocol["central_workers"]),
        },
        "fedavg_fedprox": {
            "rounds": int(protocol["fed_rounds"]),
            "clients_per_round": int(protocol["clients_per_round"]),
            "max_client_samples": int(protocol["standard_max_client_samples"]),
            "local_weight_decay": float(protocol["standard_local_weight_decay"]),
            "max_local_batches": int(protocol["max_local_batches"]),
            "workers": int(protocol["federated_workers"]),
        },
        "backbone_head": {
            "rounds": int(protocol["backbone_rounds"]),
            "clients_per_round": int(protocol["clients_per_round"]),
            "max_client_samples": int(protocol["backbone_max_client_samples"]),
            "local_weight_decay": float(protocol["backbone_local_weight_decay"]),
            "max_local_batches": int(protocol["max_local_batches"]),
            "round_val_clients": int(protocol["round_val_clients"]),
            "workers": int(protocol["federated_workers"]),
        },
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
        "--max-eval-batches", str(protocol["max_eval_batches"]),
        "--coarse-taxonomy", args.coarse_taxonomy,
        "--coarse-class-weight-mode", args.coarse_class_weight_mode,
        "--action-context-features", args.action_context_features,
        "--selection-objective", args.selection_objective,
        "--selection-tiebreakers", args.selection_tiebreakers,
        "--workers", str(protocol["central_workers"]),
    ]
    if not args.no_resume:
        common.append("--resume")
    common.extend(["--early-stop-patience", str(protocol["early_stop_patience"])])

    central_common = [
        "--epochs", str(protocol["epochs"]),
        "--bs", str(protocol["batch_size"]),
        "--lr", str(args.lr),
        "--weight-decay", str(protocol["central_weight_decay"]),
        "--max-train-samples", str(protocol["max_train"]),
        "--max-val-samples", str(protocol["max_val"]),
        "--max-test-samples", str(protocol["max_test"]),
        "--max-train-batches", "0" if args.profile == "full" else str(protocol["max_local_batches"]),
    ]

    federated_common = [
        "--clients-per-round", str(protocol["clients_per_round"]),
        "--local-bs", str(protocol["batch_size"]),
        "--local-lr", str(args.lr),
        "--local-epochs", "1",
        "--max-local-batches", str(protocol["max_local_batches"]),
        "--eval-bs", str(protocol["eval_bs"]),
        "--max-eval-batches", str(protocol["max_eval_batches"]),
        "--coarse-taxonomy", args.coarse_taxonomy,
        "--coarse-class-weight-mode", args.coarse_class_weight_mode,
        "--action-context-features", args.action_context_features,
        "--selection-objective", args.selection_objective,
        "--selection-tiebreakers", args.selection_tiebreakers,
        "--workers", str(protocol["federated_workers"]),
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
                    "--rounds", str(protocol["fed_rounds"]),
                    "--max-client-samples", str(protocol["standard_max_client_samples"]),
                    "--local-weight-decay", str(protocol["standard_local_weight_decay"]),
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
                    "--rounds", str(protocol["fed_rounds"]),
                    "--max-client-samples", str(protocol["standard_max_client_samples"]),
                    "--local-weight-decay", str(protocol["standard_local_weight_decay"]),
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
                    "--rounds", str(protocol["backbone_rounds"]),
                    "--max-client-samples", str(protocol["backbone_max_client_samples"]),
                    "--local-weight-decay", str(protocol["backbone_local_weight_decay"]),
                    "--round-val-clients", str(protocol["round_val_clients"]),
                    "--outdir", str(outdir),
                ], cwd=repo_root, env=env, outdir=outdir)

    run([
        sys.executable, str(code_dir / "hier_compare.py"),
        "--cross-run-csv", str(cross_csv),
    ], cwd=repo_root, env=env)

    print(f"\n[done] results written to {outroot}")


if __name__ == "__main__":
    main()
