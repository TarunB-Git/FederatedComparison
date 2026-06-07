#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import hierarchy as sc2_hier

_TMP_CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")) / "sc2egset_cache"
_TMP_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMP_CACHE_ROOT))

RACE_ABBR = {"Prot": "P", "Terr": "T", "Zerg": "Z"}

def split_for_unit(unit_id: str, seed: int, split_mode: str = "replay") -> str:
    """Deterministically map a unit identifier to train/val/test.

    split_mode:
        replay     -> split by replay id (legacy behavior)
        tournament -> split by tournament/event directory
        player     -> split by player id if available, else replay id
    """
    mode = str(split_mode).strip().lower()
    key = str(unit_id)
    digest = hashlib.sha1(f"{seed}:{mode}:{key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def build_split_lookup(unit_ids: list[str], seed: int) -> dict[str, str]:
    """Deterministically partition units into train/val/test with coverage."""
    ordered = sorted(str(x) for x in unit_ids if str(x))
    n = len(ordered)
    if n <= 0:
        return {}
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)

    n_val = int(round(0.15 * n))
    n_test = int(round(0.15 * n))
    if n >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
    while n_val + n_test >= n and n > 1:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break
    n_train = max(1, n - n_val - n_test)
    n_val = max(0, n - n_train - n_test)

    lookup: dict[str, str] = {}
    for rank, idx in enumerate(perm.tolist()):
        unit = ordered[int(idx)]
        if rank < n_train:
            lookup[unit] = "train"
        elif rank < n_train + n_val:
            lookup[unit] = "val"
        else:
            lookup[unit] = "test"
    return lookup

def safe_int(v, default: int = 0) -> int:
    try: return int(v)
    except Exception: return int(default)

def safe_float(v, default: float = 0.0) -> float:
    try: return float(v)
    except Exception: return float(default)

def read_json_from_zip(zf: zipfile.ZipFile, member: str) -> dict:
    return json.loads(zf.read(member))

def find_tournament_assets(root_dir: Path) -> tuple[Path, Path | None, Path | None]:
    data_zip = None
    mapping_json = None
    summary_json = None
    for p in root_dir.iterdir():
        if p.is_file() and p.name.endswith("_data.zip"): data_zip = p
        elif p.is_file() and p.name.endswith("_processed_mapping.json"): mapping_json = p
        elif p.is_file() and p.name.endswith("_summary.json"): summary_json = p
    if data_zip is None:
        raise FileNotFoundError(f"Could not find *_data.zip in {root_dir}")
    return data_zip, mapping_json, summary_json

def load_mapping(mapping_json: Path | None) -> dict[str, str]:
    if mapping_json is None or not mapping_json.exists(): return {}
    try: return json.loads(mapping_json.read_text())
    except Exception: return {}

def infer_matchup(player_race: str, opp_race: str, source_path: str) -> str:
    pa = RACE_ABBR.get(str(player_race), "?")
    oa = RACE_ABBR.get(str(opp_race), "?")
    if pa != "?" and oa != "?": return f"{pa}v{oa}"
    m = re.search(r"\s-\s([PTZ]v[PTZ])\s-\s", source_path)
    return m.group(1) if m else "unknown"


def resolve_split_unit(
    *,
    split_mode: str,
    tournament: str,
    replay_id: str,
    player_toon_id: str,
) -> tuple[str, str]:
    mode = str(split_mode).strip().lower()
    if mode == "tournament":
        return "tournament", str(tournament)
    if mode == "player":
        return "player_toon_id", str(player_toon_id or replay_id)
    return "replay_id", str(replay_id)

def extract_rows_from_replay(obj, tournament, replay_member, source_path, race_filter, split_seed, split_mode, label_mode, dedupe_gap, split_lookup=None):
    replay_id = replay_member.replace(".SC2Replay.json", "")
    toon_desc = obj.get("ToonPlayerDescMap", {}) if isinstance(obj.get("ToonPlayerDescMap"), dict) else {}

    toons = []
    user_to_toon, user_to_player, player_to_toon = {}, {}, {}
    for toon_id, desc in toon_desc.items():
        if not isinstance(desc, dict): continue
        pid, uid = safe_int(desc.get("playerID"), -1), safe_int(desc.get("userID"), -1)
        entry = {"toon_id": str(toon_id), "player_id": pid, "user_id": uid, "race": str(desc.get("race", ""))}
        toons.append(entry)
        if uid >= 0: user_to_toon[uid] = str(toon_id); user_to_player[uid] = pid
        if pid >= 0: player_to_toon[pid] = str(toon_id)

    stats_by_pid = defaultdict(list)
    tracker_events = obj.get("trackerEvents", []) if isinstance(obj.get("trackerEvents"), list) else []
    for te in tracker_events:
        if te.get("evtTypeName") == "PlayerStats":
            pid, loop = safe_int(te.get("playerId"), -1), safe_int(te.get("loop"), 0)
            stats = te.get("stats") if isinstance(te.get("stats"), dict) else {}
            if pid >= 0: stats_by_pid[pid].append((loop, stats))
    for pid in stats_by_pid: stats_by_pid[pid].sort(key=lambda x: x[0])

    cmds_by_toon = defaultdict(list)
    game_events = obj.get("gameEvents", []) if isinstance(obj.get("gameEvents"), list) else []
    for ge in game_events:
        if ge.get("evtTypeName") == "Cmd":
            abil = ge.get("abil")
            if not isinstance(abil, dict): continue
            link, idx = abil.get("abilLink"), abil.get("abilCmdIndex")
            if link is None or idx is None: continue
            uid = safe_int((ge.get("userid") or {}).get("userId"), -1)
            toon_id = user_to_toon.get(uid) or player_to_toon.get(user_to_player.get(uid, -1))
            if toon_id: cmds_by_toon[toon_id].append((safe_int(ge.get("loop"), 0), safe_int(link), safe_int(idx)))
    for toon_id in cmds_by_toon: cmds_by_toon[toon_id].sort()

    event_rows, manifest_rows = [], []
    for toon in toons:
        toon_id, player_race = toon["toon_id"], toon.get("race", "")
        if race_filter != "all" and player_race != race_filter: continue
        opp = next((o for o in toons if o["toon_id"] != toon_id), None)
        opp_race = opp.get("race", "") if opp else ""
        split_unit_kind, split_unit_id = resolve_split_unit(
            split_mode=split_mode,
            tournament=tournament,
            replay_id=replay_id,
            player_toon_id=toon_id,
        )
        if split_lookup is not None and split_unit_id in split_lookup:
            split = str(split_lookup[split_unit_id])
        else:
            split = split_for_unit(split_unit_id, split_seed, split_mode)

        action_events = cmds_by_toon.get(toon_id, [])
        pid = safe_int(toon.get("player_id"), -1)
        pstats = stats_by_pid.get(pid, [])

        manifest_rows.append({
            "tournament": tournament,
            "replay_id": replay_id,
            "split": split,
            "split_mode": str(split_mode),
            "split_unit_kind": split_unit_kind,
            "split_unit_id": split_unit_id,
            "player_toon_id": toon_id,
            "player_id": pid,
            "player_user_id": safe_int(toon.get("user_id"), -1),
            "player_race": player_race,
            "opponent_race": opp_race,
            "matchup": infer_matchup(player_race, opp_race, source_path),
        })

        if not action_events: continue
        stats_ptr = 0
        prev_loop = None
        for event_idx, (loop, link, idx) in enumerate(action_events):
            while stats_ptr + 1 < len(pstats) and pstats[stats_ptr + 1][0] <= loop: stats_ptr += 1
            stats_snap = pstats[stats_ptr][1] if pstats and pstats[stats_ptr][0] <= loop else {}
            delta_prev = 0 if prev_loop is None else max(0, loop - prev_loop)
            prev_loop = loop
            action_key = f"{link}" if label_mode == "abil_link" else f"{link}:{idx}"
            event_rows.append({"tournament": tournament, "replay_id": replay_id, "split": split, "group_id": f"{replay_id}::{toon_id}", "event_loop": loop, "delta_prev_action_loops": delta_prev, "action_link": link, "action_cmd_idx": idx, "action_key": action_key, "stats_snapshot": stats_snap})

    return event_rows, manifest_rows


def _pairwise_overlap(values_by_split: dict[str, set[str]]) -> dict[str, int]:
    splits = ["train", "val", "test"]
    out: dict[str, int] = {}
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            out[f"{left}__{right}"] = int(len(values_by_split.get(left, set()) & values_by_split.get(right, set())))
    return out


def _entropy_and_majority(counts: dict[str, int]) -> tuple[float | None, float | None]:
    values = np.asarray(list(counts.values()), dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return None, None
    probs = values / total
    majority = float(probs.max()) if probs.size else None
    probs = probs[probs > 0]
    if probs.size <= 1:
        return 0.0, majority
    entropy = float(-(probs * np.log2(probs)).sum())
    max_entropy = float(np.log2(probs.size))
    return (float(entropy / max_entropy) if max_entropy > 0 else 0.0), majority


def build_split_audit(df_manifest: pd.DataFrame, split_mode: str) -> dict:
    split_values = {}
    for key in ["replay_id", "tournament", "player_toon_id", "split_unit_id"]:
        split_values[key] = {
            split: set(
                df_manifest.loc[df_manifest["split"] == split, key].astype(str).dropna().unique().tolist()
            )
            for split in ["train", "val", "test"]
        }
    return {
        "split_mode_requested": str(split_mode),
        "effective_split_unit": str(df_manifest["split_unit_kind"].iloc[0]) if not df_manifest.empty else None,
        "leakage_overlaps": {
            key: _pairwise_overlap(value_sets)
            for key, value_sets in split_values.items()
        },
        "counts_by_split": {
            split: {
                "rows": int((df_manifest["split"] == split).sum()),
                "unique_replays": int(df_manifest.loc[df_manifest["split"] == split, "replay_id"].nunique()),
                "unique_tournaments": int(df_manifest.loc[df_manifest["split"] == split, "tournament"].nunique()),
                "unique_players": int(df_manifest.loc[df_manifest["split"] == split, "player_toon_id"].nunique()),
                "race_counts": {
                    str(k): int(v)
                    for k, v in df_manifest.loc[df_manifest["split"] == split, "player_race"].value_counts().items()
                },
            }
            for split in ["train", "val", "test"]
        },
    }


def build_hierarchy_audit(
    coarse_counts_total: dict[str, int],
    coarse_counts_by_race: dict[str, dict[str, int]],
) -> dict:
    def _summarize(counts: dict[str, int]) -> dict:
        entropy, majority = _entropy_and_majority(counts)
        return {
            "active_coarse_families": int(sum(1 for v in counts.values() if int(v) > 0)),
            "counts": {str(k): int(v) for k, v in sorted(counts.items())},
            "majority_share": majority,
            "normalized_entropy": entropy,
            "imbalance_severity": (
                "severe" if majority is not None and majority >= 0.60 else
                "moderate" if majority is not None and majority >= 0.40 else
                "mild" if majority is not None else None
            ),
            "trivial_dominance_flag": bool(majority is not None and majority >= 0.75),
        }

    return {
        "overall": _summarize(coarse_counts_total),
        "by_race": {str(race): _summarize(counts) for race, counts in sorted(coarse_counts_by_race.items())},
    }


def assert_hierarchy_is_usable(hierarchy_audit: dict) -> None:
    overall = dict(hierarchy_audit.get("overall") or {})
    overall_active = int(overall.get("active_coarse_families") or 0)
    overall_other = float((overall.get("counts") or {}).get("Other", 0))
    overall_total = float(sum(int(v) for v in (overall.get("counts") or {}).values()))
    overall_other_share = (overall_other / overall_total) if overall_total > 0 else None

    if overall_active < 2:
        raise RuntimeError(
            "Degenerate hierarchy: fewer than two active coarse families were produced. "
            "The artifact taxonomy is not usable for training."
        )
    if overall_other_share is not None and overall_other_share >= 0.95:
        raise RuntimeError(
            f"Degenerate hierarchy: 'Other' accounts for {overall_other_share:.3f} of events. "
            "This indicates a broken action-to-family mapping."
        )

    by_race = hierarchy_audit.get("by_race") or {}
    bad_races = []
    for race, summary in by_race.items():
        active = int((summary or {}).get("active_coarse_families") or 0)
        if active < 2 and str(race) in {"Prot", "Terr", "Zerg"}:
            bad_races.append(str(race))
    if bad_races:
        raise RuntimeError(
            "Degenerate hierarchy for race-specific benchmark slices: "
            + ", ".join(sorted(bad_races))
        )


def build_imbalance_audit(
    exact_counts_by_split: dict[str, dict[int, int]],
    exact_counts_by_split_race: dict[str, dict[str, dict[int, int]]],
) -> dict:
    def _summarize(counts: dict[int, int]) -> dict:
        ordered = sorted(((int(k), int(v)) for k, v in counts.items()), key=lambda kv: (-kv[1], kv[0]))
        total = int(sum(v for _, v in ordered))
        top5 = ordered[:5]
        return {
            "n_active_exact_classes": int(sum(1 for _k, v in ordered if v > 0)),
            "total_examples": total,
            "top_exact_classes": [{"exact_action_id": k, "count": v} for k, v in ordered[:20]],
            "top5_cumulative_share": (float(sum(v for _, v in top5) / total) if total > 0 else None),
        }

    return {
        "by_split": {str(split): _summarize(counts) for split, counts in sorted(exact_counts_by_split.items())},
        "by_split_race": {
            str(split): {str(race): _summarize(counts) for race, counts in sorted(race_map.items())}
            for split, race_map in sorted(exact_counts_by_split_race.items())
        },
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument(
        "--tournament-regex",
        default="",
        help="Optional regex filter over tournament directory names, e.g. 'IEM'.",
    )
    ap.add_argument("--race", default="all")
    ap.add_argument("--label-mode", default="abil_link")
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument(
        "--split-mode",
        choices=["replay", "tournament", "player"],
        default="replay",
        help="How to partition train/val/test. replay is the primary thesis execution split.",
    )
    ap.add_argument(
        "--hierarchy-taxonomy",
        choices=["legacy8", "broad5", "broad6_warp", "balanced6", "macro_tactical3", "prodecon4"],
        default="legacy8",
        help="Coarse hierarchy taxonomy baked into the processed artifact.",
    )
    ap.add_argument("--family-top-k", type=int, default=0, help="Fine-label top-k per family (0 = keep all).")
    ap.add_argument("--keep-all-fine-actions", action="store_true", help="Do not collapse rare fine labels.")
    ap.add_argument("--separate-warp-in", action="store_true", help="Keep Warp-in separate when supported.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir)
    if args.smoke:
        outdir = outdir.parent / "artifacts_smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    tmp_dir = outdir / "tmp_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        excluded_dirs = {"code", "artifacts", "artifacts_smoke", "runs", "compare", "papers", ".agents", ".git", ".cache"}
        tournament_dirs = []
        for d in root.iterdir():
            if not d.is_dir():
                continue
            if d.name in excluded_dirs or d.name.startswith("."):
                continue
            if any(p.is_file() and p.name.endswith("_data.zip") for p in d.iterdir()):
                tournament_dirs.append(d.name)
        tournament_dirs = sorted(tournament_dirs)
        if args.tournament_regex:
            pattern = re.compile(args.tournament_regex)
            tournament_dirs = [name for name in tournament_dirs if pattern.search(name)]
        if not tournament_dirs:
            print("[error] No tournament directories matched the current filters.")
            return
        print(f"[prepare] tournaments={len(tournament_dirs)} filter={args.tournament_regex or '<all>'}")
        split_lookup = build_split_lookup(tournament_dirs, args.split_seed) if args.split_mode == "tournament" else None

        # Pass 1: Extract, Expand and Save to Disk
        all_manifest_rows = []
        for tname in tournament_dirs:
            troot = root / tname
            try:
                data_zip, mapping_json, _ = find_tournament_assets(troot)
                mapping = load_mapping(mapping_json)
                with zipfile.ZipFile(data_zip) as zf:
                    members = sorted([m for m in zf.namelist() if m.endswith(".SC2Replay.json")])
                    if args.smoke: members = members[:5]

                    t_events_raw, t_manifest = [], []
                    for member in members:
                        replay_id = member.replace(".SC2Replay.json", "")
                        source_path = mapping.get(f"{replay_id}.SC2Replay", "")
                        obj = read_json_from_zip(zf, member)
                        e_rows, m_rows = extract_rows_from_replay(
                            obj,
                            tname,
                            member,
                            source_path,
                            args.race,
                            args.split_seed,
                            args.split_mode,
                            args.label_mode,
                            0,
                            split_lookup,
                        )
                        t_events_raw.extend(e_rows)
                        t_manifest.extend(m_rows)

                    if t_events_raw:
                        df = pd.DataFrame(t_events_raw)
                        all_stat_keys = sorted({k for d in df["stats_snapshot"].tolist() if isinstance(d, dict) for k in d.keys()})
                        for k in all_stat_keys:
                            df[f"feat_{k}"] = df["stats_snapshot"].apply(lambda d: safe_float(d.get(k, 0.0), 0.0) if isinstance(d, dict) else 0.0)
                        df["feat_delta_prev_action_log1p"] = np.log1p(df["delta_prev_action_loops"].clip(lower=0).astype(np.float64))
                        df.drop(columns=["stats_snapshot"], inplace=True)
                        df.to_parquet(tmp_dir / f"{tname}.parquet", index=False)
                    all_manifest_rows.extend(t_manifest)
            except Exception as e:
                print(f"[warn] Failed {tname}: {e}")

        # Pass 2: Global Stats (Incremental)
        sum_feat = defaultdict(float)
        sum_sq_feat = defaultdict(float)
        count = 0
        for p_file in tmp_dir.glob("*.parquet"):
            df = pd.read_parquet(p_file)
            train_df = df[df["split"] == "train"]
            if train_df.empty: continue
            feat_cols = [c for c in train_df.columns if c.startswith("feat_")]
            count += len(train_df)
            for c in feat_cols:
                sum_feat[c] += train_df[c].sum()
                sum_sq_feat[c] += (train_df[c]**2).sum()

        if count == 0:
            print("[error] No training data found.")
            return

        means = {c: sum_feat[c] / count for c in sum_feat}
        stds = {c: np.sqrt(max(0, sum_sq_feat[c] / count - (means[c]**2))) for c in sum_feat}
        stds = {c: (s if s > 0 else 1.0) for c, s in stds.items()}
        preprocessing = {
            "feature_cols": sorted(list(means.keys())),
            "feature_mean": means,
            "feature_std": stds,
            "split_mode": args.split_mode,
            "effective_split_unit": (
                "tournament" if args.split_mode == "tournament"
                else "player_toon_id" if args.split_mode == "player"
                else "replay_id"
            ),
            "split_seed": int(args.split_seed),
            "tournament_regex": args.tournament_regex or None,
            "selected_tournaments": list(tournament_dirs),
            "hierarchy_taxonomy": args.hierarchy_taxonomy,
            "family_top_k": int(args.family_top_k),
            "keep_all_fine_actions": bool(args.keep_all_fine_actions),
            "separate_warp_in": bool(args.separate_warp_in),
        }

        # Pass 3: Stream results to final Parquet file using PyArrow
        # First, create a manifest and combined action set to get the final action_to_id mapping
        action_counts = defaultdict(int)
        for p_file in tmp_dir.glob("*.parquet"):
            df = pd.read_parquet(p_file)
            vc = df.loc[df["split"] == "train", "action_key"].value_counts()
            for key, value in vc.items():
                action_counts[str(key)] += int(value)

        if not action_counts:
            print("[error] No training action labels found.")
            return

        ordered_action_keys = [k for k, _v in sorted(action_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        action_to_id = {k: i for i, k in enumerate(ordered_action_keys)}
        keep_action_keys = set(ordered_action_keys)

        # Now stream and normalize directly to the final parquet to avoid
        # holding the whole dataset in memory.
        family_order = sc2_hier.taxonomy_families(args.hierarchy_taxonomy, separate_warp_in=args.separate_warp_in)
        hierarchy_rules = sc2_hier.default_link_family_rules(args.hierarchy_taxonomy, separate_warp_in=args.separate_warp_in)
        df_manifest = pd.DataFrame(all_manifest_rows)
        manifest_merge_cols = [
            "replay_id", "player_toon_id", "player_race", "opponent_race", "matchup",
            "split_mode", "split_unit_kind", "split_unit_id",
            "player_id", "player_user_id",
        ]
        manifest_merge_df = df_manifest[manifest_merge_cols].drop_duplicates()
        writer = None
        output_path = outdir / "processed_events.parquet"
        hierarchy_vocab_frames: list[pd.DataFrame] = []
        coarse_counts_total: dict[str, int] = defaultdict(int)
        coarse_counts_by_race: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        exact_counts_by_split: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        exact_counts_by_split_race: dict[str, dict[str, dict[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        for p_file in sorted(tmp_dir.glob("*.parquet")):
            df = pd.read_parquet(p_file)
            # 1. Normalize
            feat_cols = preprocessing["feature_cols"]
            for c in feat_cols:
                df[c] = (df[c].astype(np.float64) - means[c]) / stds[c]
            # 2. Filter actions
            df = df[df["action_key"].isin(keep_action_keys)].copy()
            # 3. Map Action ID
            df["action_id"] = df["action_key"].map(action_to_id).astype(np.int64)
            # 4. Add player/race metadata before hierarchy assignment so race-specific
            # rules can be applied during preprocessing.
            df["player_toon_id"] = df["group_id"].astype(str).str.split("::").str[-1]
            df = df.merge(manifest_merge_df, on=["replay_id", "player_toon_id"], how="left")
            # 5. Hierarchy
            df, _ = sc2_hier.build_hierarchy_labels(
                df,
                family_top_k=int(args.family_top_k),
                family_other_token="__OTHER__",
                link_family_rules=hierarchy_rules,
                family_order=family_order,
                keep_all_fine_actions=bool(args.keep_all_fine_actions),
                taxonomy=args.hierarchy_taxonomy,
                separate_warp_in=bool(args.separate_warp_in),
            )
            # 6. Finalize per-tournament chunk and append to parquet
            df = df.sort_values(["group_id", "event_loop", "action_link", "action_cmd_idx"]).reset_index(drop=True)
            df["event_index"] = df.groupby("group_id").cumcount()
            for (split_name, exact_id), count in df.groupby(["split", "exact_action_id"]).size().items():
                exact_counts_by_split[str(split_name)][int(exact_id)] += int(count)
            for (split_name, race_name, exact_id), count in df.groupby(["split", "player_race", "exact_action_id"]).size().items():
                exact_counts_by_split_race[str(split_name)][str(race_name)][int(exact_id)] += int(count)
            for (race_name, coarse_name), count in df.groupby(["player_race", "coarse_family"]).size().items():
                coarse_counts_by_race[str(race_name)][str(coarse_name)] += int(count)
            for coarse_name, count in df["coarse_family"].value_counts().items():
                coarse_counts_total[str(coarse_name)] += int(count)
            hierarchy_vocab_frames.append(
                df[["player_race", "action_key", "action_id", "split"]].drop_duplicates().copy()
            )
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)

        if writer is not None:
            writer.close()
        else:
            print("[error] No parquet chunks were written.")
            return

        df_manifest.to_csv(outdir / "master_manifest.csv", index=False)
        (outdir / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2))

        # Save Action Vocab (including hierarchy mapping from the last processed file's output)
        sample_df = pd.concat(hierarchy_vocab_frames, ignore_index=True).drop_duplicates()
        if "split" not in sample_df.columns:
            sample_df["split"] = "train"
        sample_df["action_link"] = sample_df["action_key"].astype(str).map(
            lambda x: safe_int(str(x).split(":", 1)[0], 0)
        )
        sample_df["action_cmd_idx"] = sample_df["action_key"].astype(str).map(
            lambda x: safe_int(str(x).split(":", 1)[1], 0) if ":" in str(x) else 0
        )
        _, hierarchy_mapping = sc2_hier.build_hierarchy_labels(
            sample_df,
            family_top_k=int(args.family_top_k),
            family_other_token="__OTHER__",
            link_family_rules=hierarchy_rules,
            family_order=family_order,
            keep_all_fine_actions=bool(args.keep_all_fine_actions),
            taxonomy=args.hierarchy_taxonomy,
            separate_warp_in=bool(args.separate_warp_in),
        )

        split_audit = build_split_audit(df_manifest, args.split_mode)
        hierarchy_audit = build_hierarchy_audit(coarse_counts_total, coarse_counts_by_race)
        imbalance_audit = build_imbalance_audit(exact_counts_by_split, exact_counts_by_split_race)
        assert_hierarchy_is_usable(hierarchy_audit)

        (outdir / "action_vocab.json").write_text(json.dumps({"action_to_id": action_to_id, "id_to_action": {i: k for k, i in action_to_id.items()}, "hierarchy": hierarchy_mapping}, indent=2))
        (outdir / "split_audit.json").write_text(json.dumps(split_audit, indent=2))
        (outdir / "hierarchy_audit.json").write_text(json.dumps(hierarchy_audit, indent=2))
        (outdir / "class_imbalance_summary.json").write_text(json.dumps(imbalance_audit, indent=2))

        print(f"[done] Processed dataset saved to {outdir}")
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
