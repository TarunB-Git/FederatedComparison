#!/usr/bin/env python3
"""Shared data loading, datasets, collation, client partitioning and utilities
for the hierarchical next-action prediction pipeline on SC2EGSet."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

import hierarchy as sc2_hier


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SequencePackHier:
    """One player-replay sequence with hierarchical labels."""
    x: np.ndarray          # (T, n_features) float32
    y_exact: np.ndarray    # (T,) int64  — exact action id
    y_coarse: np.ndarray   # (T,) int64  — coarse family id
    y_fine: np.ndarray     # (T,) int64  — fine action id within family
    player_race: str = ""  # race string: "Prot", "Terr", "Zerg"


class IndexedEventTable:
    """Compact, sorted event table backend for lower-memory window datasets."""

    def __init__(self, df: pd.DataFrame, feature_cols: list[str]):
        sort_cols = ["group_id", "event_index"]
        sorted_df = df.sort_values(sort_cols).reset_index(drop=True)
        self.feature_cols = list(feature_cols)
        self.x = sorted_df[self.feature_cols].to_numpy(dtype=np.float32, copy=True)
        self.y_exact = sorted_df["exact_action_id"].to_numpy(dtype=np.int32, copy=False)
        self.y_coarse = sorted_df["coarse_family_id"].to_numpy(dtype=np.int16, copy=False)
        self.y_fine = sorted_df["fine_action_id"].to_numpy(dtype=np.int32, copy=False)
        self.splits = sorted_df["split"].astype(str).to_numpy(copy=False)
        self.player_ids = (
            sorted_df["player_toon_id"].astype(str).to_numpy(copy=False)
            if "player_toon_id" in sorted_df.columns else None
        )
        self.player_races = (
            sorted_df["player_race"].astype(str).to_numpy(copy=False)
            if "player_race" in sorted_df.columns else None
        )

        group_codes = sorted_df["group_id"].astype("category").cat.codes.to_numpy(dtype=np.int32, copy=False)
        starts: list[int] = []
        ends: list[int] = []
        splits: list[str] = []
        player_ids: list[str] = []
        races: list[str] = []
        if len(group_codes):
            start = 0
            prev = int(group_codes[0])
            for idx in range(1, len(group_codes)):
                cur = int(group_codes[idx])
                if cur != prev:
                    starts.append(start)
                    ends.append(idx)
                    splits.append(str(self.splits[start]))
                    player_ids.append("" if self.player_ids is None else str(self.player_ids[start]))
                    races.append("" if self.player_races is None else str(self.player_races[start]))
                    start = idx
                    prev = cur
            starts.append(start)
            ends.append(len(group_codes))
            splits.append(str(self.splits[start]))
            player_ids.append("" if self.player_ids is None else str(self.player_ids[start]))
            races.append("" if self.player_races is None else str(self.player_races[start]))

        self.group_starts = np.asarray(starts, dtype=np.int32)
        self.group_ends = np.asarray(ends, dtype=np.int32)
        self.group_splits = np.asarray(splits, dtype=object)
        self.group_player_ids = np.asarray(player_ids, dtype=object)
        self.group_player_races = np.asarray(races, dtype=object)
        self.group_sample_counts = np.maximum(self.group_ends - self.group_starts - 1, 0).astype(np.int32)
        del sorted_df

    def build_dataset(
        self,
        *,
        window: int,
        max_samples: int,
        shuffle: bool,
        seed: int,
        include_race: bool = False,
        split: str | None = None,
        player_ids: set[str] | None = None,
    ) -> "IndexedEventWindowDataset":
        selected_group_starts: list[int] = []
        selected_group_counts: list[int] = []
        selected_group_races: list[str] = []
        split_name = None if split is None else str(split)

        for gidx in range(len(self.group_starts)):
            gsplit = str(self.group_splits[gidx])
            if split_name is not None and gsplit != split_name:
                continue
            if player_ids is not None:
                pid = str(self.group_player_ids[gidx])
                if pid not in player_ids:
                    continue

            start = int(self.group_starts[gidx])
            end = int(self.group_ends[gidx])
            n_samples = int(end - start - 1)
            if n_samples <= 0:
                continue
            selected_group_starts.append(start)
            selected_group_counts.append(n_samples)
            selected_group_races.append(str(self.group_player_races[gidx]))

        group_starts_arr = np.asarray(selected_group_starts, dtype=np.int32)
        group_counts_arr = np.asarray(selected_group_counts, dtype=np.int32)
        group_races_arr = np.asarray(selected_group_races, dtype=object)
        cumulative_counts = np.cumsum(group_counts_arr, dtype=np.int64)
        total_samples = int(cumulative_counts[-1]) if cumulative_counts.size else 0

        sample_indices = None
        if max_samples > 0 and total_samples > max_samples:
            if shuffle:
                rng = np.random.default_rng(seed)
                sample_indices = np.sort(rng.choice(total_samples, size=max_samples, replace=False).astype(np.int64))
            total_samples = int(max_samples)

        return IndexedEventWindowDataset(
            table=self,
            window=window,
            include_race=include_race,
            group_starts=group_starts_arr,
            group_counts=group_counts_arr,
            group_races=group_races_arr,
            cumulative_counts=cumulative_counts,
            total_samples=total_samples,
            sample_indices=sample_indices,
        )

    def player_sample_counts(self, *, split: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        split_name = str(split)
        for gidx in range(len(self.group_starts)):
            if str(self.group_splits[gidx]) != split_name:
                continue
            n_samples = int(self.group_sample_counts[gidx])
            if n_samples <= 0:
                continue
            pid = str(self.group_player_ids[gidx])
            counts[pid] = counts.get(pid, 0) + n_samples
        return counts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EventWindowHierDataset(Dataset):
    """Sliding-window next-action prediction dataset for hierarchical labels.

    Each sample provides a feature window x[t-w+1:t+1] and the *next-step*
    labels y_coarse[t+1], y_fine[t+1], y_exact[t+1].
    """

    def __init__(
        self,
        sequences: dict[str, SequencePackHier],
        *,
        window: int,
        max_samples: int,
        shuffle: bool,
        seed: int,
        include_race: bool = False,
    ):
        self.sequences = sequences
        self.window = int(window)
        self.include_race = bool(include_race)
        self.samples: list[tuple[str, int]] = []
        self.sample_exact_labels: list[int] = []

        for gid, seq in self.sequences.items():
            n = int(seq.y_exact.shape[0])
            if n < 2:
                continue
            for end_idx in range(n - 1):
                self.samples.append((gid, end_idx))
                self.sample_exact_labels.append(int(seq.y_exact[end_idx + 1]))

        if shuffle and self.samples:
            rng = np.random.default_rng(seed)
            order = np.arange(len(self.samples))
            rng.shuffle(order)
            self.samples = [self.samples[i] for i in order]
            self.sample_exact_labels = [self.sample_exact_labels[i] for i in order]

        if max_samples > 0:
            self.samples = self.samples[:max_samples]
            self.sample_exact_labels = self.sample_exact_labels[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        gid, end_idx = self.samples[idx]
        seq = self.sequences[gid]
        start_idx = max(0, end_idx - self.window + 1)

        x = seq.x[start_idx : end_idx + 1]
        y_exact = int(seq.y_exact[end_idx + 1])
        y_coarse = int(seq.y_coarse[end_idx + 1])
        y_fine = int(seq.y_fine[end_idx + 1])

        base = (
            torch.from_numpy(x.astype(np.float32)),
            torch.tensor(y_coarse, dtype=torch.long),
            torch.tensor(y_fine, dtype=torch.long),
            torch.tensor(y_exact, dtype=torch.long),
            torch.tensor(x.shape[0], dtype=torch.long),
        )
        if self.include_race:
            return (*base, seq.player_race)
        return base


class IndexedEventWindowDataset(Dataset):
    """Window dataset backed by one shared sorted event table."""

    def __init__(
        self,
        *,
        table: IndexedEventTable,
        window: int,
        include_race: bool,
        group_starts: np.ndarray,
        group_counts: np.ndarray,
        group_races: np.ndarray,
        cumulative_counts: np.ndarray,
        total_samples: int,
        sample_indices: np.ndarray | None,
    ):
        self.table = table
        self.window = int(window)
        self.include_race = bool(include_race)
        self.group_starts = group_starts
        self.group_counts = group_counts
        self.group_races = group_races
        self.cumulative_counts = cumulative_counts
        self.total_samples = int(total_samples)
        self.sample_indices = sample_indices

    def __len__(self) -> int:
        return int(self.total_samples)

    def __getitem__(self, idx: int):
        logical_idx = int(idx)
        if self.sample_indices is not None:
            logical_idx = int(self.sample_indices[logical_idx])

        group_idx = int(np.searchsorted(self.cumulative_counts, logical_idx, side="right"))
        prev_cum = 0 if group_idx == 0 else int(self.cumulative_counts[group_idx - 1])
        offset = logical_idx - prev_cum
        group_start = int(self.group_starts[group_idx])
        end_idx = int(group_start + offset)
        start_idx = max(group_start, end_idx - self.window + 1)
        target_idx = end_idx + 1

        x = self.table.x[start_idx : end_idx + 1]
        y_exact = int(self.table.y_exact[target_idx])
        y_coarse = int(self.table.y_coarse[target_idx])
        y_fine = int(self.table.y_fine[target_idx])

        base = (
            torch.from_numpy(x),
            torch.tensor(y_coarse, dtype=torch.long),
            torch.tensor(y_fine, dtype=torch.long),
            torch.tensor(y_exact, dtype=torch.long),
            torch.tensor(x.shape[0], dtype=torch.long),
        )
        if self.include_race:
            race = str(self.group_races[group_idx]) if self.group_races.size else ""
            return (*base, race)
        return base


def collate_hier_batch(batch):
    """Pad variable-length windows to the longest in the batch."""
    lengths = torch.stack([b[4] for b in batch], dim=0)
    max_len = int(lengths.max().item())
    feat_dim = int(batch[0][0].shape[1])

    xs = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
    ys_coarse = torch.stack([b[1] for b in batch], dim=0)
    ys_fine = torch.stack([b[2] for b in batch], dim=0)
    ys_exact = torch.stack([b[3] for b in batch], dim=0)
    has_race = len(batch[0]) >= 6
    player_races = [b[5] for b in batch] if has_race else None

    for i, row in enumerate(batch):
        x = row[0]
        length = row[4]
        xs[i, : int(length.item())] = x

    if has_race:
        return xs, ys_coarse, ys_fine, ys_exact, lengths, player_races
    return xs, ys_coarse, ys_fine, ys_exact, lengths


# ---------------------------------------------------------------------------
# Data loading & normalisation
# ---------------------------------------------------------------------------

def resolve_dataset_paths(dataset_dir: str) -> tuple[Path, Path, Path]:
    """Resolve events, preprocessing, and action_vocab paths from a dataset dir."""
    dd = Path(dataset_dir)
    events_path = dd / "processed_events.parquet"
    if not events_path.exists():
        events_path = dd / "processed_events.csv"
    preprocessing_path = dd / "preprocessing.json"
    action_vocab_path = dd / "action_vocab.json"
    return events_path, preprocessing_path, action_vocab_path


def load_and_normalize_events(
    events_path: Path,
    preprocessing_path: Path,
    *,
    race_filter: str | None = None,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Load events table and z-score normalise features using saved stats."""
    pre = json.loads(preprocessing_path.read_text())
    feature_cols = list(pre["feature_cols"])
    mean = pre["feature_mean"]
    std = pre["feature_std"]

    requested_race = None
    if race_filter is not None and str(race_filter).strip().lower() not in {"", "all"}:
        requested_race = str(race_filter)

    if events_path.suffix.lower() == ".parquet":
        needed_cols = list(dict.fromkeys(
            feature_cols + [
                "tournament", "replay_id", "split", "group_id", "player_toon_id",
                "player_race", "opponent_race", "matchup", "event_loop",
                "delta_prev_action_loops", "action_link", "action_cmd_idx",
                "event_index", "action_id", "exact_action_id", "fine_action_id",
                "coarse_family_id", "action_key", "coarse_family", "fine_action_key",
            ]
        ))
        schema_cols = set(pq.ParquetFile(events_path).schema.names)
        read_cols = [c for c in needed_cols if c in schema_cols]
        read_kwargs = {"columns": read_cols}
        if requested_race is not None and "player_race" in schema_cols:
            read_kwargs["filters"] = [("player_race", "==", requested_race)]
        df = pd.read_parquet(events_path, **read_kwargs)
    elif events_path.suffix.lower() == ".csv":
        df = pd.read_csv(events_path)
        if requested_race is not None and "player_race" in df.columns:
            df = df[df["player_race"].astype(str) == requested_race].copy()
    else:
        raise ValueError(f"Unsupported events file suffix: {events_path.suffix}")

    if "player_race" not in df.columns:
        manifest_path = events_path.parent / "master_manifest.csv"
        if manifest_path.exists():
            manifest_df = pd.read_csv(manifest_path)
            merge_cols = ["replay_id"]
            if "player_toon_id" in df.columns and "player_toon_id" in manifest_df.columns:
                merge_cols.append("player_toon_id")
            enrich_cols = [
                c for c in ["player_race", "opponent_race", "matchup"] if c in manifest_df.columns
            ]
            if enrich_cols:
                df = df.merge(
                    manifest_df[merge_cols + enrich_cols].drop_duplicates(),
                    on=merge_cols,
                    how="left",
                )

    for c in feature_cols:
        m = float(mean[c])
        s = float(std[c])
        if s <= 0:
            s = 1.0
        df[c] = ((df[c].astype(np.float32) - m) / s).astype(np.float32)

    compact_int32 = [
        "event_loop", "delta_prev_action_loops", "action_link", "action_cmd_idx",
        "event_index", "action_id", "exact_action_id", "fine_action_id",
    ]
    compact_int16 = ["coarse_family_id"]
    for c in compact_int32:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int32)
    for c in compact_int16:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int16)

    compact_category = [
        "tournament", "replay_id", "split", "group_id", "player_toon_id",
        "player_race", "opponent_race", "matchup", "action_key",
        "coarse_family", "fine_action_key",
    ]
    for c in compact_category:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype("category")

    return df, feature_cols, pre


def build_event_table(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> IndexedEventTable:
    """Build one shared sorted backend table for low-memory window datasets."""
    return IndexedEventTable(df, feature_cols)


def add_action_context_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    enabled: bool = True,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Add current-action context to each time step for next-action prediction.

    The supervised target remains the next event at ``t+1``. These features
    describe only the observed event at ``t`` inside the input window, so they
    add action-history signal without leaking the target label.
    """
    base_cols = list(feature_cols)
    if not enabled:
        return df, base_cols, {
            "action_context_features_enabled": False,
            "action_context_feature_cols": [],
            "action_context_feature_mode": "disabled",
        }

    out = df.copy()
    train_mask = (
        out["split"].astype(str).eq("train")
        if "split" in out.columns
        else pd.Series(True, index=out.index)
    )
    new_cols: list[str] = []
    numeric_stats: dict[str, dict[str, float]] = {}

    def _add_zscore_feature(name: str, values: pd.Series | np.ndarray) -> None:
        vals = pd.Series(values, index=out.index)
        vals = pd.to_numeric(vals, errors="coerce").fillna(0.0).astype(np.float32)
        train_vals = vals[train_mask]
        ref_vals = train_vals if len(train_vals) else vals
        mean = float(ref_vals.mean()) if len(ref_vals) else 0.0
        std = float(ref_vals.std(ddof=0)) if len(ref_vals) else 1.0
        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0
        out[name] = ((vals - mean) / std).astype(np.float32)
        new_cols.append(name)
        numeric_stats[name] = {"mean": mean, "std": std}

    if "action_link" in out.columns:
        action_link = pd.to_numeric(out["action_link"], errors="coerce").fillna(0).clip(lower=0)
        _add_zscore_feature("feat_ctx_action_link_log1p", np.log1p(action_link))
    if "action_cmd_idx" in out.columns:
        action_cmd = pd.to_numeric(out["action_cmd_idx"], errors="coerce").fillna(0).clip(lower=0)
        _add_zscore_feature("feat_ctx_action_cmd_idx_log1p", np.log1p(action_cmd))
    if "action_id" in out.columns:
        action_id = pd.to_numeric(out["action_id"], errors="coerce").fillna(0).clip(lower=0)
        _add_zscore_feature("feat_ctx_action_id_log1p", np.log1p(action_id))

    if "coarse_family_id" in out.columns:
        coarse_ids = pd.to_numeric(out["coarse_family_id"], errors="coerce").fillna(0).astype(np.int32)
        _add_zscore_feature("feat_ctx_coarse_family_id", coarse_ids)
        train_ids = coarse_ids[train_mask]
        present_families = sorted({int(x) for x in (train_ids if len(train_ids) else coarse_ids).unique().tolist()})
        for fam_id in present_families:
            col = f"feat_ctx_coarse_is_{fam_id}"
            out[col] = (coarse_ids == int(fam_id)).astype(np.float32)
            new_cols.append(col)

    # ── Temporal positional features ──────────────────────────────────────
    # event_loop encodes game-time position (tick count).  Actions are
    # heavily phase-dependent (Build early, Attack late), so this is a
    # strong non-leaking contextual signal.
    if "event_loop" in out.columns:
        event_loop = pd.to_numeric(out["event_loop"], errors="coerce").fillna(0).clip(lower=0)
        _add_zscore_feature("feat_ctx_event_loop_log1p", np.log1p(event_loop))

    # Game-progress: normalise event_loop to [0, 1] within each group.
    # Uses map() instead of groupby.transform() to avoid OOM on large datasets.
    if "event_loop" in out.columns and "group_id" in out.columns:
        event_loop = pd.to_numeric(out["event_loop"], errors="coerce").fillna(0).clip(lower=0).astype(np.float32)
        group_max_dict = event_loop.groupby(out["group_id"], observed=True).max()
        group_max = out["group_id"].map(group_max_dict).astype(np.float32)
        progress = np.where(group_max > 0, event_loop / group_max, np.float32(0.0))
        out["feat_ctx_game_progress"] = progress.astype(np.float32)
        new_cols.append("feat_ctx_game_progress")
        del event_loop, group_max_dict, group_max, progress

    # Action rate: inverse of delta_prev_action_loops (actions-per-tick).
    if "delta_prev_action_loops" in out.columns:
        delta = pd.to_numeric(out["delta_prev_action_loops"], errors="coerce").fillna(0).clip(lower=0).astype(np.float32)
        action_rate = np.float32(1.0) / (delta + np.float32(1.0))
        _add_zscore_feature("feat_ctx_action_rate", action_rate)
        del delta, action_rate

    merged_cols = list(dict.fromkeys(base_cols + new_cols))
    return out, merged_cols, {
        "action_context_features_enabled": True,
        "action_context_feature_cols": list(new_cols),
        "action_context_feature_count": int(len(new_cols)),
        "action_context_feature_mode": "current_action_numeric_and_current_coarse_onehot_and_temporal",
        "action_context_numeric_stats": numeric_stats,
    }


# ---------------------------------------------------------------------------
# Sequence building
# ---------------------------------------------------------------------------

def build_hier_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: str,
) -> dict[str, SequencePackHier]:
    """Build per-group sequences with hierarchy labels for a given split."""
    split_df = df[df["split"] == split].copy()
    out: dict[str, SequencePackHier] = {}

    for gid, g in split_df.sort_values(["group_id", "event_index"]).groupby("group_id"):
        x = g[feature_cols].to_numpy(dtype=np.float32)
        y_exact = g["exact_action_id"].to_numpy(dtype=np.int64)
        y_coarse = g["coarse_family_id"].to_numpy(dtype=np.int64)
        y_fine = g["fine_action_id"].to_numpy(dtype=np.int64)
        player_race = str(g["player_race"].iloc[0]) if "player_race" in g.columns else ""
        if len(y_exact) >= 2:
            out[str(gid)] = SequencePackHier(
                x=x, y_exact=y_exact, y_coarse=y_coarse, y_fine=y_fine, player_race=player_race,
            )
    return out


def build_client_hier_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, dict[str, SequencePackHier]]:
    """Partition training data into per-player (client) sequence maps."""
    train_df = df[df["split"] == "train"].copy()
    client_map: dict[str, dict[str, SequencePackHier]] = {}

    for toon_id, g_player in train_df.groupby("player_toon_id"):
        seqs: dict[str, SequencePackHier] = {}
        for gid, g in g_player.sort_values(["group_id", "event_index"]).groupby("group_id"):
            x = g[feature_cols].to_numpy(dtype=np.float32)
            y_exact = g["exact_action_id"].to_numpy(dtype=np.int64)
            y_coarse = g["coarse_family_id"].to_numpy(dtype=np.int64)
            y_fine = g["fine_action_id"].to_numpy(dtype=np.int64)
            player_race = str(g["player_race"].iloc[0]) if "player_race" in g.columns else ""
            if len(y_exact) >= 2:
                seqs[str(gid)] = SequencePackHier(
                    x=x, y_exact=y_exact, y_coarse=y_coarse, y_fine=y_fine, player_race=player_race,
                )
        if seqs:
            client_map[str(toon_id)] = seqs

    return client_map


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def build_class_weights(
    labels: np.ndarray,
    num_classes: int,
    mode: str,
    max_weight: float,
) -> np.ndarray | None:
    """Compute inverse or inverse-sqrt class weights, capped at max_weight."""
    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if mode == "inverse":
        w = 1.0 / counts
    elif mode == "inverse_sqrt":
        w = 1.0 / np.sqrt(counts)
    else:
        return None
    w = w / np.maximum(w.mean(), 1e-12)
    if max_weight > 0:
        w = np.minimum(w, max_weight)
        # Re-normalize and cap again to prevent the re-normalization
        # from pushing values back above the cap.
        w = w / np.maximum(w.mean(), 1e-12)
        w = np.minimum(w, max_weight)
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# Hierarchy mapping helpers
# ---------------------------------------------------------------------------

def _resolve_action_keys_from_vocab(
    df: pd.DataFrame,
    vocab: dict,
) -> pd.Series:
    if "action_key" in df.columns:
        return df["action_key"].astype(str)

    id_to_action_raw = vocab.get("id_to_action") or {}
    id_to_action = {int(k): str(v) for k, v in id_to_action_raw.items()}
    if not id_to_action:
        raise RuntimeError(
            "Cannot reconstruct action_key values: action_vocab.json is missing id_to_action "
            "and processed events do not contain action_key.",
        )

    source_col = None
    if "exact_action_id" in df.columns:
        source_col = "exact_action_id"
    elif "action_id" in df.columns:
        source_col = "action_id"
    if source_col is None:
        raise RuntimeError(
            "Cannot reconstruct action_key values: neither exact_action_id nor action_id is available.",
        )

    return df[source_col].map(lambda x: id_to_action.get(int(x), str(int(x)))).astype(str)


def _drop_unused_families(
    df: pd.DataFrame,
    hierarchy: dict,
) -> tuple[pd.DataFrame, dict]:
    family_to_id_raw = hierarchy.get("family_to_id", {})
    family_to_id = {str(k): int(v) for k, v in family_to_id_raw.items()}
    if not family_to_id or "coarse_family_id" not in df.columns:
        return df, hierarchy

    old_id_to_name = {int(v): str(k) for k, v in family_to_id.items()}
    present_old_ids = sorted({int(x) for x in df["coarse_family_id"].dropna().astype(int).unique().tolist()})
    present_names = [old_id_to_name[old_id] for old_id in present_old_ids if old_id in old_id_to_name]
    if not present_names or len(present_names) == len(family_to_id):
        return df, hierarchy

    old_to_new = {old_id: new_id for new_id, old_id in enumerate(present_old_ids) if old_id in old_id_to_name}
    family_counts_raw = hierarchy.get("family_counts", {})
    family_fine_to_id_raw = hierarchy.get("family_fine_to_id", {})
    family_other_exact_raw = hierarchy.get("family_other_exact_action_id", {})

    out_df = df.copy()
    out_df["coarse_family_id"] = out_df["coarse_family_id"].map(lambda x: old_to_new[int(x)]).astype(np.int64)
    if "coarse_family" in out_df.columns:
        out_df["coarse_family"] = out_df["coarse_family_id"].map(
            {new_id: old_id_to_name[old_id] for old_id, new_id in old_to_new.items()}
        ).astype(str)

    new_family_to_id = {fam_name: idx for idx, fam_name in enumerate(present_names)}
    new_hierarchy = dict(hierarchy)
    new_hierarchy["family_order"] = list(present_names)
    new_hierarchy["family_to_id"] = dict(new_family_to_id)
    new_hierarchy["id_to_family"] = {str(v): k for k, v in new_family_to_id.items()}
    new_hierarchy["family_fine_to_id"] = {
        fam_name: dict(family_fine_to_id_raw.get(fam_name, {}))
        for fam_name in present_names
    }
    new_hierarchy["family_id_to_fine"] = {
        fam_name: {str(v): k for k, v in dict(family_fine_to_id_raw.get(fam_name, {})).items()}
        for fam_name in present_names
    }
    new_hierarchy["family_other_exact_action_id"] = {
        fam_name: int(family_other_exact_raw.get(fam_name, -1))
        for fam_name in present_names
    }
    if family_counts_raw:
        new_hierarchy["family_counts"] = {
            fam_name: int(family_counts_raw.get(fam_name, 0))
            for fam_name in present_names
        }

    return out_df, new_hierarchy


def prepare_hierarchy_targets(
    df: pd.DataFrame,
    vocab: dict,
    *,
    coarse_taxonomy: str = "auto",
    separate_warp_in: bool = False,
) -> tuple[pd.DataFrame, dict, dict]:
    """Prepare hierarchy labels for training/eval.

    Supported ``coarse_taxonomy`` values:
      - ``"auto"``       — use legacy8 (8-class family-based taxonomy, best coarse accuracy)
      - ``"dataset"``    — use whatever taxonomy the artifact was built with
      - ``"legacy8"``    — 8-class family-based taxonomy (Build/Train/Tech/Expand/Attack/Scout/Defend/Other)
      - ``"broad5"``     — broad 6-family taxonomy (Economy/Production/Tech/Combat/Information/Other)
      - ``"broad6_warp"``— broad 7-family taxonomy (with separate Warp-in family)
      - ``"balanced6"``  — balanced 6-family taxonomy from merged legacy groups
      - ``"macro_tactical3"`` — 3-family taxonomy (Macro, Tactical, Information)
      - ``"prodecon4"``  — 4-family taxonomy with Production+Economy merged (ProdEcon, Technology, TacticalCombat, Information)
    """
    _NAMED = {"legacy8", "broad5", "broad6_warp", "balanced6", "macro_tactical3", "prodecon4"}
    _ALIASES = {"legacy": "legacy8", "broad": "broad5"}
    _ALL_VALID = {"auto", "dataset"} | _NAMED | set(_ALIASES.keys())

    requested = str(coarse_taxonomy).strip().lower()
    if requested not in _ALL_VALID:
        raise ValueError(f"Unsupported coarse taxonomy: {coarse_taxonomy}")

    # Resolve aliases
    effective_requested = _ALIASES.get(requested, requested)

    source_hierarchy = dict(vocab.get("hierarchy") or {})
    dataset_taxonomy = str(source_hierarchy.get("hierarchy_taxonomy", "")).strip().lower()

    # Decide whether we need to rebuild labels
    if effective_requested == "dataset":
        rebuild_taxonomy = None
    elif effective_requested == "auto":
        # Auto → use legacy8 (8-class) for best coarse accuracy (primary benchmark)
        rebuild_taxonomy = "legacy8" if dataset_taxonomy not in {"legacy8", "legacy"} else None
    else:
        rebuild_taxonomy = effective_requested

    out_df = df.copy()
    hierarchy = source_hierarchy

    if rebuild_taxonomy is not None:
        work_df = out_df.copy()
        work_df["action_key"] = _resolve_action_keys_from_vocab(work_df, vocab)
        if "action_id" not in work_df.columns:
            if "exact_action_id" not in work_df.columns:
                raise RuntimeError(
                    f"Cannot rebuild {rebuild_taxonomy} hierarchy labels: processed events "
                    "are missing both action_id and exact_action_id.",
                )
            work_df["action_id"] = work_df["exact_action_id"].astype(np.int64)

        # For broad6_warp, set separate_warp_in so the broad taxonomy dispatches correctly
        use_separate_warp_in = separate_warp_in or rebuild_taxonomy == "broad6_warp"

        family_order = sc2_hier.taxonomy_families(rebuild_taxonomy, use_separate_warp_in)
        link_rules = sc2_hier.load_hierarchy_rules(
            "",
            taxonomy=rebuild_taxonomy,
            separate_warp_in=use_separate_warp_in,
            family_order=family_order,
        )
        out_df, hierarchy = sc2_hier.build_hierarchy_labels(
            work_df,
            family_top_k=0,
            family_other_token="__OTHER__",
            link_family_rules=link_rules,
            family_order=family_order,
            keep_all_fine_actions=True,
            taxonomy=rebuild_taxonomy,
            separate_warp_in=use_separate_warp_in,
        )
    else:
        required_dataset_cols = {"coarse_family_id", "coarse_family", "fine_action_id", "exact_action_id"}
        missing_dataset_cols = sorted(required_dataset_cols - set(out_df.columns))
        if missing_dataset_cols:
            raise RuntimeError(
                "Dataset taxonomy was requested, but processed events are missing hierarchy columns: "
                + ", ".join(missing_dataset_cols)
            )

    out_df, hierarchy = _drop_unused_families(out_df, hierarchy)
    meta = {
        "requested_coarse_taxonomy": requested,
        "dataset_coarse_taxonomy": dataset_taxonomy or None,
        "effective_coarse_taxonomy": (
            str(hierarchy.get("hierarchy_taxonomy", "")).strip().lower() or "dataset"
        ),
        "separate_warp_in": bool(separate_warp_in),
        "coarse_targets_rebuilt": rebuild_taxonomy is not None,
    }
    return out_df, hierarchy, meta


def assert_hierarchy_targets_usable(
    df: pd.DataFrame,
    *,
    race: str = "all",
    min_active_families: int = 2,
    max_other_share: float = 0.95,
) -> None:
    required_cols = {"split", "coarse_family_id", "coarse_family"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise RuntimeError(
            "Processed dataset is missing required hierarchy columns: "
            + ", ".join(missing)
        )

    work_df = df
    selected_race = str(race or "all")
    if selected_race != "all" and "player_race" in work_df.columns:
        work_df = work_df[work_df["player_race"].astype(str) == selected_race]

    for split_name in ["train", "val", "test"]:
        split_df = work_df[work_df["split"].astype(str) == split_name]
        if split_df.empty:
            raise RuntimeError(
                f"No {split_name} rows available for race={selected_race}. "
                "The split or race filter is not usable for training."
            )
        coarse_counts = split_df["coarse_family"].astype(str).value_counts()
        active = int(coarse_counts.shape[0])
        if active < min_active_families:
            raise RuntimeError(
                f"Degenerate hierarchy for race={selected_race} split={split_name}: "
                f"only {active} active coarse families."
            )
        other_share = float(coarse_counts.get("Other", 0) / max(1, int(coarse_counts.sum())))
        if other_share >= max_other_share:
            raise RuntimeError(
                f"Degenerate hierarchy for race={selected_race} split={split_name}: "
                f"'Other' share is {other_share:.3f}, indicating a broken taxonomy mapping."
            )


def _env_flag(name: str, default: str = "1") -> bool:
    return str(os.environ.get(name, default)).strip().lower() not in {"", "0", "false", "no", "off"}


def _support_entropy(counts: pd.Series) -> float | None:
    values = counts.to_numpy(dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return None
    probs = values[values > 0] / total
    if probs.size <= 1:
        return 0.0
    entropy = float(-(probs * np.log2(probs)).sum())
    max_entropy = float(np.log2(probs.size))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _append_coarse_audit(outdir: Path, event: dict) -> None:
    path = Path(outdir) / "coarse_inflation_audit.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except Exception:
            payload = {"events": []}
    else:
        payload = {"events": []}
    events = list(payload.get("events") or [])
    events.append(event)
    payload["events"] = events
    payload["status"] = "failed" if any(e.get("failed") for e in events) else (
        "rectified" if any(e.get("rectifications") for e in events) else "ok"
    )
    path.write_text(json.dumps(payload, indent=2))


def apply_pretrain_coarse_audit(
    df: pd.DataFrame,
    *,
    race: str,
    outdir: Path,
    coarse_class_weight_mode: str,
    min_active_families: int = 3,
    max_other_share: float = 0.75,
    max_majority_share: float = 0.75,
    min_entropy: float = 0.45,
) -> str:
    """Audit coarse target support and return the effective class-weight mode."""
    strict = _env_flag("STRICT_COARSE_AUDIT", "1")
    rectify = _env_flag("COARSE_INFLATION_RECTIFY", "1")
    selected_race = str(race or "all")
    race_values = [selected_race]
    if selected_race == "all" and "player_race" in df.columns:
        race_values = ["all"] + [r for r in ["Prot", "Terr", "Zerg"] if (df["player_race"].astype(str) == r).any()]

    summaries: list[dict] = []
    flags: list[dict] = []
    for race_name in race_values:
        race_df = df
        if race_name != "all" and "player_race" in race_df.columns:
            race_df = race_df[race_df["player_race"].astype(str) == race_name]

        for split_name in ["train", "val", "test"]:
            split_df = race_df[race_df["split"].astype(str) == split_name]
            if split_df.empty:
                flags.append({
                    "kind": "empty_split",
                    "race": race_name,
                    "split": split_name,
                    "hard_failure": True,
                })
                summaries.append({
                    "race": race_name,
                    "split": split_name,
                    "n_examples": 0,
                    "active_coarse_families": 0,
                })
                continue

            counts = split_df["coarse_family"].astype(str).value_counts()
            total = int(counts.sum())
            active = int(counts.shape[0])
            other_share = float(counts.get("Other", 0) / max(1, total))
            majority_family = str(counts.index[0])
            majority_share = float(counts.iloc[0] / max(1, total))
            entropy = _support_entropy(counts)
            summary = {
                "race": race_name,
                "split": split_name,
                "n_examples": total,
                "active_coarse_families": active,
                "other_share": other_share,
                "majority_family": majority_family,
                "majority_share": majority_share,
                "normalized_entropy": entropy,
                "counts": {str(k): int(v) for k, v in counts.items()},
            }
            summaries.append(summary)

            if active < min_active_families:
                flags.append({
                    "kind": "too_few_active_families",
                    "race": race_name,
                    "split": split_name,
                    "value": active,
                    "threshold": min_active_families,
                    "hard_failure": True,
                })
            if other_share > max_other_share:
                flags.append({
                    "kind": "other_share_too_high",
                    "race": race_name,
                    "split": split_name,
                    "value": other_share,
                    "threshold": max_other_share,
                    "hard_failure": True,
                })
            if majority_share > max_majority_share:
                flags.append({
                    "kind": "majority_share_too_high",
                    "race": race_name,
                    "split": split_name,
                    "value": majority_share,
                    "threshold": max_majority_share,
                    "hard_failure": False,
                })
            if entropy is not None and entropy < min_entropy:
                flags.append({
                    "kind": "support_entropy_too_low",
                    "race": race_name,
                    "split": split_name,
                    "value": entropy,
                    "threshold": min_entropy,
                    "hard_failure": False,
                })

    effective_mode = str(coarse_class_weight_mode or "none")
    rectifications: list[str] = []
    hard_flags = [f for f in flags if bool(f.get("hard_failure"))]
    imbalance_flags = [f for f in flags if not bool(f.get("hard_failure"))]
    if imbalance_flags and rectify and effective_mode == "none":
        effective_mode = "inverse_sqrt"
        rectifications.append("coarse_class_weight_mode=inverse_sqrt")

    failed = bool(hard_flags) or (bool(imbalance_flags) and strict and not rectifications and not rectify)
    event = {
        "phase": "pretrain",
        "race": selected_race,
        "strict": strict,
        "rectify": rectify,
        "input_coarse_class_weight_mode": str(coarse_class_weight_mode),
        "effective_coarse_class_weight_mode": effective_mode,
        "summaries": summaries,
        "flags": flags,
        "rectifications": rectifications,
        "failed": failed,
    }
    _append_coarse_audit(Path(outdir), event)

    warnings = flags + [{"kind": "rectification_applied", "detail": x} for x in rectifications]
    if warnings:
        (Path(outdir) / "hierarchy_warning.json").write_text(json.dumps({"warnings": warnings}, indent=2))

    if failed:
        reasons = ", ".join(
            f"{f.get('race')}:{f.get('split')}:{f.get('kind')}" for f in flags
        )
        raise RuntimeError(f"Coarse target audit failed before training: {reasons}")

    return effective_mode


def assert_runtime_coarse_not_inflated(
    metrics: dict,
    *,
    outdir: Path,
    phase: str,
    step: int | None,
    min_gain_over_majority: float = 0.05,
    min_balanced_accuracy: float = 0.45,
    min_macro_f1: float = 0.35,
    high_accuracy_threshold: float = 0.80,
) -> None:
    """Stop runs where high coarse accuracy is explained by collapse/majority only."""
    strict = _env_flag("STRICT_COARSE_AUDIT", "1")
    coarse_top1 = metrics.get("coarse_top1")
    if coarse_top1 is None or float(coarse_top1) < high_accuracy_threshold:
        return

    flags: list[dict] = []
    majority = metrics.get("coarse_majority_baseline_top1")
    balanced = metrics.get("coarse_balanced_accuracy")
    macro_f1 = metrics.get("coarse_f1_macro")

    if majority is not None and float(coarse_top1) - float(majority) < min_gain_over_majority:
        flags.append({
            "kind": "too_close_to_majority_baseline",
            "coarse_top1": float(coarse_top1),
            "majority_baseline": float(majority),
            "min_gain": min_gain_over_majority,
        })
    if balanced is not None and float(balanced) < min_balanced_accuracy:
        flags.append({
            "kind": "balanced_accuracy_too_low",
            "coarse_top1": float(coarse_top1),
            "balanced_accuracy": float(balanced),
            "threshold": min_balanced_accuracy,
        })
    if macro_f1 is not None and float(macro_f1) < min_macro_f1:
        flags.append({
            "kind": "macro_f1_too_low",
            "coarse_top1": float(coarse_top1),
            "coarse_f1_macro": float(macro_f1),
            "threshold": min_macro_f1,
        })

    if not flags:
        return

    event = {
        "phase": phase,
        "step": step,
        "strict": strict,
        "metrics": {
            "coarse_top1": coarse_top1,
            "coarse_majority_baseline_top1": majority,
            "coarse_balanced_accuracy": balanced,
            "coarse_f1_macro": macro_f1,
        },
        "flags": flags,
        "failed": strict,
    }
    _append_coarse_audit(Path(outdir), event)
    if strict:
        reasons = ", ".join(str(f.get("kind")) for f in flags)
        raise RuntimeError(f"Coarse accuracy inflation audit failed at {phase}={step}: {reasons}")


def build_family_maps(
    hierarchy: dict,
    action_to_id: dict[str, int],
    df: pd.DataFrame | None = None,
) -> tuple[dict[int, str], list[int], dict[int, dict[int, int]], int]:
    """Build fine-dim list and family_fine→exact_id map from hierarchy vocab.
    
    If *df* is provided, fine_dims are enlarged to cover any fine_action_id
    values present in the actual data that exceed the hierarchy-declared size.
    This prevents CUDA assertion errors from out-of-range targets.
    """
    family_to_id_raw = hierarchy.get("family_to_id", {})
    family_to_id = {str(k): int(v) for k, v in family_to_id_raw.items()}
    family_id_to_name = {int(v): str(k) for k, v in family_to_id.items()}

    if not family_id_to_name:
        raise RuntimeError("No hierarchy families found in action_vocab.")

    family_fine_to_id_raw = hierarchy.get("family_fine_to_id", {})
    family_other_token = str(hierarchy.get("family_other_token", "__OTHER__"))
    family_other_exact_raw = hierarchy.get("family_other_exact_action_id", {})

    num_families = max(family_id_to_name.keys()) + 1
    fine_dims = [1] * num_families
    family_fine_to_exact_id: dict[int, dict[int, int]] = {}

    for fam_name, fam_id in family_to_id.items():
        fam_id_i = int(fam_id)
        fine_map_raw = family_fine_to_id_raw.get(fam_name, {})
        fine_map = {str(k): int(v) for k, v in fine_map_raw.items()}

        if not fine_map:
            fine_dims[fam_id_i] = 1
            family_fine_to_exact_id[fam_id_i] = {
                0: int(family_other_exact_raw.get(fam_name, -1))
            }
            continue

        fine_dims[fam_id_i] = max(fine_map.values()) + 1
        fam_exact_map: dict[int, int] = {}
        other_exact = int(family_other_exact_raw.get(fam_name, -1))
        for fine_key, fine_id in fine_map.items():
            if fine_key == family_other_token:
                fam_exact_map[int(fine_id)] = other_exact
            else:
                fam_exact_map[int(fine_id)] = int(
                    action_to_id.get(fine_key, other_exact)
                )
        family_fine_to_exact_id[fam_id_i] = fam_exact_map

    # Reconcile fine_dims against actual data so no target is out-of-range
    if df is not None and "coarse_family_id" in df.columns and "fine_action_id" in df.columns:
        for fam_id in range(num_families):
            fam_mask = df["coarse_family_id"] == fam_id
            if not fam_mask.any():
                continue
            data_max = int(df.loc[fam_mask, "fine_action_id"].max())
            if data_max >= fine_dims[fam_id]:
                fine_dims[fam_id] = data_max + 1

    default_exact_id = 0
    return family_id_to_name, fine_dims, family_fine_to_exact_id, default_exact_id


def make_loaders(
    ds: EventWindowHierDataset,
    bs: int,
    shuffle: bool,
    workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for a hierarchical dataset."""
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        collate_fn=collate_hier_batch, num_workers=workers,
    )


# ---------------------------------------------------------------------------
# Race-specific hierarchy support
# ---------------------------------------------------------------------------

def build_race_maps(
    df: pd.DataFrame,
) -> tuple[dict[str, int], list[str]]:
    """Build race → ID mapping from the dataframe's player_race column.

    Returns:
        race_to_id: {race_str → race_id} e.g., {"Prot": 0, "Terr": 1, "Zerg": 2}
        race_order: list of unique races in order
    """
    if "player_race" not in df.columns:
        raise ValueError("DataFrame must have 'player_race' column for race routing.")
    
    unique_races = sorted(df["player_race"].unique())
    race_to_id = {race: idx for idx, race in enumerate(unique_races)}
    return race_to_id, unique_races


def build_race_family_maps(
    hierarchy: dict,
    action_to_id: dict[str, int],
) -> tuple[dict[str, dict[int, str]], dict[str, list[int]], dict[str, dict[int, dict[int, int]]], dict[str, int]]:
    """Build race-specific family maps from hierarchy.

    For each race, extract the list of coarse families and their fine actions.

    Returns:
        race_family_id_to_name: {race → {family_id → family_name}}
        race_fine_dims: {race → [fine_dim for each family]}
        race_family_fine_to_exact_id: {race → {family_id → {fine_id → exact_id}}}
        race_default_exact_id: {race → default_exact_id}
    """
    family_id_to_name, fine_dims, family_fine_to_exact_id, default_exact_id = build_family_maps(
        hierarchy, action_to_id
    )
    
    # For now, all races share the same family structure
    # In a more complex setup, we could have per-race hierarchies
    races = ["Prot", "Terr", "Zerg"]
    race_family_id_to_name = {race: dict(family_id_to_name) for race in races}
    race_fine_dims = {race: list(fine_dims) for race in races}
    race_family_fine_to_exact_id = {race: dict(family_fine_to_exact_id) for race in races}
    race_default_exact_id = {race: default_exact_id for race in races}

    return race_family_id_to_name, race_fine_dims, race_family_fine_to_exact_id, race_default_exact_id
