#!/usr/bin/env python3
"""Generate README result figures from the checked-in paper result matrix."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "paper_results.csv"
FIGURES_DIR = ROOT / "figures"

RACES = ("Prot", "Terr", "Zerg")
ARCHITECTURES = ("GRU", "LSTM", "Transformer")
PARADIGMS = ("Centralized", "FedAvg", "FedProx")
COLORS = {
    "Centralized": "#4C78A8",
    "FedAvg": "#F58518",
    "FedProx": "#54A24B",
}


def load_rows() -> list[dict[str, str]]:
    with RESULTS_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30:
        raise ValueError(f"Expected the 30 paper runs, found {len(rows)}")
    return rows


def plot_metric(
    rows: list[dict[str, str]],
    *,
    metric: str,
    title: str,
    ylabel: str,
    output_name: str,
    ymax: float,
) -> None:
    lookup = {
        (row["race"], row["architecture"], row["paradigm"]): float(row[metric]) * 100
        for row in rows
        if row["race"] in RACES and row[metric]
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), sharey=True)
    x = np.arange(len(ARCHITECTURES))
    width = 0.25

    for axis, race in zip(axes, RACES):
        for offset, paradigm in enumerate(PARADIGMS):
            values = [lookup[(race, architecture, paradigm)] for architecture in ARCHITECTURES]
            positions = x + (offset - 1) * width
            bars = axis.bar(
                positions,
                values,
                width,
                label=paradigm,
                color=COLORS[paradigm],
                edgecolor="white",
                linewidth=0.7,
            )
            axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=7, rotation=90)

        axis.set_title({"Prot": "Protoss", "Terr": "Terran", "Zerg": "Zerg"}[race], weight="bold")
        axis.set_xticks(x, ARCHITECTURES)
        axis.set_ylim(0, ymax)
        axis.grid(axis="y", alpha=0.22, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(title, y=1.06, fontsize=14, weight="bold")
    fig.text(
        0.5,
        -0.01,
        "Values are test-set percentages from one fixed-seed run per configuration.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / output_name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    plot_metric(
        rows,
        metric="exact_direct_top1",
        title="Exact Direct Top-1 Accuracy",
        ylabel="Accuracy (%)",
        output_name="exact_direct_top1.png",
        ymax=50,
    )
    plot_metric(
        rows,
        metric="macro_f1_direct",
        title="Direct Macro-F1 Under Severe Class Imbalance",
        ylabel="Macro-F1 (%)",
        output_name="macro_f1_direct.png",
        ymax=13,
    )
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
