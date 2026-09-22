"""Regression tests for the self-contained action hierarchy implementation."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import hierarchy  # noqa: E402


class HierarchyTests(unittest.TestCase):
    def test_legacy8_labels_known_and_unknown_actions(self) -> None:
        events = pd.DataFrame(
            {
                "action_key": ["41:0", "172:0", "999:0"],
                "action_id": [0, 1, 2],
                "player_race": ["Prot", "Prot", "Prot"],
                "split": ["train", "train", "test"],
            }
        )
        family_order = hierarchy.taxonomy_families("legacy8")
        labeled, mapping = hierarchy.build_hierarchy_labels(
            events,
            family_top_k=0,
            family_other_token="__OTHER__",
            link_family_rules=hierarchy.default_link_family_rules("legacy8"),
            family_order=family_order,
            keep_all_fine_actions=True,
            taxonomy="legacy8",
            separate_warp_in=False,
        )

        self.assertEqual(labeled["coarse_family"].tolist(), ["Build", "Train", "Other"])
        self.assertEqual(mapping["family_order"], family_order)
        self.assertEqual(mapping["hierarchy_taxonomy"], "legacy8")

    def test_public_entrypoints_load_from_a_fresh_checkout(self) -> None:
        for script in (
            "prepare.py",
            "run_hier_pipeline.py",
            "hier_centralized.py",
            "hier_fedavg.py",
            "hier_fedprox.py",
            "hier_backbone_head_race.py",
            "hier_compare.py",
        ):
            with self.subTest(script=script):
                completed = subprocess.run(
                    [sys.executable, str(CODE_DIR / script), "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
