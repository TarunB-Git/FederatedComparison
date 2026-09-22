"""Regression tests for the documented experiment-profile defaults."""

from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from run_hier_pipeline import resolve_protocol  # noqa: E402


def protocol_args(profile: str) -> Namespace:
    return Namespace(
        profile=profile,
        workers=None,
        central_workers=None,
        federated_workers=None,
        epochs=20,
        rounds=None,
        backbone_rounds=None,
        bs=None,
        clients_per_round=None,
        max_train=0,
        max_val=0,
        max_test=0,
        max_local_batches=None,
        max_eval_batches=0,
        max_client_samples=2000,
        local_weight_decay=1e-5,
        backbone_max_client_samples=0,
        backbone_local_weight_decay=0.0,
        central_weight_decay=1e-5,
        eval_bs=256,
        early_stop_patience=None,
        round_val_clients=None,
    )


class PipelineProtocolTests(unittest.TestCase):
    def test_full_profile_matches_saved_paper_runs(self) -> None:
        protocol = resolve_protocol(protocol_args("full"))

        self.assertEqual(protocol["batch_size"], 512)
        self.assertEqual(protocol["clients_per_round"], 25)
        self.assertEqual(protocol["fed_rounds"], 50)
        self.assertEqual(protocol["backbone_rounds"], 150)
        self.assertEqual(protocol["standard_max_client_samples"], 2000)
        self.assertEqual(protocol["standard_local_weight_decay"], 1e-5)
        self.assertEqual(protocol["backbone_max_client_samples"], 0)
        self.assertEqual(protocol["backbone_local_weight_decay"], 0.0)
        self.assertEqual(protocol["max_local_batches"], 75)
        self.assertEqual(protocol["early_stop_patience"], 100)
        self.assertEqual(protocol["round_val_clients"], 50)
        self.assertEqual(protocol["central_workers"], 8)
        self.assertEqual(protocol["federated_workers"], 0)

    def test_worker_override_applies_to_every_mode(self) -> None:
        args = protocol_args("smoke")
        args.workers = 3
        protocol = resolve_protocol(args)

        self.assertEqual(protocol["central_workers"], 3)
        self.assertEqual(protocol["federated_workers"], 3)


if __name__ == "__main__":
    unittest.main()
