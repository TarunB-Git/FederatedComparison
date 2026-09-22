"""Fast, dataset-free smoke tests for every supported sequence encoder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from hier_models import HierGRU  # noqa: E402


class SequenceModelShapeTests(unittest.TestCase):
    def test_all_encoders_produce_expected_logits(self) -> None:
        batch_size = 4
        sequence_length = 8
        input_dim = 12
        hidden_dim = 32
        family_dims = [3, 5, 2, 6]
        exact_classes = 17

        inputs = torch.randn(batch_size, sequence_length, input_dim)
        lengths = torch.tensor([8, 6, 4, 2], dtype=torch.long)

        for architecture in ("gru", "lstm", "transformer"):
            with self.subTest(architecture=architecture):
                model = HierGRU(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    layers=2,
                    dropout=0.0,
                    model_name=architecture,
                    num_families=len(family_dims),
                    fine_dims=family_dims,
                    num_exact_classes=exact_classes,
                    use_exact_head=True,
                ).eval()

                with torch.no_grad():
                    coarse_logits, fine_logits, exact_logits = model(inputs, lengths)

                self.assertEqual(coarse_logits.shape, (batch_size, len(family_dims)))
                self.assertEqual(len(fine_logits), len(family_dims))
                for logits, output_dim in zip(fine_logits, family_dims):
                    self.assertEqual(logits.shape, (batch_size, output_dim))
                self.assertEqual(exact_logits.shape, (batch_size, exact_classes))
                self.assertTrue(torch.isfinite(coarse_logits).all())
                self.assertTrue(torch.isfinite(exact_logits).all())


if __name__ == "__main__":
    unittest.main()
