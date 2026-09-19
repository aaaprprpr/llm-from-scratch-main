import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pretrain.train_model import estimate_loss, sample_eval_window_starts


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x, use_cache=False):
        self.inputs.append(x.clone())
        logits = F.one_hot(x % 5, 5).float() * 3
        return logits, None


class EvaluationTests(unittest.TestCase):
    def test_sampling_spans_file_without_overlap_or_global_rng_changes(self):
        np.random.seed(21)
        before = np.random.get_state()
        starts = sample_eval_window_starts(100_001, 100, 32, seed=42)
        self.assertEqual(starts, sample_eval_window_starts(100_001, 100, 32, seed=42))
        self.assertNotEqual(starts, sample_eval_window_starts(100_001, 100, 32, seed=43))
        self.assertEqual(len(set(starts)), 32)
        for index, start in enumerate(starts):
            self.assertGreaterEqual(start // 100, index * 1000 // 32)
            self.assertLess(start // 100, (index + 1) * 1000 // 32)
        self.assertTrue(all(b - a >= 100 for a, b in zip(starts, starts[1:])))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_small_data_is_not_repeated_to_fill_budget(self):
        self.assertEqual(sample_eval_window_starts(36, 10, 32), [0, 10, 20])
        with self.assertRaises(ValueError):
            sample_eval_window_starts(10, 10, 1)

    def test_labels_token_weighting_and_batch_size_invariance(self):
        data = np.array([0, 0, 1, 2, 4, 4, 4, 4, 3, 1, 2, 0, 2, 3, 4, 1])
        starts = [0, 4, 8]
        model = RecordingModel().train()
        actual = estimate_loss(model, data, 2, 4, "cpu", 2, window_starts=starts)
        self.assertTrue(model.training)
        x = torch.cat(model.inputs)
        expected_x = torch.tensor(np.stack([data[s:s+4] for s in starts]))
        y = torch.tensor(np.stack([data[s+1:s+5] for s in starts]))
        self.assertTrue(torch.equal(x, expected_x))
        expected = F.cross_entropy((F.one_hot(x % 5, 5).float()*3).flatten(0, 1), y.flatten())
        self.assertAlmostEqual(actual, expected.item(), places=6)
        other = estimate_loss(model, data, 1, 4, "cpu", 3, window_starts=starts)
        self.assertAlmostEqual(actual, other, places=6)

    def test_eval_mode_restored_after_forward_error(self):
        class BrokenModel(nn.Module):
            def forward(self, *args, **kwargs):
                raise RuntimeError("probe")
        model = BrokenModel().train()
        with self.assertRaisesRegex(RuntimeError, "probe"):
            estimate_loss(model, np.arange(101), 2, 10, "cpu", 2)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
