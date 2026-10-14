"""检查教学中必须成立的性质，而不依赖特定生成句子。"""

import math
import unittest

from attention_demo import Q, K, V, attention


class AttentionChecks(unittest.TestCase):
    def test_row_normalization_and_manual_example(self):
        _, weights, output = attention(Q, K, V)
        for row in weights:
            self.assertAlmostEqual(sum(row), 1)
        self.assertAlmostEqual(weights[0][0], 0.401112, places=5)
        self.assertAlmostEqual(output[0][0], 1.604448, places=5)

    def test_causal_mask_and_no_future_value_leakage(self):
        _, weights, output = attention(Q, K, V, causal=True)
        self.assertEqual(weights[0], [1.0, 0.0, 0.0])
        for i, row in enumerate(weights):
            self.assertAlmostEqual(sum(row), 1)
            for j in range(i + 1, len(row)):
                self.assertEqual(row[j], 0)
        changed = [list(row) for row in V]
        changed[-1] = [999.0, -999.0]
        _, _, changed_output = attention(Q, K, changed, causal=True)
        self.assertEqual(output[:2], changed_output[:2])


class TransformerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import tiny_transformer as tiny
        cls.torch, cls.tiny = torch, tiny
        cls.model, cls.report = tiny.train_model(verbose=False)

    def test_tensor_shapes_and_mask(self):
        torch, tiny = self.torch, self.tiny
        x, _ = tiny.make_batch()
        with torch.no_grad():
            logits, weights = self.model(x)
        self.assertEqual(tuple(logits.shape), (8, 6, len(tiny.VOCAB)))
        self.assertEqual(tuple(weights.shape), (8, 2, 6, 6))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(8, 2, 6))
        self.assertEqual(int(torch.count_nonzero(weights.triu(1))), 0)

    def test_future_tokens_do_not_change_earlier_predictions(self):
        torch, tiny = self.torch, self.tiny
        x, _ = tiny.make_batch()
        altered = x.clone()
        altered[:, 4:] = tiny.TOKEN_TO_ID["茶"]
        with torch.no_grad():
            original, _ = self.model(x)
            changed, _ = self.model(altered)
        torch.testing.assert_close(original[:, :4], changed[:, :4])

    def test_training_learns_and_generation_stops(self):
        tiny, report = self.tiny, self.report
        self.assertTrue(math.isfinite(report["final_loss"]))
        self.assertLess(report["final_loss"], report["initial_loss"] / 2)
        for prompt in ("我 喜欢", "你 喜欢 喝", "我 喜欢 吃"):
            generated = tiny.generate(self.model, prompt)
            self.assertTrue(generated.endswith(tiny.EOS), generated)
            self.assertIn(generated.removesuffix(" " + tiny.EOS), tiny.SENTENCES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
