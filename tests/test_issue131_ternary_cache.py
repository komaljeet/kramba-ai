"""Issue #131 -- inference-time dequantized-weight cache on TernaryLinear.

Author: Rambarun Komaljeet

The cache is a pure speed change (measured ~2x on EdgeHVN decode), so the tests
that matter are not "is it faster" but "is it still the same model". The
dangerous failure mode is silent staleness: serving weights from before an
optimizer step while the loss curve still looks plausible. Issue #117 was
exactly that bug and no aggregate-loss test caught it, so the staleness check
here asserts a specific weight's effect survives a train/eval round trip
directly.

CPU-only and fast by design -- this is a correctness suite, not a benchmark.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.ternary import TernaryLinear  # noqa: E402


def _layer(seed: int = 0, in_f: int = 16, out_f: int = 8) -> TernaryLinear:
    torch.manual_seed(seed)
    return TernaryLinear(in_f, out_f, bias=True)


class TernaryCacheTests(unittest.TestCase):

    def test_eval_cache_is_bit_identical(self):
        """The cached value must equal the freshly quantized one, exactly.

        This is the known-answer cell: caching a deterministic function of
        frozen weights is a no-op, so anything other than an exact match means
        the cache is wrong and any speed number measured with it is void.
        """
        layer = _layer().eval()
        x = torch.randn(4, 16)
        with torch.no_grad():
            first = layer(x)      # populates the cache
            second = layer(x)     # reads the cache
        self.assertIsNotNone(layer._wcache, "cache was never populated")
        self.assertTrue(torch.equal(first, second))

        layer.invalidate_cache()
        with torch.no_grad():
            fresh = layer(x)
        self.assertTrue(torch.equal(first, fresh),
                        "cached output differs from a fresh recompute")

    def test_weight_update_survives_train_eval_cycle(self):
        """#117-style staleness guard: a trained weight must reach inference.

        Populate the cache in eval, run a real optimizer step, return to eval,
        and require the output to CHANGE. If the cache were not invalidated on
        the mode transition, this returns the pre-training value and every
        served token would silently use untrained weights.
        """
        layer = _layer().eval()
        x = torch.randn(4, 16)
        with torch.no_grad():
            before = layer(x).clone()

        layer.train()
        opt = torch.optim.SGD(layer.parameters(), lr=5.0)
        for _ in range(5):
            opt.zero_grad()
            layer(x).pow(2).mean().backward()
            opt.step()

        layer.eval()
        with torch.no_grad():
            after = layer(x)

        self.assertFalse(torch.equal(before, after),
                         "output unchanged after training -- the cache served "
                         "stale pre-training weights (the issue #117 failure)")

        layer.invalidate_cache()
        with torch.no_grad():
            recomputed = layer(x)
        self.assertTrue(torch.equal(after, recomputed),
                        "post-training cache does not match a fresh recompute")

    def test_training_path_is_uncached_and_keeps_ste_gradient(self):
        """Training must never read the cache; the STE gradient must survive."""
        layer = _layer().eval()
        x = torch.randn(4, 16)
        with torch.no_grad():
            layer(x)  # populate

        layer.train()
        self.assertIsNone(layer._wcache, "train() did not invalidate the cache")

        out = layer(x)
        out.pow(2).mean().backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertGreater(float(layer.weight.grad.abs().sum()), 0.0,
                           "STE gradient did not reach the shadow weight")
        self.assertIsNone(layer._wcache,
                          "a training forward populated the inference cache")

    def test_grad_enabled_eval_is_uncached(self):
        """Eval WITH grad enabled must stay on the live path.

        A detached cached weight would silently cut the graph for anyone
        differentiating through an eval-mode forward.
        """
        layer = _layer().eval()
        x = torch.randn(4, 16, requires_grad=True)
        out = layer(x)               # grad enabled, eval mode
        self.assertIsNone(layer._wcache)
        out.pow(2).mean().backward()
        self.assertIsNotNone(x.grad)

    def test_cache_never_enters_state_dict(self):
        layer = _layer().eval()
        with torch.no_grad():
            layer(torch.randn(2, 16))
        self.assertIsNotNone(layer._wcache)
        keys = set(layer.state_dict().keys())
        self.assertFalse(any("wcache" in k for k in keys),
                         f"cache leaked into state_dict: {keys}")

    def test_load_state_dict_invalidates(self):
        src, dst = _layer(seed=1), _layer(seed=2)
        x = torch.randn(4, 16)
        dst.eval()
        with torch.no_grad():
            dst(x)  # populate with seed-2 weights
        dst.load_state_dict(src.state_dict())
        self.assertIsNone(dst._wcache, "load_state_dict left a stale cache")

        src.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(dst(x), src(x)),
                            "loaded model does not match its source")

    def test_dtype_change_invalidates(self):
        layer = _layer().eval()
        with torch.no_grad():
            layer(torch.randn(4, 16))
            cached = layer._wcache
            self.assertEqual(cached.dtype, torch.float32)
            layer(torch.randn(4, 16).to(torch.float64).float())
            layer_out = layer(torch.randn(4, 16))
        self.assertEqual(layer._wcache.dtype, torch.float32)
        self.assertEqual(layer_out.dtype, torch.float32)

    def test_freeze_invalidates_and_stays_consistent(self):
        layer = _layer().eval()
        x = torch.randn(4, 16)
        with torch.no_grad():
            layer(x)
        layer.freeze()
        self.assertIsNone(layer._wcache, "freeze() left a stale cache")
        with torch.no_grad():
            a = layer(x)
            b = layer(x)
        self.assertTrue(torch.equal(a, b))

    def test_in_eval_weight_surgery_needs_explicit_invalidation(self):
        """Documents the cache's one real boundary, rather than hiding it.

        Mutating a weight in place while STAYING in eval mode is not observable
        by any hook, so the cache will serve the old value until
        ``invalidate_cache()`` is called. In practice weights change via the
        optimizer (train mode) or ``load_state_dict`` (hooked), both covered
        above -- but anyone doing direct weight surgery must invalidate.
        """
        layer = _layer().eval()
        x = torch.randn(4, 16)
        with torch.no_grad():
            before = layer(x).clone()
            layer.weight.mul_(5.0)
            still_cached = layer(x)
        self.assertTrue(torch.equal(before, still_cached),
                        "boundary changed -- update the docs if so")

        layer.invalidate_cache()
        with torch.no_grad():
            after = layer(x)
        self.assertFalse(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main(verbosity=2)
