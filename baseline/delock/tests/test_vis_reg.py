"""Unit tests for the DeLock visual-encoder weight-drift regularizer.

Tests `_vis_drift_l2` (the pure-pytree L2 helper added to scripts/train.py)
and the regex/filter selection. All tests run on CPU and avoid loading any
real model weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make scripts/train.py importable as a module.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Force JAX onto CPU before any other openpi import that might try to allocate GPU memory.
jax.config.update("jax_platforms", "cpu")

import flax.nnx as nnx  # noqa: E402

import openpi.shared.nnx_utils as nnx_utils  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import train as _train  # noqa: E402  (scripts/train.py)


def test_vis_drift_l2_zero_at_init():
    """L2 drift between identical pytrees must be exactly 0."""
    rng = np.random.default_rng(0)
    pre = {
        "a": jnp.asarray(rng.standard_normal((4, 4)), dtype=jnp.float32),
        "b": jnp.asarray(rng.standard_normal((3,)), dtype=jnp.float32),
    }
    cur = jax.tree.map(lambda x: x, pre)  # copy
    drift = _train._vis_drift_l2(cur, pre)
    assert float(drift) == 0.0


def test_vis_drift_l2_known_perturbation():
    """L2 drift must equal Σ ||p_cur − p_pre||² over leaves."""
    pre = {
        "a": jnp.zeros((2, 3), dtype=jnp.float32),
        "b": jnp.zeros((4,), dtype=jnp.float32),
    }
    cur = {
        "a": jnp.ones((2, 3), dtype=jnp.float32) * 2.0,  # 6 entries × 4 = 24
        "b": jnp.ones((4,), dtype=jnp.float32) * 3.0,    # 4 entries × 9 = 36
    }
    drift = _train._vis_drift_l2(cur, pre)
    assert float(drift) == pytest.approx(60.0)


def test_vis_drift_l2_empty_tree_is_zero():
    """Empty pytree → 0 (defensive: avoids jnp.stack([]) crash)."""
    drift = _train._vis_drift_l2({}, {})
    assert float(drift) == 0.0


def test_vis_drift_l2_dtype_promotion():
    """bfloat16 inputs must promote to fp32 internally so the sum is exact."""
    pre = {"a": jnp.zeros((8,), dtype=jnp.bfloat16)}
    cur = {"a": jnp.ones((8,), dtype=jnp.bfloat16)}
    drift = _train._vis_drift_l2(cur, pre)
    assert drift.dtype == jnp.float32
    assert float(drift) == pytest.approx(8.0)


def test_vis_reg_filter_matches_only_siglip_in_pi05():
    """Sanity-check the default regex against the actual pi0.5 NNX param tree.

    Asserts the regex picks up the SigLIP image tower (PaliGemma/img/...) and
    nothing in the LLM (which would silently regularize the wrong subtree).
    Uses the heavyweight `gemma_2b_lora` config to match the registered
    pi05_droid_delock — but only `eval_shape`, no actual allocation.
    """
    cfg = _config.get_config("pi05_droid_delock")

    def init(rng):
        model = cfg.model.create(rng)
        return nnx.state(model)

    shape = jax.eval_shape(init, jax.random.key(0))
    flat = jax.tree_util.tree_flatten_with_path(shape)[0]

    import re
    pat = re.compile(cfg.vis_reg_path_regex)
    matches = []
    for path, leaf in flat:
        p = "/".join(str(getattr(k, "key", k)) for k in path)
        if pat.match(p):
            matches.append((p, leaf))

    assert len(matches) > 0, "Default vis-reg regex should match SigLIP params."
    for p, _ in matches:
        assert "llm" not in p.lower(), (
            f"Vis-reg regex must not match LLM params; got: {p}"
        )
        assert "PaliGemma/img/" in p, (
            f"Vis-reg regex matched a non-SigLIP path: {p}"
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
