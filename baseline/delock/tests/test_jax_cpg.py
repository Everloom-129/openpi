"""Tests for JAX `Pi0.sample_actions_cpg`.

Uses the `dummy` paligemma variant — small enough to allocate on CPU. Tests
the structural invariant: if the two observations are identical, CPG must
reduce to vanilla pos sampling regardless of `w` (since `v_pos == v_neg`
implies `v_cpg = v_neg + w*(v_pos - v_neg) == v_pos`). This catches
KV-cache mixups, prompt-leakage between paths, and anything that breaks
the basic dual-forward contract.

Heavyweight (boots a real-but-tiny Pi0 model). Marked with `@pytest.mark.slow`
so callers can skip it on a quick run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402

jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp  # noqa: E402

import openpi.models.model as _model  # noqa: E402
import openpi.models.pi0_config as _pi0c  # noqa: E402


def _build_dummy_pi05():
    cfg = _pi0c.Pi0Config(
        pi05=True,
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    rng = jax.random.key(0)
    model = cfg.create(rng)
    model.eval()
    return cfg, model


def _make_obs(cfg, batch_size=1, prompt_token_value=1):
    """A controllable observation: zero images, fixed state, deterministic
    prompt tokens (different prompts produce different tokenized_prompt)."""
    image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    images = {k: jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32) for k in image_keys}
    image_masks = {k: jnp.ones((batch_size,), dtype=jnp.bool_) for k in image_keys}
    state = jnp.zeros((batch_size, cfg.action_dim), dtype=jnp.float32)
    prompt = jnp.full((batch_size, cfg.max_token_len), prompt_token_value, dtype=jnp.int32)
    prompt_mask = jnp.ones((batch_size, cfg.max_token_len), dtype=jnp.bool_)
    return _model.Observation(
        images=images, image_masks=image_masks, state=state,
        tokenized_prompt=prompt, tokenized_prompt_mask=prompt_mask,
    )


@pytest.mark.slow
def test_jax_cpg_identical_prompts_reduces_to_vanilla():
    """If τ⁺ == τ⁻, then v_pos == v_neg per-step, so v_cpg ≡ v_pos for any w.
    Therefore sample_actions_cpg(obs, obs, w=anything, noise=N) must equal
    sample_actions(obs, noise=N) bit-exact (no extra arithmetic on the
    interpolation collapses cleanly because the diff term is identically 0).
    """
    cfg, model = _build_dummy_pi05()
    obs = _make_obs(cfg)
    noise = jnp.asarray(
        np.random.default_rng(1).standard_normal(
            (1, cfg.action_horizon, cfg.action_dim)
        ),
        dtype=jnp.float32,
    )

    rng = jax.random.key(42)
    vanilla = model.sample_actions(rng, obs, num_steps=4, noise=noise)
    for w in [0.0, 0.5, 1.0, 2.5]:
        cpg_out = model.sample_actions_cpg(
            rng, obs, obs, cpg_w=w, num_steps=4, noise=noise,
        )
        assert vanilla.shape == cpg_out.shape == (1, cfg.action_horizon, cfg.action_dim)
        # When v_pos == v_neg, v_cpg == v_pos. With identical noise + RNG,
        # the trajectory is deterministic.
        assert jnp.allclose(cpg_out, vanilla, atol=1e-5), (
            f"CPG with identical prompts must match vanilla sampling at w={w}"
        )


@pytest.mark.slow
def test_jax_cpg_different_prompts_diverges_from_vanilla():
    """If τ⁺ ≠ τ⁻ AND w ≠ 1, the CPG output should differ from vanilla
    sampling on either prompt — proves the dual-forward path is actually
    using both KV caches rather than silently reusing one.

    NOTE: this is informative only if the dummy model can actually produce
    different outputs for the two prompts in the first place. The dummy
    paligemma variant (4 layers / 64 width / 16 head_dim, bf16) often
    can't distinguish two single-token-fill prompts after attention, so
    we add a precondition skip in that case. The substantive correctness
    test is `test_jax_cpg_w1_matches_vanilla_pos_sampling` plus the
    PyTorch body-math test (test_cpg.py) — together they pin the
    combination math at every w.
    """
    cfg, model = _build_dummy_pi05()
    # Use a prompt with multiple distinct tokens so the embedding-then-attention
    # pipeline has more signal to differentiate.
    obs_a = _make_obs(cfg, prompt_token_value=1)
    obs_b = _make_obs(cfg, prompt_token_value=2)

    noise = jnp.asarray(
        np.random.default_rng(1).standard_normal(
            (1, cfg.action_horizon, cfg.action_dim)
        ),
        dtype=jnp.float32,
    )
    rng = jax.random.key(0)

    out_pos_only = model.sample_actions(rng, obs_a, num_steps=4, noise=noise)
    out_neg_only = model.sample_actions(rng, obs_b, num_steps=4, noise=noise)

    # Precondition: the dummy model must distinguish the two prompts at all
    # — otherwise CPG can't either, and the test is meaningless.
    if jnp.allclose(out_pos_only, out_neg_only, atol=1e-4):
        pytest.skip(
            "Dummy model's bf16 attention collapsed two single-token-fill prompts "
            "to identical outputs; CPG divergence cannot be tested with this "
            "model size. The mathematical invariants are pinned by "
            "test_jax_cpg_w1_matches_vanilla_pos_sampling and the PyTorch "
            "body-math test instead."
        )

    out_cpg = model.sample_actions_cpg(rng, obs_a, obs_b, cpg_w=2.0, num_steps=4, noise=noise)

    # If the model genuinely sees the prompts as different, CPG at w=2 must
    # produce yet another distinct output (extrapolation along v_pos − v_neg).
    assert not jnp.allclose(out_cpg, out_pos_only, atol=1e-3), (
        "CPG at w=2 should differ from vanilla pos-only sampling."
    )
    assert not jnp.allclose(out_cpg, out_neg_only, atol=1e-3), (
        "CPG at w=2 should differ from vanilla neg-only sampling."
    )


@pytest.mark.slow
def test_jax_cpg_w1_matches_vanilla_pos_sampling():
    """At w=1: v_cpg = v_neg + 1*(v_pos - v_neg) = v_pos exactly.
    So sample_actions_cpg(rng, obs_pos, obs_neg, w=1) should match
    sample_actions(rng, obs_pos) regardless of obs_neg."""
    cfg, model = _build_dummy_pi05()
    obs_pos = _make_obs(cfg, prompt_token_value=1)
    obs_neg = _make_obs(cfg, prompt_token_value=2)
    noise = jnp.asarray(
        np.random.default_rng(1).standard_normal(
            (1, cfg.action_horizon, cfg.action_dim)
        ),
        dtype=jnp.float32,
    )
    rng = jax.random.key(0)

    out_vanilla = model.sample_actions(rng, obs_pos, num_steps=4, noise=noise)
    out_cpg = model.sample_actions_cpg(rng, obs_pos, obs_neg, cpg_w=1.0, num_steps=4, noise=noise)

    # At w=1 the dual-forward still happens (cost burned), but the math
    # collapses to v_pos. Tolerance accommodates fp32→bf16 inside the
    # transformer and the extra arithmetic in the CPG path.
    assert jnp.allclose(out_cpg, out_vanilla, atol=1e-3), (
        "At w=1, CPG must match vanilla pos-only sampling."
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "-m", "slow"])
