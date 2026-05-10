"""Unit tests for DeLock contrastive prompt guidance (CPG) inference.

Runs on CPU. Avoids loading the full PI0Pytorch model (which depends on
HuggingFace transformers + the `transformers_replace` patch); instead we
monkeypatch the policy-internal forwards so we can verify the linear
combination invariants:

    v_cpg = v_neg + w * (v_pos - v_neg)

    w = 1.0 ⇒ v_cpg ≡ v_pos
    w = 0.0 ⇒ v_cpg ≡ v_neg
    w = 0.5 ⇒ midpoint
    w > 1   ⇒ extrapolation along the contrastive direction

The math test is pure tensor algebra; the "reduces to vanilla" test
exercises sample_actions_cpg with a stub forward + denoise_step so we
can prove the dual-cache wiring agrees with single-prefix sampling
when the two prompts are the same.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))


def _cpg_combine(v_pos: torch.Tensor, v_neg: torch.Tensor, w: float) -> torch.Tensor:
    return v_neg + w * (v_pos - v_neg)


def test_cpg_w1_recovers_vpos():
    """At w=1: out = v_neg + 1*(v_pos - v_neg) = v_pos (within fp roundoff)."""
    g = torch.Generator().manual_seed(0)
    v_pos = torch.randn(4, 8, 32, generator=g)
    v_neg = torch.randn(4, 8, 32, generator=g)
    out = _cpg_combine(v_pos, v_neg, w=1.0)
    # Float subtraction-then-add introduces ~ulp roundoff; can't require bit-exact.
    assert torch.allclose(out, v_pos, atol=1e-6, rtol=1e-6)


def test_cpg_w0_recovers_vneg():
    """At w=0: out = v_neg (this case IS bit-exact since 0*x = 0 exactly)."""
    g = torch.Generator().manual_seed(0)
    v_pos = torch.randn(4, 8, 32, generator=g)
    v_neg = torch.randn(4, 8, 32, generator=g)
    out = _cpg_combine(v_pos, v_neg, w=0.0)
    assert torch.allclose(out, v_neg, atol=0, rtol=0)


def test_cpg_w_half_is_midpoint():
    g = torch.Generator().manual_seed(0)
    v_pos = torch.randn(4, 8, 32, generator=g)
    v_neg = torch.randn(4, 8, 32, generator=g)
    out = _cpg_combine(v_pos, v_neg, w=0.5)
    assert torch.allclose(out, 0.5 * (v_pos + v_neg), atol=1e-6)


def test_cpg_extrapolation_w2_is_v_pos_plus_diff():
    """w=2: out = v_neg + 2*(v_pos - v_neg) = 2*v_pos - v_neg = v_pos + (v_pos - v_neg)."""
    g = torch.Generator().manual_seed(0)
    v_pos = torch.randn(4, 8, 32, generator=g)
    v_neg = torch.randn(4, 8, 32, generator=g)
    out = _cpg_combine(v_pos, v_neg, w=2.0)
    expected = v_pos + (v_pos - v_neg)
    assert torch.allclose(out, expected, atol=1e-6)


def test_sample_actions_cpg_calls_denoise_twice_per_step():
    """When the wired sample_actions_cpg runs N denoising steps, denoise_step
    must be called 2*N times (once per cache). This pins the dual-forward
    contract — a regression that calls denoise_step once and tries to share
    a cache would silently produce wrong attention/state mixing.

    We import lazily and skip if the heavyweight transformers_replace
    machinery isn't importable on this machine.
    """
    try:
        from openpi.models_pytorch import pi0_pytorch as _pp
    except Exception as e:  # pragma: no cover
        pytest.skip(f"PI0Pytorch import not available in this environment: {e}")

    # Build a minimal stand-in for self that exercises only the methods the
    # CPG path actually calls. We bind sample_actions_cpg as an unbound method
    # against this stand-in.
    n_steps = 4
    action_horizon = 3
    action_dim = 7
    bsize = 1
    device = torch.device("cpu")

    state_t = torch.zeros(bsize, 5, dtype=torch.float32, device=device)
    images_t = [torch.zeros(bsize, 3, 8, 8, dtype=torch.float32, device=device)]
    img_masks_t = [torch.ones(bsize, dtype=torch.bool, device=device)]
    lang_pos = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.int32, device=device)
    lang_neg = torch.tensor([[4, 5, 6, 0, 0]], dtype=torch.int32, device=device)
    lang_mask = torch.ones_like(lang_pos, dtype=torch.bool)

    obs_pos = SimpleNamespace(state=state_t)
    obs_neg = SimpleNamespace(state=state_t)

    config = SimpleNamespace(action_horizon=action_horizon, action_dim=action_dim)

    denoise_calls = {"n": 0, "caches_seen": []}

    def fake_preprocess(self, obs, *, train):
        if obs is obs_pos:
            return list(images_t), list(img_masks_t), lang_pos, lang_mask, state_t
        return list(images_t), list(img_masks_t), lang_neg, lang_mask, state_t

    def fake_embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        emb = torch.zeros(bsize, 4, 8)
        pad = torch.ones(bsize, 4, dtype=torch.bool)
        att = torch.zeros(bsize, 4, dtype=torch.bool)
        return emb, pad, att

    def fake_make_att_2d(pad_masks, att_masks):
        return torch.ones(bsize, pad_masks.shape[1], pad_masks.shape[1], dtype=torch.bool)

    def fake_prepare_4d(self, m):
        return m[:, None, :, :].float()

    forward_calls = {"n": 0}

    def fake_forward(*, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, **_):
        forward_calls["n"] += 1
        cache_id = f"cache_{forward_calls['n']}"
        return None, cache_id

    pgwe = SimpleNamespace(
        forward=fake_forward,
        paligemma=SimpleNamespace(
            language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation=None)),
        ),
    )

    def fake_denoise(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        denoise_calls["n"] += 1
        denoise_calls["caches_seen"].append(past_key_values)
        # zero vector field → x_t doesn't change
        return torch.zeros_like(x_t)

    def fake_sample_noise(shape, device):
        return torch.zeros(shape, dtype=torch.float32, device=device)

    def fake_make_att_2d_global(pad, att):
        return torch.ones(pad.shape[0], pad.shape[1], pad.shape[1], dtype=torch.bool)

    self_obj = SimpleNamespace(
        config=config,
        sample_noise=fake_sample_noise,
        _preprocess_observation=lambda obs, *, train: fake_preprocess(self_obj, obs, train=train),
        embed_prefix=lambda *a, **k: fake_embed_prefix(self_obj, *a, **k),
        _prepare_attention_masks_4d=lambda m: fake_prepare_4d(self_obj, m),
        denoise_step=lambda *a, **k: fake_denoise(self_obj, *a, **k),
        paligemma_with_expert=pgwe,
    )

    with mock.patch.object(_pp, "make_att_2d_masks", side_effect=fake_make_att_2d_global):
        out = _pp.PI0Pytorch.sample_actions_cpg(
            self_obj, device, obs_pos, obs_neg, cpg_w=1.5, num_steps=n_steps,
        )

    # Two prefix forwards (one per prompt) + 2*n_steps denoise calls.
    assert forward_calls["n"] == 2, f"expected 2 prefix forwards, got {forward_calls['n']}"
    assert denoise_calls["n"] == 2 * n_steps, (
        f"expected {2 * n_steps} denoise calls, got {denoise_calls['n']}"
    )

    # Caches must alternate pos / neg — never the same cache twice in a row,
    # which would mean we accidentally fed both forwards from the same prompt.
    caches = denoise_calls["caches_seen"]
    pos_cache, neg_cache = caches[0], caches[1]
    assert pos_cache != neg_cache, "pos and neg caches must be distinct"
    for i in range(0, len(caches), 2):
        assert caches[i] == pos_cache, "even-indexed denoise calls must use the pos cache"
        assert caches[i + 1] == neg_cache, "odd-indexed denoise calls must use the neg cache"

    # Output shape sanity.
    assert out.shape == (bsize, action_horizon, action_dim)


def test_sample_actions_cpg_combination_math_in_body():
    """Pin the actual `v_cpg = v_neg + w*(v_pos - v_neg)` line inside
    `sample_actions_cpg` (not just the math helper). Previous test mocked
    denoise_step → 0 so x_t never moved; this test makes denoise_step return
    *different* nonzero tensors for pos vs neg caches and asserts the final
    x_t matches the analytic Euler integration of v_cpg.

    A regression like `v_pos + w*(v_pos - v_neg)` (wrong base) would slip
    past every other test in this file.
    """
    try:
        from openpi.models_pytorch import pi0_pytorch as _pp
    except Exception as e:  # pragma: no cover
        pytest.skip(f"PI0Pytorch import not available: {e}")

    n_steps = 5
    action_horizon = 3
    action_dim = 4
    bsize = 1
    device = torch.device("cpu")
    cpg_w = 2.0  # extrapolation regime — exposes the base-point choice

    state_t = torch.zeros(bsize, 5, dtype=torch.float32)
    images_t = [torch.zeros(bsize, 3, 4, 4, dtype=torch.float32)]
    img_masks_t = [torch.ones(bsize, dtype=torch.bool)]
    lang_pos = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    lang_neg = torch.tensor([[4, 5, 6]], dtype=torch.int32)
    lang_mask = torch.ones_like(lang_pos, dtype=torch.bool)

    obs_pos = SimpleNamespace(state=state_t)
    obs_neg = SimpleNamespace(state=state_t)
    config = SimpleNamespace(action_horizon=action_horizon, action_dim=action_dim)

    # Distinct, deterministic vector fields per cache so we can compute the
    # expected x_t analytically.
    V_POS = torch.full((bsize, action_horizon, action_dim), 0.7, dtype=torch.float32)
    V_NEG = torch.full((bsize, action_horizon, action_dim), 0.3, dtype=torch.float32)

    def fake_preprocess(obs, *, train):
        if obs is obs_pos:
            return list(images_t), list(img_masks_t), lang_pos, lang_mask, state_t
        return list(images_t), list(img_masks_t), lang_neg, lang_mask, state_t

    def fake_embed_prefix(*a, **k):
        return torch.zeros(bsize, 4, 8), torch.ones(bsize, 4, dtype=torch.bool), torch.zeros(bsize, 4, dtype=torch.bool)

    def fake_prepare_4d(m):
        return m[:, None, :, :].float()

    forward_n = {"n": 0}

    def fake_forward(*, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, **_):
        forward_n["n"] += 1
        # First forward → pos cache, second → neg cache.
        cache_id = "pos_cache" if forward_n["n"] == 1 else "neg_cache"
        return None, cache_id

    pgwe = SimpleNamespace(
        forward=fake_forward,
        paligemma=SimpleNamespace(
            language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation=None)),
        ),
    )

    def fake_denoise(state, prefix_pad_masks, past_key_values, x_t, timestep):
        if past_key_values == "pos_cache":
            return V_POS.clone()
        if past_key_values == "neg_cache":
            return V_NEG.clone()
        raise ValueError(f"unexpected cache: {past_key_values}")

    NOISE = torch.full((bsize, action_horizon, action_dim), 0.1, dtype=torch.float32)

    self_obj = SimpleNamespace(
        config=config,
        sample_noise=lambda shape, device: NOISE.clone(),
        _preprocess_observation=fake_preprocess,
        embed_prefix=fake_embed_prefix,
        _prepare_attention_masks_4d=fake_prepare_4d,
        denoise_step=fake_denoise,
        paligemma_with_expert=pgwe,
    )

    with mock.patch.object(_pp, "make_att_2d_masks",
                           side_effect=lambda pad, att: torch.ones(pad.shape[0], pad.shape[1], pad.shape[1], dtype=torch.bool)):
        out = _pp.PI0Pytorch.sample_actions_cpg(
            self_obj, device, obs_pos, obs_neg,
            cpg_w=cpg_w, num_steps=n_steps,
        )

    # Analytic expected: v_cpg = v_neg + w*(v_pos - v_neg) is constant over
    # all steps because V_POS / V_NEG are constants. Euler with dt = -1/n_steps,
    # n_steps total updates, starting from NOISE.
    v_cpg_expected = V_NEG + cpg_w * (V_POS - V_NEG)        # = -0.1 (when w=2: 0.3 + 2*(0.4) = 1.1)
    dt = -1.0 / n_steps
    expected = NOISE + n_steps * dt * v_cpg_expected

    assert torch.allclose(out, expected, atol=1e-5), (
        f"CPG body math regression: out={out[0,0,0].item()} expected={expected[0,0,0].item()}\n"
        f"  v_pos=0.7  v_neg=0.3  w={cpg_w}  v_cpg_expected={v_cpg_expected[0,0,0].item()}\n"
        f"  noise=0.1  n_steps={n_steps}  dt={dt}"
    )

    # Cross-check at w=0: x_t should evolve under v_neg only.
    forward_n["n"] = 0
    with mock.patch.object(_pp, "make_att_2d_masks",
                           side_effect=lambda pad, att: torch.ones(pad.shape[0], pad.shape[1], pad.shape[1], dtype=torch.bool)):
        out_w0 = _pp.PI0Pytorch.sample_actions_cpg(
            self_obj, device, obs_pos, obs_neg, cpg_w=0.0, num_steps=n_steps,
        )
    expected_w0 = NOISE + n_steps * dt * V_NEG
    assert torch.allclose(out_w0, expected_w0, atol=1e-5), (
        "At w=0, sample_actions_cpg must evolve under v_neg alone."
    )

    # Cross-check at w=1: x_t should evolve under v_pos only.
    forward_n["n"] = 0
    with mock.patch.object(_pp, "make_att_2d_masks",
                           side_effect=lambda pad, att: torch.ones(pad.shape[0], pad.shape[1], pad.shape[1], dtype=torch.bool)):
        out_w1 = _pp.PI0Pytorch.sample_actions_cpg(
            self_obj, device, obs_pos, obs_neg, cpg_w=1.0, num_steps=n_steps,
        )
    expected_w1 = NOISE + n_steps * dt * V_POS
    assert torch.allclose(out_w1, expected_w1, atol=1e-5), (
        "At w=1, sample_actions_cpg must evolve under v_pos alone (vanilla τ+ sampling)."
    )


def test_cpg_combine_is_linear_in_w():
    """v_cpg(w) = v_neg + w*(v_pos - v_neg) is linear in w by construction.

    Property: for any w_a, w_b and t = (w_a + w_b) / 2,
        v_cpg(t) == 0.5 * (v_cpg(w_a) + v_cpg(w_b))   (within fp roundoff)

    A regression that introduces a nonlinearity (clamp, abs, sign-flip,
    spurious sigmoid) would fail this even if the w=0/w=1 endpoint tests
    still pass — endpoints alone don't pin a linear function.
    """
    g = torch.Generator().manual_seed(7)
    v_pos = torch.randn(8, 12, 16, generator=g)
    v_neg = torch.randn(8, 12, 16, generator=g)

    for w_a, w_b in [(0.0, 1.0), (0.5, 1.5), (-0.3, 2.7), (1.0, 1.0)]:
        out_a = _cpg_combine(v_pos, v_neg, w=w_a)
        out_b = _cpg_combine(v_pos, v_neg, w=w_b)
        out_mid = _cpg_combine(v_pos, v_neg, w=(w_a + w_b) / 2)
        assert torch.allclose(out_mid, 0.5 * (out_a + out_b), atol=1e-5), (
            f"CPG must be linear in w; failed for ({w_a}, {w_b})"
        )


def test_cpg_combine_pos_neg_symmetry():
    """v_cpg(w; pos, neg) == v_cpg(1 - w; neg, pos) by algebra:
        v_neg + w(v_pos − v_neg) = v_pos + (1−w)(v_neg − v_pos).
    Proves the contrastive form is a true interpolation between the two
    conditional fields (and that swapping prompts with w↔1−w is a no-op).
    """
    g = torch.Generator().manual_seed(13)
    v_pos = torch.randn(4, 8, 8, generator=g)
    v_neg = torch.randn(4, 8, 8, generator=g)

    for w in [0.0, 0.25, 0.5, 1.0, 1.7]:
        a = _cpg_combine(v_pos, v_neg, w=w)
        b = _cpg_combine(v_neg, v_pos, w=1 - w)
        assert torch.allclose(a, b, atol=1e-5), f"pos/neg/w↔1−w symmetry failed at w={w}"


def test_sample_actions_cpg_image_mismatch_raises():
    """The CPG observation pair must agree on images / state — otherwise the
    method asserts. This guards against feeding two different camera frames
    by accident, which would defeat the whole 'only the prompt differs' premise.
    """
    try:
        from openpi.models_pytorch import pi0_pytorch as _pp
    except Exception as e:  # pragma: no cover
        pytest.skip(f"PI0Pytorch import not available: {e}")

    bsize = 1
    device = torch.device("cpu")
    state_t = torch.zeros(bsize, 5)

    obs_pos = SimpleNamespace(state=state_t)
    obs_neg = SimpleNamespace(state=state_t)
    config = SimpleNamespace(action_horizon=2, action_dim=4)

    img_pos = torch.zeros(bsize, 3, 8, 8)
    img_neg = torch.ones(bsize, 3, 8, 8)  # different!
    img_mask = torch.ones(bsize, dtype=torch.bool)

    def fake_preprocess(obs, *, train):
        if obs is obs_pos:
            return [img_pos], [img_mask], torch.tensor([[1, 2, 0]]), torch.ones(1, 3, dtype=torch.bool), state_t
        return [img_neg], [img_mask], torch.tensor([[3, 4, 0]]), torch.ones(1, 3, dtype=torch.bool), state_t

    self_obj = SimpleNamespace(
        config=config,
        sample_noise=lambda s, d: torch.zeros(s),
        _preprocess_observation=fake_preprocess,
    )
    with pytest.raises(AssertionError, match="images must match"):
        _pp.PI0Pytorch.sample_actions_cpg(
            self_obj, device, obs_pos, obs_neg, cpg_w=1.0, num_steps=1,
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
