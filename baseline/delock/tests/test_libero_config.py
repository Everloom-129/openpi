"""Tests for the LIBERO DeLock TrainConfig.

Verifies the config matches paper Appendix B and that the freeze filter
behaves the same way as the DROID DeLock variant: LLM + action-expert
base frozen, LoRA + vis encoder trainable.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402

jax.config.update("jax_platforms", "cpu")

import flax.nnx as nnx  # noqa: E402

import openpi.training.config as _config  # noqa: E402


def test_pi05_libero_delock_registered():
    cfg = _config.get_config("pi05_libero_delock")
    assert cfg.name == "pi05_libero_delock"


def test_pi05_libero_delock_matches_paper_appendix_b():
    cfg = _config.get_config("pi05_libero_delock")
    assert cfg.batch_size == 32, "Paper Appendix B: bs=32"
    assert cfg.num_train_steps == 10_000, "Paper Appendix B: 10k steps"
    assert cfg.ema_decay is None, "Paper Appendix B: EMA disabled"
    assert cfg.model.action_horizon == 10, "Paper Appendix B: horizon=10"
    assert cfg.model.pi05 is True, "Paper: starts from pi0.5-BASE"
    assert cfg.model.paligemma_variant == "gemma_2b_lora", "Paper: r=16 LoRA on Gemma_2b"
    assert cfg.model.action_expert_variant == "gemma_300m_lora", "Paper: r=32 LoRA on Gemma_300m"
    # LR schedule: cosine warmup 1k, peak 5e-5, decay over 50k → effectively
    # constant 5e-5 after warmup over the 10k training horizon.
    assert cfg.lr_schedule.warmup_steps == 1_000
    assert cfg.lr_schedule.peak_lr == 5e-5


def test_pi05_libero_delock_vis_reg_active():
    cfg = _config.get_config("pi05_libero_delock")
    assert cfg.vis_reg_lambda > 0
    assert cfg.vis_reg_path_regex == ".*PaliGemma/img/.*"


def test_pi05_libero_delock_freeze_filter():
    """Same shape as the DROID variant: LLM + action-expert base frozen,
    LoRA + vis encoder trainable. Pinned here separately because the LIBERO
    variant has slightly different model kwargs (discrete_state_input=False)
    that could in principle change the freeze filter."""
    cfg = _config.get_config("pi05_libero_delock")
    abstract_model = nnx.eval_shape(cfg.model.create, jax.random.key(0))
    frozen = nnx.state(abstract_model, nnx.All(nnx.Param, cfg.freeze_filter)).flat_state()
    trainable = nnx.state(abstract_model, nnx.All(nnx.Param, nnx.Not(cfg.freeze_filter))).flat_state()

    frozen_paths = ["/".join(str(k) for k in path) for path in frozen.keys()]
    trainable_paths = ["/".join(str(k) for k in path) for path in trainable.keys()]

    assert any("llm" in p.lower() for p in frozen_paths), (
        "LLM base must be frozen (paper §3.2)."
    )
    assert not any("img" in p.lower() for p in frozen_paths), (
        "Vis encoder must NOT be frozen — it's regularized."
    )
    assert not any("lora" in p.lower() for p in frozen_paths), (
        "LoRA adapters must not be frozen."
    )
    assert any("img" in p.lower() for p in trainable_paths)
    assert any("lora" in p.lower() for p in trainable_paths)


def test_libero_data_uses_pi_libero_dataset():
    """Catches drift if someone swaps the dataset out (e.g. to a stale local copy)."""
    cfg = _config.get_config("pi05_libero_delock")
    assert cfg.data.repo_id == "physical-intelligence/libero"


if __name__ == "__main__":  # pragma: no cover
    import pytest
    pytest.main([__file__, "-v"])
