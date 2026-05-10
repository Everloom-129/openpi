"""Smoke test: pi05_droid_delock TrainConfig is registered and valid.

Verifies the config plumbs through to the right LoRA variants and matches
the paper's Appendix B Table 4 hyperparameters where applicable.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402

jax.config.update("jax_platforms", "cpu")

import openpi.training.config as _config  # noqa: E402


def test_pi05_droid_delock_registered():
    cfg = _config.get_config("pi05_droid_delock")
    assert cfg.name == "pi05_droid_delock"


def test_pi05_droid_delock_matches_paper_appendix_b():
    cfg = _config.get_config("pi05_droid_delock")
    # Paper Appendix B Table 4 / text:
    assert cfg.batch_size == 32, "Paper: bs=32"
    assert cfg.num_train_steps == 10_000, "Paper: 10k steps"
    assert cfg.ema_decay is None, "Paper: EMA disabled"
    assert cfg.model.action_horizon == 10, "Paper: action horizon 10"
    # LoRA variants from Appendix B.
    assert cfg.model.paligemma_variant == "gemma_2b_lora", "Paper: rank-16 LoRA on Gemma_2b"
    assert cfg.model.action_expert_variant == "gemma_300m_lora", "Paper: rank-32 LoRA on Gemma_300m"
    # Pi0.5
    assert cfg.model.pi05 is True


def test_pi05_droid_delock_vis_reg_active():
    cfg = _config.get_config("pi05_droid_delock")
    assert cfg.vis_reg_lambda > 0, (
        "pi05_droid_delock must have vis_reg_lambda > 0; otherwise it's just standard LoRA SFT."
    )
    assert cfg.vis_reg_path_regex == ".*PaliGemma/img/.*"


def test_freeze_filter_freezes_llm_action_expert_keeps_vis_trainable():
    """The paper says: PaliGemma backbone frozen, LoRA adapters trainable, vis encoder
    NOT frozen (regularized instead). The freeze_filter should reflect this.

    Uses nnx.eval_shape on the model directly (not on its state) — the filter
    walks Variable types on the abstract model, which jax.eval_shape on a state
    pytree would lose.
    """
    import flax.nnx as nnx
    cfg = _config.get_config("pi05_droid_delock")

    abstract_model = nnx.eval_shape(cfg.model.create, jax.random.key(0))
    frozen = nnx.state(abstract_model, nnx.All(nnx.Param, cfg.freeze_filter)).flat_state()
    trainable = nnx.state(abstract_model, nnx.All(nnx.Param, nnx.Not(cfg.freeze_filter))).flat_state()

    frozen_paths = ["/".join(str(k) for k in path) for path in frozen.keys()]
    trainable_paths = ["/".join(str(k) for k in path) for path in trainable.keys()]

    assert len(frozen_paths) > 0, "Frozen set should be non-empty (LLM base + action expert base)."
    assert len(trainable_paths) > 0, "Trainable set should be non-empty (LoRA + vis encoder)."

    # Frozen set should contain LLM params (paper: PaliGemma backbone frozen).
    assert any("llm" in p.lower() for p in frozen_paths), (
        f"Frozen set must contain LLM params; sample: {frozen_paths[:3]}"
    )
    # Frozen set must NOT contain LoRA adapters (those train).
    assert not any("lora" in p.lower() for p in frozen_paths), (
        "LoRA adapter params must not be frozen."
    )
    # Frozen set must NOT contain vis-encoder params (those are regularized, not frozen).
    assert not any("img" in p.lower() for p in frozen_paths), (
        "Vis encoder must be trainable under DeLock (regularized via λ‖θ_v − θ_v_pre‖²)."
    )

    # Trainable set must contain vis encoder params and LoRA params.
    assert any("img" in p.lower() for p in trainable_paths), (
        f"Vis encoder params must appear in trainable set; sample: {trainable_paths[:3]}"
    )
    assert any("lora" in p.lower() for p in trainable_paths), (
        "LoRA adapter params must be trainable."
    )


if __name__ == "__main__":  # pragma: no cover
    import pytest
    pytest.main([__file__, "-v"])
