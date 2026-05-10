"""Smoke test: AttnCapturingPolicy routes to infer_cpg under CPG and skips attn buffer.

Avoids spinning up an actual websocket server — exercises just the routing
logic in `viz_sim/serve_policy_attn.py:AttnCapturingPolicy.infer`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))


def _load_attn_capturing_policy():
    """Lazy-import the server module without triggering its CLI side effects."""
    try:
        # Add viz_sim to path so the module imports work
        sys.path.insert(0, str(_REPO_ROOT / "viz_sim"))
        # set_jax_config is run at import time; tolerate failure under pytest.
        import importlib
        import openpi.models_pytorch.gemma_pytorch  # noqa: F401  pre-load to avoid lazy issues
        return importlib.import_module("serve_policy_attn")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"server module not importable in this env: {e}")


def _build_inner_with_recording():
    """A stand-in for the wrapped Policy that records which method was called."""
    calls = {"infer": 0, "infer_cpg": 0, "args_seen": []}

    def infer(obs):
        calls["infer"] += 1
        calls["args_seen"].append(("infer", dict(obs)))
        return {"actions": [[0.0]]}

    def infer_cpg(obs_pos, obs_neg, *, cpg_w):
        calls["infer_cpg"] += 1
        calls["args_seen"].append(("infer_cpg", dict(obs_pos), dict(obs_neg), cpg_w))
        return {"actions": [[1.0]]}

    inner = SimpleNamespace(
        infer=infer,
        infer_cpg=infer_cpg,
        metadata={},
        # AttnCapturingPolicy uses inner._input_transform for the real-text-token
        # count; supply a no-op that returns a plausible mask.
        _input_transform=lambda d: {"tokenized_prompt_mask": [1, 1, 1, 0]},
    )
    return inner, calls


def _build_gpt_buf_recorder():
    """Patch the gemma_pytorch buffer methods to be no-op recorders.

    Tracks attention and action-trajectory buffer separately so tests can
    assert on each. The action-traj buffer is *always* armed (CPG sweep
    runner depends on it); the attn buffer is only armed when CPG is OFF.
    """
    rec = {
        "attn_enable": 0, "attn_clear": 0, "attn_get": 0,
        "traj_enable": 0, "traj_clear": 0, "traj_get": 0,
    }

    class FakeGpt:
        @staticmethod
        def enable_attn_buffer():
            rec["attn_enable"] += 1

        @staticmethod
        def clear_attn_buffer():
            rec["attn_clear"] += 1

        @staticmethod
        def get_attn_buffer():
            rec["attn_get"] += 1
            return {}

        @staticmethod
        def enable_action_traj_buffer():
            rec["traj_enable"] += 1

        @staticmethod
        def clear_action_traj_buffer():
            rec["traj_clear"] += 1

        @staticmethod
        def get_action_traj_buffer():
            rec["traj_get"] += 1
            return []  # empty → server skips action_trajectory in result

        @staticmethod
        def set_perturbation(*a, **k):
            pass

        @staticmethod
        def clear_perturbation():
            pass

    return FakeGpt, rec


def test_routes_to_infer_when_no_cpg():
    server = _load_attn_capturing_policy()
    inner, calls = _build_inner_with_recording()
    fake_gpt, buf = _build_gpt_buf_recorder()

    pol = server.AttnCapturingPolicy.__new__(server.AttnCapturingPolicy)
    pol._inner = inner
    pol._gpt = fake_gpt
    pol._has_tokenizer = False

    res = pol.infer({"prompt": "do the thing"})
    assert calls["infer"] == 1
    assert calls["infer_cpg"] == 0
    # Attn buffer was enabled and cleared in the non-CPG path.
    assert buf["attn_enable"] == 1
    assert buf["attn_clear"] == 1
    # Action-trajectory buffer is always armed.
    assert buf["traj_enable"] == 1
    assert buf["traj_clear"] == 1


def test_routes_to_infer_cpg_when_both_flags_present():
    server = _load_attn_capturing_policy()
    inner, calls = _build_inner_with_recording()
    fake_gpt, buf = _build_gpt_buf_recorder()

    pol = server.AttnCapturingPolicy.__new__(server.AttnCapturingPolicy)
    pol._inner = inner
    pol._gpt = fake_gpt
    pol._has_tokenizer = False

    res = pol.infer({
        "prompt": "novel instruction",
        "prompt_neg": "trained instruction",
        "cpg_w": 1.5,
    })
    assert calls["infer_cpg"] == 1, "Must route to infer_cpg when both prompt_neg + cpg_w are set."
    assert calls["infer"] == 0

    # In CPG mode the attn buffer must NOT be touched (otherwise pos+neg
    # forwards would mash into the same global buffer).
    assert buf["attn_enable"] == 0, "Attn buffer must NOT be enabled in CPG mode."
    assert buf["attn_clear"] == 0, "Attn buffer must NOT be cleared in CPG mode."
    # Action-trajectory buffer is still armed in CPG mode — the sweep runner
    # needs the trajectory back regardless of which sampling path was taken.
    assert buf["traj_enable"] == 1, "Action-traj buffer should be armed even in CPG mode."
    assert buf["traj_clear"] == 1

    # CPG kwargs reach infer_cpg correctly: positive prompt is the original
    # `prompt`, negative is `prompt_neg`, scalar is the float.
    method, obs_pos, obs_neg, w = calls["args_seen"][0]
    assert method == "infer_cpg"
    assert obs_pos["prompt"] == "novel instruction"
    assert obs_neg["prompt"] == "trained instruction"
    assert w == 1.5
    # CPG-control keys must not leak into the obs dicts handed to the model.
    assert "prompt_neg" not in obs_pos and "cpg_w" not in obs_pos
    assert "prompt_neg" not in obs_neg and "cpg_w" not in obs_neg


def test_only_one_cpg_flag_falls_back_to_normal():
    """If only one of (prompt_neg, cpg_w) is set, treat as a malformed request
    and fall back to the standard infer path — refusing to mute one of them
    silently catches client bugs."""
    server = _load_attn_capturing_policy()
    inner, calls = _build_inner_with_recording()
    fake_gpt, buf = _build_gpt_buf_recorder()

    pol = server.AttnCapturingPolicy.__new__(server.AttnCapturingPolicy)
    pol._inner = inner
    pol._gpt = fake_gpt
    pol._has_tokenizer = False

    pol.infer({"prompt": "x", "prompt_neg": "y"})  # missing cpg_w
    assert calls["infer"] == 1
    assert calls["infer_cpg"] == 0

    pol.infer({"prompt": "x", "cpg_w": 1.0})  # missing prompt_neg
    assert calls["infer"] == 2
    assert calls["infer_cpg"] == 0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
