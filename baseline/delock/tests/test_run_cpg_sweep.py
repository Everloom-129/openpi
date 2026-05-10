"""Smoke tests for baseline/delock/run_cpg_sweep.py.

Stubs the policy.infer() call so we can verify the sweep stacking + npz
schema without needing a real websocket server or model checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from baseline.delock import export_cpg_results as _exp  # noqa: E402
from baseline.delock import run_cpg_sweep as _sweep  # noqa: E402


def _make_stub_policy(num_steps=10, action_horizon=8, action_dim=4):
    """Stub a server: returns a deterministic action_trajectory keyed off cpg_w
    so we can verify the sweep ordering and stacking are right."""
    calls = []

    def policy_infer(obs):
        calls.append(dict(obs))
        w = float(obs["cpg_w"])
        # Trajectory anneals from noise to a w-scaled target so each w produces
        # a visually different curve — useful for the export render to look real.
        traj = np.empty((num_steps, action_horizon, action_dim), dtype=np.float32)
        for t in range(num_steps):
            alpha = (t + 1) / num_steps
            traj[t] = (1 - alpha) * np.full((action_horizon, action_dim), 0.1, dtype=np.float32) + \
                      alpha * (w * np.full((action_horizon, action_dim), 0.5, dtype=np.float32))
        return {"actions": np.zeros((action_horizon, action_dim), dtype=np.float32),
                "action_trajectory": traj}

    return policy_infer, calls


def test_sweep_stacks_correctly():
    policy_infer, calls = _make_stub_policy()
    base_obs = {"prompt": "ignored — overwritten by sweep", "state": np.zeros(8)}
    w_values = [0.0, 0.5, 1.0, 1.5, 2.0]

    res = _sweep.sweep_cpg(
        policy_infer=policy_infer,
        base_obs=base_obs,
        w_values=w_values,
        prompt_pos="τ+ novel",
        prompt_neg="τ- trained",
        task="BlockStacking",
        ckpt="run0/step_10000",
    )
    assert res.trajectory.shape == (5, 10, 8, 4)
    assert res.trajectory.dtype == np.float32
    assert np.allclose(res.w_values, np.asarray(w_values, dtype=np.float32))
    assert res.prompt_pos == "τ+ novel"
    assert res.prompt_neg == "τ- trained"

    # Each call must have been issued with prompt_pos and the right w.
    assert len(calls) == len(w_values)
    for c, w in zip(calls, w_values):
        assert c["prompt"] == "τ+ novel"
        assert c["prompt_neg"] == "τ- trained"
        assert c["cpg_w"] == float(w)


def test_sweep_npz_round_trips_through_exporter(tmp_path: Path):
    """End-to-end: sweep → npz → exporter renders all 3 artifacts."""
    policy_infer, _ = _make_stub_policy(num_steps=8, action_horizon=6, action_dim=3)
    base_obs = {"prompt": "x", "state": np.zeros(8)}

    res = _sweep.sweep_cpg(
        policy_infer=policy_infer,
        base_obs=base_obs,
        w_values=[0.0, 1.0, 2.0],
        prompt_pos="novel",
        prompt_neg="trained",
        task="SmokeTask",
    )
    npz_path = res.to_npz(tmp_path / "sweep.npz")
    paths = _exp.export_all(npz_path, tmp_path / "out", fps=4)
    assert paths["lines_png"].exists() and paths["lines_png"].stat().st_size > 1024
    assert paths["denoising_webp"].exists() and paths["denoising_webp"].stat().st_size > 1024
    assert paths["summary_json"].exists()


def test_sweep_rejects_missing_action_trajectory():
    """If the server forgets to ship action_trajectory, fail loud (not silent)."""
    def bad_infer(obs):
        return {"actions": np.zeros((4, 2))}  # no action_trajectory

    with pytest.raises(ValueError, match="action_trajectory"):
        _sweep.sweep_cpg(
            policy_infer=bad_infer,
            base_obs={"prompt": "x", "state": np.zeros(8)},
            w_values=[1.0],
            prompt_pos="a",
            prompt_neg="b",
        )


def test_sweep_rejects_inconsistent_shapes():
    """If trajectories differ in shape between calls, fail loud."""
    def varying_infer(obs):
        w = float(obs["cpg_w"])
        if w == 0.0:
            return {"action_trajectory": np.zeros((10, 8, 4), dtype=np.float32)}
        return {"action_trajectory": np.zeros((10, 6, 4), dtype=np.float32)}  # H differs

    with pytest.raises(ValueError, match="share shape"):
        _sweep.sweep_cpg(
            policy_infer=varying_infer,
            base_obs={"prompt": "x"},
            w_values=[0.0, 1.0],
            prompt_pos="a",
            prompt_neg="b",
        )


def test_sweep_rejects_empty_w_values():
    with pytest.raises(ValueError, match="non-empty"):
        _sweep.sweep_cpg(
            policy_infer=lambda obs: {"action_trajectory": np.zeros((1, 1, 1))},
            base_obs={},
            w_values=[],
            prompt_pos="a",
            prompt_neg="b",
        )


def test_sweep_does_not_mutate_base_obs():
    """The sweep must not mutate the caller's obs dict — it overwrites prompt
    fields per-call but each call should get its own copy."""
    policy_infer, calls = _make_stub_policy()
    base_obs = {"prompt": "ORIGINAL", "state": np.zeros(8)}
    _sweep.sweep_cpg(
        policy_infer=policy_infer,
        base_obs=base_obs,
        w_values=[0.0, 1.0],
        prompt_pos="τ+",
        prompt_neg="τ-",
    )
    # Caller's dict still has the original prompt and no CPG fields.
    assert base_obs["prompt"] == "ORIGINAL"
    assert "prompt_neg" not in base_obs
    assert "cpg_w" not in base_obs


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
