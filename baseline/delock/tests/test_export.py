"""Smoke tests for baseline/delock/export_cpg_results.py.

Generates a synthetic CPG trajectory, runs the export, and verifies the
expected artifacts (png, webp, json) appear and are non-empty / parseable.
This is the integration test that proves the result-export feature works
end to end without needing a real checkpoint or a real sim run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from baseline.delock import export_cpg_results as _exp  # noqa: E402


def _make_synthetic_trajectory(n_w=4, n_steps=10, action_horizon=8, action_dim=4, seed=0):
    """Synthesize a trajectory that obeys the CPG invariant by construction.

    For each w_i, we generate a trajectory of action chunks where the final
    chunk equals `gt + w_i * delta` — so different w values produce visibly
    different final chunks, exercising the export plotters' multi-line
    rendering.
    """
    rng = np.random.default_rng(seed)
    w_values = np.array([0.0, 0.5, 1.0, 1.5][:n_w], dtype=np.float32)
    gt = rng.standard_normal((action_horizon, action_dim)).astype(np.float32)
    delta = 0.3 * rng.standard_normal((action_horizon, action_dim)).astype(np.float32)
    noise = rng.standard_normal((1, action_horizon, action_dim)).astype(np.float32)

    traj = np.zeros((len(w_values), n_steps, action_horizon, action_dim), dtype=np.float32)
    for i, w in enumerate(w_values):
        # Start from noise, linearly anneal to gt + w*delta.
        endpoint = gt + w * delta
        for t in range(n_steps):
            alpha = (t + 1) / n_steps
            traj[i, t] = (1 - alpha) * noise[0] + alpha * endpoint
    return {
        "trajectory": traj,
        "w_values": w_values,
        "prompt_pos": "stack green block on blue block",
        "prompt_neg": "stack blue block on green block",
        "task": "BlockStacking",
        "ckpt": "delock_run0/step_10000",
        "gt_action": gt,
    }


def test_export_smoke_synthetic(tmp_path: Path):
    data = _make_synthetic_trajectory()
    npz_path = tmp_path / "cpg_run.npz"
    np.savez(npz_path, **data)

    out = tmp_path / "exports"
    paths = _exp.export_all(npz_path, out, fps=4)

    assert paths["lines_png"].exists()
    assert paths["lines_png"].stat().st_size > 1024, "PNG should be non-trivial in size."

    assert paths["denoising_webp"].exists()
    assert paths["denoising_webp"].stat().st_size > 1024, "WebP should be non-trivial in size."

    assert paths["summary_json"].exists()
    s = json.loads(paths["summary_json"].read_text())
    assert s["n_w"] == 4
    assert s["w_values"] == [0.0, 0.5, 1.0, 1.5]
    assert s["action_horizon"] == 8
    assert s["action_dim"] == 4
    assert s["prompt_pos"] == "stack green block on blue block"
    assert s["prompt_neg"] == "stack blue block on green block"
    assert "argmax_w_per_dim" in s
    assert len(s["argmax_w_per_dim"]) == 4
    # Each argmax_w entry should be one of the configured w values.
    for e in s["argmax_w_per_dim"]:
        assert e["w"] in [0.0, 0.5, 1.0, 1.5]


def test_export_rejects_missing_keys(tmp_path: Path):
    npz_path = tmp_path / "bad.npz"
    np.savez(npz_path, junk=np.zeros(3))
    with pytest.raises(KeyError, match="trajectory"):
        _exp.export_all(npz_path, tmp_path / "out")


def test_export_handles_no_gt_action(tmp_path: Path):
    """gt_action is optional; lines plot should render without it."""
    data = _make_synthetic_trajectory()
    data.pop("gt_action")
    npz_path = tmp_path / "cpg_run_nogt.npz"
    np.savez(npz_path, **data)
    paths = _exp.export_all(npz_path, tmp_path / "out_nogt")
    assert paths["lines_png"].exists()
    assert paths["denoising_webp"].exists()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
