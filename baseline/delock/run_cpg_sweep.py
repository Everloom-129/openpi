"""Sweep DeLock contrastive prompt guidance over a list of `w` values.

For a single observation, calls the policy server N times — once per `w` —
and stacks the per-denoising-step action trajectories into a single npz
with the schema consumed by `baseline/delock/export_cpg_results.py`.

Output schema:
    {
        "trajectory": float32 (n_w, n_steps, action_horizon, action_dim),
        "w_values":   float32 (n_w,),
        "prompt_pos": str,
        "prompt_neg": str,
        "task": str,            # optional
        "ckpt": str,            # optional
        "gt_action": ...,       # optional (passed-through if present in obs)
    }

Server contract (see `viz_sim/serve_policy_attn.py:AttnCapturingPolicy.infer`):
- When `prompt_neg` + `cpg_w` are present in the obs, the server runs
  `Policy.infer_cpg` and ships back `result["action_trajectory"]` of shape
  `(num_steps, action_horizon, action_dim)`.

Pure-Python core (`sweep_cpg`) takes any callable matching the
`policy.infer(obs) -> dict` shape, so it can be tested without the
websocket layer (see `tests/test_run_cpg_sweep.py`).

CLI:
    .venv/bin/python baseline/delock/run_cpg_sweep.py \
        --obs-pickle obs.pkl \
        --prompt-pos "stack green block on blue block" \
        --prompt-neg "stack blue block on green block" \
        --w-values 0.0 0.5 1.0 1.5 2.0 \
        --host localhost --port 8000 \
        --out baseline/delock/results/run_42/cpg_run.npz

To grab `obs.pkl` from a real sim run:
    # Inside run_pi0_policy_sim.py at any env step where you want the sweep
    # observation, dump the constructed `policy_obs` to pickle:
    #     import pickle; pickle.dump(policy_obs, open("obs.pkl", "wb"))
"""
from __future__ import annotations

import argparse
import dataclasses
import pickle
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PolicyInfer = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclasses.dataclass
class SweepResult:
    trajectory: np.ndarray  # (n_w, n_steps, action_horizon, action_dim) float32
    w_values: np.ndarray    # (n_w,) float32
    prompt_pos: str
    prompt_neg: str
    task: str | None = None
    ckpt: str | None = None
    gt_action: np.ndarray | None = None  # (action_horizon, action_dim) float32

    def to_npz(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "trajectory": self.trajectory,
            "w_values": self.w_values,
            "prompt_pos": np.asarray(self.prompt_pos),
            "prompt_neg": np.asarray(self.prompt_neg),
        }
        if self.task is not None:
            kwargs["task"] = np.asarray(self.task)
        if self.ckpt is not None:
            kwargs["ckpt"] = np.asarray(self.ckpt)
        if self.gt_action is not None:
            kwargs["gt_action"] = self.gt_action.astype(np.float32)
        np.savez(path, **kwargs)
        return path


def sweep_cpg(
    policy_infer: PolicyInfer,
    base_obs: Mapping[str, Any],
    w_values: Sequence[float],
    *,
    prompt_pos: str,
    prompt_neg: str,
    task: str | None = None,
    ckpt: str | None = None,
    gt_action: np.ndarray | None = None,
) -> SweepResult:
    """Run one infer call per `w` and stack `action_trajectory` outputs.

    Args:
        policy_infer: callable that maps `obs -> result`. Must support the
            DeLock CPG contract: when `obs` carries `prompt_neg` + `cpg_w`,
            return `result["action_trajectory"]` of shape
            `(num_steps, action_horizon, action_dim)`.
        base_obs: the observation to evaluate at — typically taken from one
            env step. The `prompt`, `prompt_neg`, and `cpg_w` fields are
            overwritten per call.
        w_values: ordered guidance scales to sweep.
        prompt_pos / prompt_neg: novel / trained instructions.
        task / ckpt: optional metadata embedded in the resulting npz.
        gt_action: optional GT action chunk to embed for export-time overlay.

    Returns:
        SweepResult — call .to_npz(path) to persist.

    Raises:
        ValueError: empty w_values or missing `action_trajectory` in any
            response (means the server isn't returning the field, e.g.
            an old server build).
    """
    if not w_values:
        raise ValueError("w_values must be non-empty.")

    trajectories: list[np.ndarray] = []
    for w in w_values:
        obs = dict(base_obs)
        obs["prompt"] = prompt_pos
        obs["prompt_neg"] = prompt_neg
        obs["cpg_w"] = float(w)
        result = policy_infer(obs)
        traj = result.get("action_trajectory")
        if traj is None:
            raise ValueError(
                f"policy.infer() did not return 'action_trajectory' for w={w}. "
                "Make sure you're running a serve_policy_attn.py with the "
                "action-trajectory return wired in."
            )
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim != 3:
            raise ValueError(
                f"action_trajectory must have shape (num_steps, H, D); got {traj.shape}."
            )
        trajectories.append(traj)

    # Sanity: all trajectories must agree on (num_steps, H, D).
    shapes = {t.shape for t in trajectories}
    if len(shapes) != 1:
        raise ValueError(
            f"All sweep trajectories must share shape (num_steps, H, D); got {shapes}."
        )

    stacked = np.stack(trajectories, axis=0)  # (n_w, n_steps, H, D)
    return SweepResult(
        trajectory=stacked,
        w_values=np.asarray(list(w_values), dtype=np.float32),
        prompt_pos=prompt_pos,
        prompt_neg=prompt_neg,
        task=task,
        ckpt=ckpt,
        gt_action=None if gt_action is None else np.asarray(gt_action, dtype=np.float32),
    )


def _load_pickle(path: Path) -> Mapping[str, Any]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(obj).__name__}.")
    return obj


def _build_websocket_policy(host: str, port: int):
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Need openpi_client installed; on the sim env run "
            "`bash viz_sim/install_pi0_client_in_sim_env.sh`."
        ) from e
    return WebsocketClientPolicy(host=host, port=port)


def _cli():
    ap = argparse.ArgumentParser(description="DeLock CPG sweep runner.")
    ap.add_argument("--obs-pickle", type=Path, required=True,
                    help="Pickle containing a single observation dict.")
    ap.add_argument("--prompt-pos", required=True,
                    help="Novel (positive / τ+) instruction.")
    ap.add_argument("--prompt-neg", required=True,
                    help="Trained (negative / τ-) instruction.")
    ap.add_argument("--w-values", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0],
                    help="Guidance scales to sweep (default: 0.0 0.5 1.0 1.5 2.0).")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--task", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", type=Path, required=True,
                    help="Path to write the npz (consumed by export_cpg_results.py).")
    args = ap.parse_args()

    obs = _load_pickle(args.obs_pickle)
    policy = _build_websocket_policy(args.host, args.port)
    print(f"Connected to ws://{args.host}:{args.port}")
    print(f"Sweep over w_values = {args.w_values}")
    res = sweep_cpg(
        policy_infer=policy.infer,
        base_obs=obs,
        w_values=args.w_values,
        prompt_pos=args.prompt_pos,
        prompt_neg=args.prompt_neg,
        task=args.task,
        ckpt=args.ckpt,
    )
    out_path = res.to_npz(args.out)
    print(f"Wrote {out_path}  trajectory shape={res.trajectory.shape}")


if __name__ == "__main__":  # pragma: no cover
    _cli()
