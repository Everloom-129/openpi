"""Batch inference pipeline — saves attention directly to HDF5 (no temp files).

For every episode under DATA_ROOT, runs Pi0.5 inference on every keyframe
(downsampled by OPEN_LOOP_HORIZON) and writes one HDF5 file per inference:

  * main inference:          {frame:05d}/{frame:05d}.h5
  * counterfactual prompts:  {frame:05d}/{frame:05d}_{cf_key}.h5

Output layout:

    RESULTS_ROOT/
    └── {success,failure}/
        └── {date}/
            └── {episode}/
                ├── pi05.md                  ← completion marker
                ├── 00000/
                │   ├── 00000.h5             ← main inference
                │   ├── 00000_cube.h5        ← counterfactual
                │   └── ...
                ├── 00008/
                │   └── ...
                └── ...

Counterfactual prompts are loaded from viz/config/counterfactual.yaml.
Edit that file to add / remove / change prompt variants without touching code.

Usage:
    uv run python viz/pipeline.py <DATA_ROOT> <RESULTS_ROOT>
    uv run python viz/pipeline.py <DATA_ROOT> <RESULTS_ROOT> --no-counterfactual
    uv run python viz/pipeline.py <DATA_ROOT> <RESULTS_ROOT> --force   # reprocess all
    uv run python viz/pipeline.py <DATA_ROOT> <RESULTS_ROOT> \
        --checkpoint ./checkpoints/my_ckpt \
        --cf-config viz/config/my_prompts.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from attn_h5_writer import write_attn_h5_from_buffer
from attn_map import get_keyframes, get_policy, select_best_gpu

# ── Configuration ──────────────────────────────────────────────────────────────
OPEN_LOOP_HORIZON: int = 8
CAMERA: str = "right"   # "right" = varied_camera_2, "left" = varied_camera_1
DEFAULT_CHECKPOINT: str = "./checkpoints/viz/pi05_droid_pytorch"
DEFAULT_CF_CONFIG: str = str(Path(__file__).parent / "config" / "counterfactual.yaml")


# ── Config loader ──────────────────────────────────────────────────────────────

def load_cf_config(config_path: str | Path) -> list[dict]:
    """Load counterfactual prompt list from YAML.

    Returns a list of dicts with keys: key, prompt, method.
    Returns [] if the file is missing or 'prompts' is absent.
    """
    import yaml
    p = Path(config_path)
    if not p.exists():
        print(f"[warn] CF config not found: {p} — skipping counterfactuals")
        return []
    with open(p) as f:
        data = yaml.safe_load(f)
    return data.get("prompts", [])


# ── Data helpers ───────────────────────────────────────────────────────────────

def get_video_length(data_dir: Path) -> int:
    hand_dir = data_dir / "recordings" / "frames" / "hand_camera"
    return len(list(hand_dir.glob("*.jpg"))) if hand_dir.exists() else 0


def load_example(data_dir: Path, index: int, camera: str = "right") -> dict:
    """Load one frame from a DROID-format episode directory."""
    side_camera = "varied_camera_2" if camera == "right" else "varied_camera_1"
    frames_dir = data_dir / "recordings" / "frames"

    ext_path = frames_dir / side_camera / f"{index:05d}.jpg"
    hand_path = frames_dir / "hand_camera" / f"{index:05d}.jpg"

    if not ext_path.exists():
        raise FileNotFoundError(f"Exterior image not found: {ext_path}")
    if not hand_path.exists():
        raise FileNotFoundError(f"Hand image not found: {hand_path}")

    ext_img = np.array(Image.open(ext_path).convert("RGB"))
    hand_img = np.array(Image.open(hand_path).convert("RGB"))

    instruction_path = data_dir / "instruction.txt"
    instruction = instruction_path.read_text().strip() if instruction_path.exists() else ""

    traj_path = data_dir / "trajectory.h5"
    with h5py.File(traj_path, "r") as f:
        joint_position = (
            f["observation/robot_state/joint_positions"][index].astype(np.float64)
        )
        gripper_position = (
            f["observation/robot_state/gripper_position"][index : index + 1].astype(np.float64)
        )
        cartesian_position = None
        if "observation/robot_state/cartesian_position" in f:
            cartesian_position = (
                f["observation/robot_state/cartesian_position"][index].astype(np.float64)
            )

        # GT actions for the next OPEN_LOOP_HORIZON steps: (8, action_dim) float32
        gt_action: np.ndarray | None = None
        if "action/joint_velocity" in f:
            traj_len = f["action/joint_velocity"].shape[0]
            end = min(index + OPEN_LOOP_HORIZON, traj_len)
            n = end - index
            jv = f["action/joint_velocity"][index:end].astype(np.float32)   # (n, 7)
            gp = f["action/gripper_position"][index:end].astype(np.float32) # (n,) or (n,1)
            if gp.ndim == 1:
                gp = gp[:, None]
            gt_action = np.concatenate([jv, gp], axis=1)                    # (n, 8)
            if n < OPEN_LOOP_HORIZON:
                pad = np.full((OPEN_LOOP_HORIZON - n, gt_action.shape[1]), np.nan, dtype=np.float32)
                gt_action = np.concatenate([gt_action, pad], axis=0)        # (8, 8)

    return {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": hand_img,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "observation/cartesian_position": cartesian_position,
        "prompt": instruction,
        "gt_action": gt_action,
    }


# ── Robocasa adapter ──────────────────────────────────────────────────────────
# Auto-applied when the loaded policy uses RobocasaInputs (e.g. pi05_robocasa365).
# Toy_cube / DROID episodes are fixed-base Franka with axis-angle eef, so we
# synthesize: eef_pos = cartesian[:3], eef_quat = quat(axisangle(cartesian[3:6])),
# base_pos = 0, base_quat = identity (xyzw), gripper_qpos = duplicated scalar.
# Image keys are renamed to the robocasa schema (observation/image,
# observation/wrist_image). The result is OOD vs the model's training
# distribution (PandaOmron kitchen scenes) — it runs, it is not "correct".

def _policy_uses_robocasa(policy) -> bool:
    from openpi.policies.robocasa_policy import RobocasaInputs
    transform = getattr(policy, "_input_transform", None)
    stack = [transform]
    while stack:
        t = stack.pop()
        if t is None:
            continue
        if isinstance(t, RobocasaInputs):
            return True
        sub = getattr(t, "transforms", None)
        if sub is not None:
            stack.extend(sub)
    return False


def _droid_to_robocasa_example(example: dict) -> dict:
    """Repack a DROID-format example for RobocasaInputs (16-D state, renamed images)."""
    from scipy.spatial.transform import Rotation

    cart = example.get("observation/cartesian_position")
    if cart is None:
        raise KeyError(
            "observation/cartesian_position missing — robocasa adapter needs DROID "
            "trajectory.h5 with observation/robot_state/cartesian_position"
        )
    eef_pos = np.asarray(cart[:3], dtype=np.float32)
    eef_quat = Rotation.from_rotvec(np.asarray(cart[3:6])).as_quat().astype(np.float32)  # xyzw
    base_pos = np.zeros(3, dtype=np.float32)
    base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # identity xyzw
    g = float(np.asarray(example["observation/gripper_position"]).reshape(-1)[0])
    gripper_qpos = np.array([g, g], dtype=np.float32)

    state16 = np.concatenate([eef_pos, eef_quat, base_pos, base_quat, gripper_qpos])
    return {
        "observation/state": state16,
        "observation/image": example["observation/exterior_image_1_left"],
        "observation/wrist_image": example["observation/wrist_image_left"],
        "prompt": example["prompt"],
        # Carried through for the H5 writer (not consumed by RobocasaInputs):
        "gt_action": example.get("gt_action"),
    }


# ── Core inference + H5 write ──────────────────────────────────────────────────

def _is_pi05(policy) -> bool:
    """Return True if the loaded policy uses the π₀.₅ architecture (state in text tokens)."""
    return bool(getattr(getattr(policy, "_model", None), "pi05", True))


def infer_and_save(
    policy,
    example: dict,
    h5_path: Path,
    frame_idx: int,
    infer_example: dict | None = None,
) -> dict:
    """Run one inference step and write attention directly to HDF5 from RAM.

    Protocol:
        1. enable_attn_buffer()   — arms the in-RAM capture in gemma_pytorch
        2. policy.infer()         — model forward; attention accumulates in buffer
        3. get_attn_buffer()      — grab the dict before clearing
        4. clear_attn_buffer()    — always in finally, even if infer() raises
        5. write_attn_h5_from_buffer()  — compress + write HDF5
    """
    from openpi.models_pytorch import gemma_pytorch as _gpt

    _gpt.enable_attn_buffer()
    _gpt.enable_suffix_attn_buffer(capture_steps=1)   # averaged first-step (existing /suffix)
    _gpt.enable_suffix_attn_steps_buffer()             # all steps → /suffix_denoising attn
    _gpt.enable_action_traj_buffer()                   # x_t after each Euler step
    try:
        result          = policy.infer(infer_example if infer_example is not None else example)
        buf             = _gpt.get_attn_buffer()
        suffix_buf      = _gpt.get_suffix_attn_buffer()
        suffix_steps    = _gpt.get_suffix_attn_steps_buffer()
        action_traj     = _gpt.get_action_traj_buffer()
    finally:
        _gpt.clear_attn_buffer()
        _gpt.clear_suffix_attn_buffer()
        _gpt.clear_suffix_attn_steps_buffer()
        _gpt.clear_action_traj_buffer()

    write_attn_h5_from_buffer(
        attn_buffer=buf or {},
        h5_path=h5_path,
        ext_img=example["observation/exterior_image_1_left"],
        wrist_img=example["observation/wrist_image_left"],
        instruction=example["prompt"],
        frame_idx=frame_idx,
        suffix_attn_buffer=suffix_buf or {},
        suffix_steps_buffer=suffix_steps or [],
        gt_action=example.get("gt_action"),
        pred_action=result.get("actions"),
        action_traj=action_traj or [],
        is_pi05=_is_pi05(policy),
    )
    return result


# ── Episode processing ─────────────────────────────────────────────────────────

def process_episode(
    policy,
    data_dir: Path,
    episode_dir: Path,
    cf_prompts: list[dict],
    uses_robocasa: bool = False,
) -> dict:
    """Process all keyframes of one episode.

    Returns stats dict: {total, ok, skipped, errors}.
    """
    total_frames = get_video_length(data_dir)
    if total_frames == 0:
        return {"total": 0, "ok": 0, "skipped": 0, "errors": 0}

    keyframes = get_keyframes(total_frames, OPEN_LOOP_HORIZON)
    stats = {"total": len(keyframes), "ok": 0, "skipped": 0, "errors": 0}

    for frame_idx in keyframes:
        frame_dir = episode_dir / f"{frame_idx:05d}"
        frame_dir.mkdir(exist_ok=True)

        try:
            example = load_example(data_dir, frame_idx, camera=CAMERA)
            infer_example = _droid_to_robocasa_example(example) if uses_robocasa else None

            # ── Main inference ─────────────────────────────────────────────
            h5_main = frame_dir / f"{frame_idx:05d}.h5"
            if h5_main.exists():
                print(f"    {frame_idx:05d}.h5  (skip)")
                stats["skipped"] += 1
            else:
                infer_and_save(policy, example, h5_main, frame_idx, infer_example=infer_example)
                print(f"    {frame_idx:05d}.h5  ✓")

            # ── Counterfactual prompts ─────────────────────────────────────
            for cf in cf_prompts:
                h5_cf = frame_dir / f"{frame_idx:05d}_{cf['key']}.h5"
                if h5_cf.exists():
                    continue
                cf_example = {**example, "prompt": cf["prompt"]}
                cf_infer = (
                    {**infer_example, "prompt": cf["prompt"]} if infer_example is not None else None
                )
                infer_and_save(policy, cf_example, h5_cf, frame_idx, infer_example=cf_infer)
                print(f"    {frame_idx:05d}_{cf['key']}.h5  ✓  [{cf['method']}]")

            stats["ok"] += 1

        except FileNotFoundError as e:
            print(f"    frame {frame_idx:05d}: file not found — {e}")
            stats["errors"] += 1
        except Exception as e:
            import traceback
            print(f"    frame {frame_idx:05d}: error — {e}")
            traceback.print_exc()
            stats["errors"] += 1

    return stats


# ── Entry point ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pi0/Pi0.5 batch attention pipeline")
    parser.add_argument("data_root",    help="Root dir with success/ and failure/ subdirs")
    parser.add_argument("results_root", help="Output root directory")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", default=None,
                        help="Training config name, e.g. 'pi0_droid' or 'pi05_droid'. "
                             "If omitted, inferred from checkpoint directory name.")
    parser.add_argument("--cf-config",  default=DEFAULT_CF_CONFIG,
                        help="Path to counterfactual YAML config")
    parser.add_argument("--no-counterfactual", dest="counterfactual",
                        action="store_false",
                        help="Skip counterfactual prompt inference")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess episodes even if pi05.md marker exists")
    args = parser.parse_args(argv)

    DATA_ROOT    = Path(args.data_root)
    RESULTS_ROOT = Path(args.results_root)

    cf_prompts = load_cf_config(args.cf_config) if args.counterfactual else []
    if cf_prompts:
        print(f"Loaded {len(cf_prompts)} counterfactual prompts from {args.cf_config}")
        for cf in cf_prompts:
            print(f"  [{cf['method']}] {cf['key']!r}: {cf['prompt']!r}")
        print()

    device = select_best_gpu()
    print(f"Loading policy from {args.checkpoint} on {device} ...")
    policy = get_policy(args.checkpoint, device=device, config_name=args.model)
    uses_robocasa = _policy_uses_robocasa(policy)
    if uses_robocasa:
        print("[adapter] Detected RobocasaInputs — converting DROID examples to robocasa schema "
              "(synthesized 16-D state, renamed image keys). Output is OOD vs training.")
    print("Policy loaded.\n")

    total_episodes = processed = skipped = errors = 0

    for outcome in ("success", "failure"):
        outcome_dir = DATA_ROOT / outcome
        if not outcome_dir.exists():
            continue

        for date_dir in sorted(outcome_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            for traj_path in sorted(date_dir.rglob("trajectory.h5")):
                data_dir  = traj_path.parent
                episode_id = data_dir.name
                total_episodes += 1

                rel_path    = data_dir.relative_to(DATA_ROOT)
                episode_dir = RESULTS_ROOT / rel_path
                episode_dir.mkdir(parents=True, exist_ok=True)

                marker = episode_dir / "pi05.md"
                if marker.exists() and not args.force:
                    print(f"[skip] {outcome}/{date_dir.name}/{episode_id}")
                    skipped += 1
                    continue

                print(f"\n[{total_episodes}] {outcome}/{date_dir.name}/{episode_id}")
                t0 = time.perf_counter()

                stats = process_episode(
                    policy=policy,
                    data_dir=data_dir,
                    episode_dir=episode_dir,
                    cf_prompts=cf_prompts,
                    uses_robocasa=uses_robocasa,
                )

                elapsed = time.perf_counter() - t0
                print(
                    f"  done {elapsed:.0f}s  "
                    f"ok={stats['ok']} skip={stats['skipped']} err={stats['errors']}"
                )

                if stats["errors"] > 0 and stats["ok"] == 0:
                    print("  all frames failed, skipping marker")
                    errors += 1
                else:
                    total_frames = get_video_length(data_dir)
                    marker.write_text(
                        f"# Processing Complete\n\n"
                        f"Episode: {episode_id}\n"
                        f"Outcome: {outcome}\n"
                        f"Date: {date_dir.name}\n"
                        f"Total Frames: {total_frames}\n"
                        f"Keyframes: {get_keyframes(total_frames, OPEN_LOOP_HORIZON)}\n"
                        f"Counterfactuals: {[c['key'] for c in cf_prompts]}\n"
                    )
                    processed += 1

    print(f"\n{'=' * 50}")
    print(f"Total episodes:         {total_episodes}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped (done):         {skipped}")
    print(f"Errors:                 {errors}")
    print(f"Results:                {RESULTS_ROOT}")


if __name__ == "__main__":
    t_start = time.perf_counter()
    main()
    print(f"\nTotal wall time: {time.perf_counter() - t_start:.1f}s")
