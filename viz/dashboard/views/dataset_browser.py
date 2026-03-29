"""Dataset browser for Online (Dataset) mode.

Scans a DATA_ROOT directory (toy-cube-benchmark / DROID format), renders an
interactive episode grid and per-episode keyframe strip, and triggers online
inference when the user clicks a keyframe thumbnail.

Expected DATA_ROOT layout:
    DATA_ROOT/{success,failure}/{date}/{episode}/
        instruction.txt
        trajectory.h5
        recordings/frames/varied_camera_2/{frame:05d}.jpg
        recordings/frames/hand_camera/{frame:05d}.jpg
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

OPEN_LOOP_HORIZON = 8
THUMB_W,  THUMB_H  = 160, 120   # episode card thumbnail
KF_W,     KF_H     = 112,  84   # keyframe strip thumbnail
EP_COLS   = 5                    # episode cards per row
KF_COLS   = 8                    # keyframe thumbnails per row


# ── Cached data helpers ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _scan_episodes(data_root: str) -> list[dict]:
    """Scan DATA_ROOT for all episodes. Result is cached per path."""
    root = Path(data_root)
    episodes: list[dict] = []
    for outcome in ("success", "failure"):
        outcome_dir = root / outcome
        if not outcome_dir.exists():
            continue
        for date_dir in sorted(outcome_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            for traj_path in sorted(date_dir.rglob("trajectory.h5")):
                data_dir = traj_path.parent
                hand_dir = data_dir / "recordings" / "frames" / "hand_camera"
                n_frames = len(list(hand_dir.glob("*.jpg"))) if hand_dir.exists() else 0
                keyframes = list(range(0, n_frames, OPEN_LOOP_HORIZON))
                instr_path = data_dir / "instruction.txt"
                instruction = instr_path.read_text().strip() if instr_path.exists() else ""
                thumb_path = (
                    data_dir / "recordings" / "frames" / "varied_camera_2" / "00000.jpg"
                )
                episodes.append({
                    "outcome":    outcome,
                    "date":       date_dir.name,
                    "episode_id": data_dir.name,
                    "data_dir":   str(data_dir),
                    "instruction": instruction,
                    "n_frames":   n_frames,
                    "n_keyframes": len(keyframes),
                    "keyframes":  keyframes,
                    "thumb_path": str(thumb_path),
                })
    return episodes


@st.cache_data(show_spinner=False)
def _load_thumb(path: str, w: int, h: int) -> np.ndarray | None:
    """Load and resize one thumbnail. Cached per (path, w, h)."""
    from PIL import Image as _PIL
    try:
        img = _PIL.open(path).convert("RGB").resize((w, h), _PIL.BILINEAR)
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None


# ── Inference helper ───────────────────────────────────────────────────────────

def _run_inference(
    ep: dict,
    frame_idx: int,
    checkpoint_path: str,
    gpu_device: str,
) -> None:
    """Load example from trajectory.h5, run inference, store in session state."""
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
    for p in [project_root, os.path.join(project_root, "src"),
              os.path.join(project_root, "viz")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from pipeline import load_example as _load_example
    from viz.dashboard import inference as _inf

    if gpu_device == "Auto":
        try:
            from attn_map import select_best_gpu as _sbg
            _dev = _sbg()
            gpu = f"cuda:{_dev}" if isinstance(_dev, int) else str(_dev)
        except Exception:
            gpu = "cuda:0"
    else:
        gpu = gpu_device

    example = _load_example(
        data_dir=Path(ep["data_dir"]), index=frame_idx, camera="right"
    )
    example["prompt"] = ep["instruction"]

    policy = _inf.load_model(checkpoint_path, device=gpu)
    result = _inf.run_inference(policy, example)

    st.session_state["online_data"]      = result
    st.session_state["ds_selected_frame"] = frame_idx


# ── UI rendering ───────────────────────────────────────────────────────────────

def render(data_root: str, checkpoint_path: str, gpu_device: str) -> None:
    """Main entry point: episode grid → keyframe strip → inference."""
    if not os.path.isdir(data_root):
        st.warning(f"DATA_ROOT not found: `{data_root}`")
        return

    with st.spinner("Scanning episodes…"):
        episodes = _scan_episodes(data_root)

    if not episodes:
        st.warning(f"No episodes found in `{data_root}`")
        return

    selected_ep    = st.session_state.get("ds_selected_episode")
    selected_frame = st.session_state.get("ds_selected_frame")

    # ── Episode grid ─────────────────────────────────────────────────────────
    n_success = sum(1 for e in episodes if e["outcome"] == "success")
    n_failure = len(episodes) - n_success
    st.subheader(f"Episodes — {len(episodes)} total  ({n_success} ✅ success · {n_failure} ❌ failure)")

    for row_start in range(0, len(episodes), EP_COLS):
        row_eps = episodes[row_start : row_start + EP_COLS]
        cols    = st.columns(EP_COLS)
        for j, ep in enumerate(row_eps):
            with cols[j]:
                thumb = _load_thumb(ep["thumb_path"], THUMB_W, THUMB_H)
                if thumb is not None:
                    st.image(thumb, use_container_width=True)
                else:
                    st.markdown("🖼 *(no image)*")

                badge = "✅" if ep["outcome"] == "success" else "❌"
                instr = ep["instruction"]
                label = (instr[:48] + "…") if len(instr) > 48 else (instr or "*(no instruction)*")
                st.caption(f"{badge} {label}")
                st.caption(f"🎞 {ep['n_keyframes']} keyframes")

                is_sel = (
                    selected_ep is not None
                    and selected_ep["data_dir"] == ep["data_dir"]
                )
                if st.button(
                    "▶ Selected" if is_sel else "Select",
                    key=f"ds_ep_{ep['outcome']}_{ep['date']}_{ep['episode_id']}",
                    use_container_width=True,
                    type="primary" if is_sel else "secondary",
                ):
                    if not is_sel:
                        st.session_state["ds_selected_episode"] = ep
                        st.session_state.pop("ds_selected_frame", None)
                        st.session_state.pop("online_data", None)
                    st.rerun()

    if selected_ep is None:
        st.info("Click an episode above to browse its keyframes.")
        return

    # ── Keyframe strip ────────────────────────────────────────────────────────
    st.divider()
    instr_display = selected_ep["instruction"] or "*(no instruction)*"
    st.subheader(f"Keyframes — `{selected_ep['episode_id']}`")
    st.caption(
        f"📋 **{instr_display}** · "
        f"{selected_ep['n_keyframes']} keyframes · "
        f"{selected_ep['n_frames']} total frames"
    )

    keyframes = selected_ep["keyframes"]
    for row_start in range(0, len(keyframes), KF_COLS):
        row_kf = keyframes[row_start : row_start + KF_COLS]
        cols   = st.columns(KF_COLS)
        for j, frame_idx in enumerate(row_kf):
            with cols[j]:
                kf_path = str(
                    Path(selected_ep["data_dir"]) / "recordings" / "frames"
                    / "varied_camera_2" / f"{frame_idx:05d}.jpg"
                )
                kf_thumb = _load_thumb(kf_path, KF_W, KF_H)
                if kf_thumb is not None:
                    st.image(kf_thumb, use_container_width=True)

                is_sel = selected_frame == frame_idx
                st.caption(f"{'▶ ' if is_sel else ''}f{frame_idx:05d}")

                if st.button(
                    "✓ Done" if is_sel else "▶ Run",
                    key=f"ds_kf_{row_start + j}",
                    use_container_width=True,
                    type="primary" if is_sel else "secondary",
                ):
                    with st.spinner(f"Running inference on frame {frame_idx}…"):
                        try:
                            _run_inference(
                                selected_ep, frame_idx, checkpoint_path, gpu_device
                            )
                        except Exception as e:
                            st.error(f"Inference failed: {e}")
                            import traceback; traceback.print_exc()
                    st.rerun()
