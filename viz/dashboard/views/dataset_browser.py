"""Dataset browser for Online (Dataset) mode.

Three-page navigation driven by st.session_state["ds_page"]:

    "grid"      → episode card grid (5/row)
    "keyframes" → keyframe strip for selected episode (8/row)
    "results"   → back-navigation bar; app.py renders the attention tabs below

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
THUMB_W, THUMB_H = 160, 120   # episode card thumbnail
KF_W,    KF_H    = 120,  90   # keyframe strip thumbnail
EP_COLS  = 5                   # episode cards per row
KF_COLS  = 8                   # keyframe thumbnails per row


# ── Cached helpers ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _scan_episodes(data_root: str) -> list[dict]:
    """Scan DATA_ROOT for all episodes. Cached per path."""
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
                    "outcome":     outcome,
                    "date":        date_dir.name,
                    "episode_id":  data_dir.name,
                    "data_dir":    str(data_dir),
                    "instruction": instruction,
                    "n_frames":    n_frames,
                    "n_keyframes": len(keyframes),
                    "keyframes":   keyframes,
                    "thumb_path":  str(thumb_path),
                })
    return episodes


@st.cache_data(show_spinner=False)
def _load_thumb(path: str, w: int, h: int) -> np.ndarray | None:
    """Load and resize a thumbnail. Cached per (path, w, h)."""
    from PIL import Image as _PIL
    try:
        img = _PIL.open(path).convert("RGB").resize((w, h), _PIL.BILINEAR)
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None


# ── Navigation helpers ─────────────────────────────────────────────────────────

def _go(page: str, **extra) -> None:
    """Set ds_page and any extra session state keys, then rerun."""
    st.session_state["ds_page"] = page
    for k, v in extra.items():
        st.session_state[k] = v
    st.rerun()


# ── Inference ──────────────────────────────────────────────────────────────────

def _run_inference(ep: dict, frame_idx: int, checkpoint_path: str, gpu_device: str) -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
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

    example = _load_example(data_dir=Path(ep["data_dir"]), index=frame_idx, camera="right")
    example["prompt"] = ep["instruction"]
    policy = _inf.load_model(checkpoint_path, device=gpu)
    result = _inf.run_inference(policy, example)
    st.session_state["online_data"]       = result
    st.session_state["ds_selected_frame"] = frame_idx


# ── Page renderers ─────────────────────────────────────────────────────────────

def _page_grid(data_root: str) -> None:
    """Page 1: episode card grid."""
    with st.spinner("Scanning episodes…"):
        episodes = _scan_episodes(data_root)

    if not episodes:
        st.warning(f"No episodes found in `{data_root}`")
        return

    n_success = sum(1 for e in episodes if e["outcome"] == "success")
    n_failure = len(episodes) - n_success
    st.subheader(
        f"Episodes — {len(episodes)} total  "
        f"({n_success} ✅ success · {n_failure} ❌ failure)"
    )

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
                st.markdown(f"**{badge} {label}**")
                st.markdown(f"🎞 **{ep['n_keyframes']}** keyframes")

                if st.button(
                    "Select →",
                    key=f"ds_ep_{ep['outcome']}_{ep['date']}_{ep['episode_id']}",
                    use_container_width=True,
                ):
                    st.session_state.pop("ds_selected_frame", None)
                    st.session_state.pop("online_data", None)
                    _go("keyframes", ds_selected_episode=ep)


def _page_keyframes(checkpoint_path: str, gpu_device: str) -> None:
    """Page 2: keyframe strip for the selected episode."""
    ep = st.session_state.get("ds_selected_episode")
    if ep is None:
        _go("grid")
        return

    # ── Back button ──────────────────────────────────────────────────────────
    if st.button("← Back to episodes", key="ds_back_to_grid"):
        _go("grid")

    st.subheader(f"Keyframes — `{ep['episode_id']}`")
    instr_display = ep["instruction"] or "*(no instruction)*"
    st.markdown(
        f"📋 **{instr_display}** &nbsp;·&nbsp; "
        f"**{ep['n_keyframes']}** keyframes &nbsp;·&nbsp; "
        f"{ep['n_frames']} total frames"
    )

    keyframes = ep["keyframes"]
    for row_start in range(0, len(keyframes), KF_COLS):
        row_kf = keyframes[row_start : row_start + KF_COLS]
        cols   = st.columns(KF_COLS)
        for j, frame_idx in enumerate(row_kf):
            with cols[j]:
                kf_path = str(
                    Path(ep["data_dir"]) / "recordings" / "frames"
                    / "varied_camera_2" / f"{frame_idx:05d}.jpg"
                )
                kf_thumb = _load_thumb(kf_path, KF_W, KF_H)
                if kf_thumb is not None:
                    st.image(kf_thumb, use_container_width=True)
                st.markdown(f"f{frame_idx:05d}")

                if st.button(
                    "▶ Run",
                    key=f"ds_kf_{row_start + j}",
                    use_container_width=True,
                ):
                    with st.spinner(f"Running inference on frame {frame_idx}…"):
                        try:
                            _run_inference(ep, frame_idx, checkpoint_path, gpu_device)
                        except Exception as e:
                            st.error(f"Inference failed: {e}")
                            import traceback; traceback.print_exc()
                            return
                    _go("results")


def _page_results(ep: dict, frame_idx: int) -> None:
    """Page 3: navigation bar shown above the attention tabs (rendered by app.py)."""
    col_back, col_info = st.columns([2, 8])
    with col_back:
        if st.button("← Back to keyframes", key="ds_back_to_kf"):
            st.session_state.pop("online_data", None)
            _go("keyframes")
    with col_info:
        badge = "✅" if ep.get("outcome") == "success" else "❌"
        st.markdown(
            f"{badge} **{ep.get('episode_id', '')}** &nbsp;·&nbsp; "
            f"frame **{frame_idx:05d}** &nbsp;·&nbsp; "
            f"_{ep.get('instruction', '')}_"
        )
    st.divider()


# ── Public entry point ─────────────────────────────────────────────────────────

def render(data_root: str, checkpoint_path: str, gpu_device: str) -> None:
    """Dispatch to the correct page based on ds_page session state."""
    if not os.path.isdir(data_root):
        st.warning(f"DATA_ROOT not found: `{data_root}`")
        return

    page = st.session_state.get("ds_page", "grid")

    if page == "grid":
        _page_grid(data_root)

    elif page == "keyframes":
        _page_keyframes(checkpoint_path, gpu_device)

    elif page == "results":
        ep        = st.session_state.get("ds_selected_episode", {})
        frame_idx = st.session_state.get("ds_selected_frame", 0)
        _page_results(ep, frame_idx)
        # app.py renders attention tabs below this point
