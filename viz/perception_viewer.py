"""Standalone perception viewer.

Browse raw DROID episodes, visualize Gemini-detected objects, SAM2 masks,
and the Gemini+SAM2 gripper segmentation mask.

Launch:
    uv run streamlit run viz/perception_viewer.py
    PERCEPTION_DATA_ROOT=/path/to/data uv run streamlit run viz/perception_viewer.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import streamlit as st
from PIL import Image

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src"), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from perception.gripper_mask import overlay_gripper_mask, segment_gripper

# ── Constants ──────────────────────────────────────────────────────────────────
_DEFAULT_DATA_ROOT = os.environ.get(
    "PERCEPTION_DATA_ROOT",
    "/mnt/sda/edward/projects/toy_cube_benchmark/cube_gold",
)
OPEN_LOOP_HORIZON = 8
_BBOX_COLORS = [
    (80,  200,  80),
    (80,  120, 255),
    (255, 200,  50),
    (200,  80, 255),
    (50,  220, 220),
    (255, 140,  20),
]
_SAM2_CHECKPOINT = str(_PROJECT_ROOT / "checkpoints/viz/sam2.1_hiera_large.pt")

st.set_page_config(
    page_title="Perception Viewer",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────────
for _k, _v in [("playing", False), ("frame_idx", 0), ("gripper_cache", {})]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Data helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _scan_episodes(data_root: str) -> list[tuple[str, str, str]]:
    root = Path(data_root)
    results = []
    for outcome in ("success", "failure"):
        od = root / outcome
        if not od.is_dir():
            continue
        for date_dir in sorted(od.iterdir()):
            if not date_dir.is_dir():
                continue
            for ep_dir in sorted(date_dir.iterdir()):
                if (ep_dir / "trajectory.h5").exists():
                    results.append((outcome, date_dir.name, ep_dir.name))
    return results


def _ep_dir(data_root: str, outcome: str, date: str, episode: str) -> Path:
    return Path(data_root) / outcome / date / episode


def _wrist_frame_path(ep: Path, idx: int) -> Path:
    return ep / "recordings" / "frames" / "hand_camera" / f"{idx:05d}.jpg"


def _load_wrist_frame(ep: Path, idx: int) -> np.ndarray | None:
    p = _wrist_frame_path(ep, idx)
    if not p.exists():
        return None
    return np.array(Image.open(p).convert("RGB"))


@st.cache_data(ttl=300)
def _load_gripper_positions(traj: str) -> np.ndarray:
    with h5py.File(traj, "r") as f:
        return f["observation/robot_state/gripper_position"][:]


@st.cache_data(ttl=300)
def _load_perception_h5(path: str) -> dict | None:
    if not Path(path).exists():
        return None
    with h5py.File(path, "r") as f:
        if "wrist" not in f:
            return None
        g = f["wrist"]
        image  = g["image"][:]
        labels = [s.decode() if isinstance(s, bytes) else s for s in g["bboxes/labels"][:]]
        box_2d = g["bboxes/box_2d"][:]
        masks  = g["masks"][:] if g["masks"].shape[0] > 0 else np.zeros((0, *image.shape[:2]), dtype=np.uint8)
    return {"image": image, "labels": labels, "box_2d": box_2d, "masks": masks}


def _n_frames(ep: Path) -> int:
    d = ep / "recordings" / "frames" / "hand_camera"
    return len(list(d.glob("*.jpg"))) if d.is_dir() else 0


def _keyframe_indices(n: int) -> list[int]:
    return list(range(0, n, OPEN_LOOP_HORIZON))


# ── Overlay helpers ────────────────────────────────────────────────────────────

def _draw_bboxes(img: np.ndarray, labels: list[str], box_2d: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    out = img.copy()
    for i, (label, box) in enumerate(zip(labels, box_2d)):
        c = _BBOX_COLORS[i % len(_BBOX_COLORS)]
        y0, x0, y1, x1 = (int(v / 1000 * d) for v, d in zip(box, [H, W, H, W]))
        cv2.rectangle(out, (x0, y0), (x1, y1), c, 2)
        cv2.putText(out, label, (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out


def _draw_masks(img: np.ndarray, masks: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = img.copy().astype(np.float32)
    for i, mask in enumerate(masks):
        c = np.array(_BBOX_COLORS[i % len(_BBOX_COLORS)], dtype=np.float32)
        m = mask.astype(bool)
        out[m] = out[m] * (1 - alpha) + c * alpha
    return out.clip(0, 255).astype(np.uint8)


def _build_annotated(
    raw: np.ndarray,
    perc: dict | None,
    gripper_masks: np.ndarray | None,
    gripper_bboxes: list[dict] | None,
    show_objects: bool,
    show_gripper: bool,
) -> np.ndarray:
    out = raw.copy()
    if show_objects and perc is not None:
        if perc["masks"].shape[0] > 0:
            out = _draw_masks(out, perc["masks"])
        if len(perc["labels"]) > 0:
            out = _draw_bboxes(out, perc["labels"], perc["box_2d"])
    if show_gripper and gripper_masks is not None and gripper_masks.shape[0] > 0:
        out = overlay_gripper_mask(out, gripper_masks, bboxes=gripper_bboxes)
    return out


# ── Video export ───────────────────────────────────────────────────────────────

def _export_video(
    ep: Path,
    frame_indices: list[int],
    data_root: str,
    outcome: str,
    date: str,
    episode: str,
    gripper_cache: dict,
    show_objects: bool,
    show_gripper: bool,
    fps: int = 10,
) -> bytes | None:
    """Render selected frames to an MP4 and return raw bytes."""
    if not frame_indices:
        return None

    first = _load_wrist_frame(ep, frame_indices[0])
    if first is None:
        return None
    H, W = first.shape[:2]

    traj = str(ep / "trajectory.h5")
    try:
        gpos = _load_gripper_positions(traj)
    except Exception:
        gpos = None

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (W, H))

    for idx in frame_indices:
        raw = _load_wrist_frame(ep, idx)
        if raw is None:
            continue

        perc_path = str(ep / "perception" / f"{idx:05d}" / "perception.h5")
        perc = _load_perception_h5(perc_path)

        cache_key = f"{ep}:{idx}"
        g_masks, g_bboxes = gripper_cache.get(cache_key, (None, None))

        annotated = _build_annotated(raw, perc, g_masks, g_bboxes, show_objects, show_gripper)
        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    writer.release()

    with open(tmp_path, "rb") as f:
        video_bytes = f.read()
    os.unlink(tmp_path)
    return video_bytes


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("👁 Perception Viewer")
    st.markdown("---")

    data_root = st.text_input("DATA_ROOT", value=_DEFAULT_DATA_ROOT)

    episodes = _scan_episodes(data_root)
    if not episodes:
        st.error(f"No episodes found under `{data_root}`.")
        st.stop()

    # Episode navigation
    outcomes = sorted({o for o, _, _ in episodes})
    sel_outcome = st.selectbox("Outcome", outcomes)
    dates = sorted({d for o, d, _ in episodes if o == sel_outcome})
    sel_date = st.selectbox("Date", dates)
    eps = [ep for o, d, ep in episodes if o == sel_outcome and d == sel_date]
    sel_episode = st.selectbox("Episode", eps)

    ep = _ep_dir(data_root, sel_outcome, sel_date, sel_episode)
    n_total = _n_frames(ep)
    if n_total == 0:
        st.error("No hand-camera frames found.")
        st.stop()

    keyframes = _keyframe_indices(n_total)
    keyframe_only = st.checkbox(f"Keyframes only (every {OPEN_LOOP_HORIZON})", value=False)
    available_frames = keyframes if keyframe_only else list(range(n_total))
    max_frame_idx = len(available_frames) - 1

    # Clamp current frame_idx to valid range for this episode
    if st.session_state["frame_idx"] > max_frame_idx:
        st.session_state["frame_idx"] = 0

    frame_pos = st.slider(
        "Frame", 0, max_frame_idx,
        key="frame_idx",
        help="Position in the (possibly filtered) frame list",
    )
    frame_idx = available_frames[frame_pos]

    traj_path = ep / "trajectory.h5"
    gpos_all = _load_gripper_positions(str(traj_path)) if traj_path.exists() else None
    gripper_pos = float(gpos_all[frame_idx]) if gpos_all is not None and frame_idx < len(gpos_all) else 0.0

    perc_h5 = str(ep / "perception" / f"{frame_idx:05d}" / "perception.h5")
    has_perc = Path(perc_h5).exists()

    st.metric("Gripper position (m)", f"{gripper_pos:.4f}")
    st.caption(f"Frame: **{frame_idx}** / {n_total - 1}")
    st.caption(f"Objects: {'✓' if has_perc else '○ not labeled'}")

    # ── Overlays ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Overlays**")
    show_objects = st.checkbox("Object bboxes / masks", value=True)
    show_gripper = st.checkbox("Gripper mask",          value=True)

    # ── Gripper detection ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Gripper detection**")

    sam2_device = st.selectbox("SAM2 device", ["cuda", "cpu"], index=0)
    gemini_model = st.selectbox(
        "Gemini model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-robotics-er-1.5-preview"],
        index=0,
    )

    cache_key_cur = f"{ep}:{frame_idx}"
    gripper_done = cache_key_cur in st.session_state["gripper_cache"]
    st.caption(f"Current frame: {'✓ cached' if gripper_done else '○ not run'}")

    run_current = st.button("🔍 Run gripper detection (this frame)", use_container_width=True)

    run_all = st.button(
        f"🔍 Run all {'keyframes' if keyframe_only else 'frames'} in episode",
        use_container_width=True,
    )

    # ── Playback ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Playback**")
    fps_val = st.slider("FPS", 1, 30, 8)

    play_col, stop_col = st.columns(2)
    if play_col.button("▶ Play", use_container_width=True, disabled=st.session_state["playing"]):
        st.session_state["playing"] = True
        st.rerun()
    if stop_col.button("⏸ Pause", use_container_width=True, disabled=not st.session_state["playing"]):
        st.session_state["playing"] = False
        st.rerun()

    export_btn = st.button("📹 Export video", use_container_width=True)


# ── Run gripper detection ──────────────────────────────────────────────────────

def _run_gripper_for_frame(ep: Path, idx: int, model_id: str, device: str) -> None:
    raw = _load_wrist_frame(ep, idx)
    if raw is None:
        return
    pil = Image.fromarray(raw)
    bboxes, masks = segment_gripper(
        pil,
        checkpoint=_SAM2_CHECKPOINT,
        device=device,
        model_id=model_id,
    )
    st.session_state["gripper_cache"][f"{ep}:{idx}"] = (masks, bboxes)


if run_current:
    with st.spinner(f"Running Gemini + SAM2 on frame {frame_idx}…"):
        _run_gripper_for_frame(ep, frame_idx, gemini_model, sam2_device)
    st.rerun()

if run_all:
    targets = available_frames
    progress = st.progress(0, text="Running gripper detection…")
    for i, idx in enumerate(targets):
        k = f"{ep}:{idx}"
        if k not in st.session_state["gripper_cache"]:
            _run_gripper_for_frame(ep, idx, gemini_model, sam2_device)
        progress.progress((i + 1) / len(targets), text=f"Frame {idx} ({i+1}/{len(targets)})")
    progress.empty()
    st.rerun()


# ── Main content ───────────────────────────────────────────────────────────────

st.subheader(f"{sel_outcome} / {sel_date} / {sel_episode}")

raw_image = _load_wrist_frame(ep, frame_idx)
if raw_image is None:
    st.error(f"Frame {frame_idx:05d} not found.")
    st.stop()

perc_data = _load_perception_h5(perc_h5)
g_masks, g_bboxes = st.session_state["gripper_cache"].get(cache_key_cur, (None, None))

annotated = _build_annotated(raw_image, perc_data, g_masks, g_bboxes, show_objects, show_gripper)

col_raw, col_ann = st.columns(2)
with col_raw:
    st.caption(f"Raw — frame **{frame_idx}**")
    st.image(raw_image, use_container_width=True)
with col_ann:
    st.caption("Annotated")
    st.image(annotated, use_container_width=True)

# ── Object detection table ─────────────────────────────────────────────────────
if perc_data and len(perc_data["labels"]) > 0:
    st.markdown("---")
    st.markdown("**Detected objects** (from perception.h5)")
    import pandas as pd
    st.dataframe(
        pd.DataFrame([
            {"label": lbl, "ymin": int(b[0]), "xmin": int(b[1]), "ymax": int(b[2]), "xmax": int(b[3])}
            for lbl, b in zip(perc_data["labels"], perc_data["box_2d"])
        ]),
        use_container_width=True, hide_index=True,
    )

if g_bboxes:
    st.markdown("**Gripper components**")
    import pandas as pd
    st.dataframe(
        pd.DataFrame([
            {"label": b["label"], "ymin": b["box_2d"][0], "xmin": b["box_2d"][1],
             "ymax": b["box_2d"][2], "xmax": b["box_2d"][3]}
            for b in g_bboxes
        ]),
        use_container_width=True, hide_index=True,
    )

# ── Gripper timeline ───────────────────────────────────────────────────────────
if gpos_all is not None:
    st.markdown("---")
    st.markdown("**Gripper position over episode**")
    import pandas as pd
    df = pd.DataFrame({"frame": range(len(gpos_all)), "gripper_position_m": gpos_all})
    st.line_chart(df.set_index("frame"), use_container_width=True, height=140)

# ── Video export ───────────────────────────────────────────────────────────────
if export_btn:
    with st.spinner(f"Rendering {len(available_frames)} frames to MP4…"):
        video_bytes = _export_video(
            ep=ep,
            frame_indices=available_frames,
            data_root=data_root,
            outcome=sel_outcome,
            date=sel_date,
            episode=sel_episode,
            gripper_cache=st.session_state["gripper_cache"],
            show_objects=show_objects,
            show_gripper=show_gripper,
            fps=fps_val,
        )
    if video_bytes:
        st.success(f"Done — {len(video_bytes)//1024} KB")
        st.download_button(
            "⬇ Download MP4",
            data=video_bytes,
            file_name=f"{sel_episode}_perception.mp4",
            mime="video/mp4",
        )
    else:
        st.error("Export failed.")

# ── Auto-playback ──────────────────────────────────────────────────────────────
if st.session_state["playing"]:
    time.sleep(1.0 / fps_val)
    nxt = st.session_state["frame_idx"] + 1
    if nxt > max_frame_idx:
        st.session_state["playing"] = False
    else:
        st.session_state["frame_idx"] = nxt
    st.rerun()
