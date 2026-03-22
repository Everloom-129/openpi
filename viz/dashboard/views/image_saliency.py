"""Image Occlusion Saliency Tab — "Visual Jenga for VLA".

Adapts the core idea from the Visual Jenga paper (arxiv 2503.21770):
  - Paper: mask each scene object → measure inpainting diversity → rank by replaceability
  - Here:  mask each image patch → measure action-prediction change → rank by importance

Regions with HIGH action delta are "load-bearing" for the robot's decision.
Regions with LOW action delta are ignorable — the robot doesn't depend on them.

The diversity score analogue is:
    saliency[r, c] = ||policy.infer(image_with_patch_masked)["actions"]
                       - policy.infer(image_baseline)["actions"]||_2

Masking strategy: mean-fill (per-channel mean of the full image).
Less distribution shift than zero-fill; avoids the cost of real inpainting.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

_IMG_SIZE = 224
_CAMERA_DIR_MAP = {"right": "varied_camera_2", "left": "varied_camera_1"}

# Supported grid resolutions: (label, grid_size, n_inferences_per_camera)
_GRID_OPTIONS = {
    "4×4  (fast, 16 inferences / camera)":  4,
    "8×8  (balanced, 64 inferences / camera)": 8,
    "16×16 (full token resolution, 256 inferences / camera)": 16,
}


# ── Dataset catalogue (mirrors counterfactual.py) ────────────────────────────

def _catalogue() -> dict[str, dict]:
    example_dir = os.path.join(_PROJECT_ROOT, "data/example")
    cat: dict[str, dict] = {}
    if not os.path.isdir(example_dir):
        return cat
    for ep_name in sorted(os.listdir(example_dir)):
        ep_dir = os.path.join(example_dir, ep_name)
        if not os.path.isdir(ep_dir):
            continue
        if os.path.isdir(os.path.join(ep_dir, "recordings", "frames")):
            frames_root = os.path.join(ep_dir, "recordings", "frames")
        elif os.path.isdir(os.path.join(ep_dir, "frames")):
            frames_root = os.path.join(ep_dir, "frames")
        else:
            continue
        hand_dir = os.path.join(frames_root, "hand_camera")
        if not os.path.isdir(hand_dir):
            continue
        n = len([f for f in os.listdir(hand_dir) if f.endswith(".jpg")])
        if n == 0:
            continue
        instr_path = os.path.join(ep_dir, "instruction.txt")
        cat[ep_name] = {
            "frames_root": frames_root,
            "traj_h5": os.path.join(ep_dir, "trajectory.h5"),
            "default_prompt": open(instr_path).read().strip() if os.path.exists(instr_path) else "",
            "n_frames": n,
        }
    return cat


def _resolve_ext_dir(frames_root: str, camera: str) -> str:
    preferred = _CAMERA_DIR_MAP.get(camera, "varied_camera_2")
    if os.path.isdir(os.path.join(frames_root, preferred)):
        return preferred
    for name in sorted(os.listdir(frames_root)):
        if name != "hand_camera" and os.path.isdir(os.path.join(frames_root, name)):
            return name
    return preferred


def _load_frame(dataset: dict, frame_idx: int, camera: str = "right"):
    from PIL import Image
    import h5py
    frames_root = dataset["frames_root"]
    ext_dir = _resolve_ext_dir(frames_root, camera)
    ext_img = np.array(Image.open(
        os.path.join(frames_root, ext_dir, f"{frame_idx:05d}.jpg")
    ).convert("RGB"))
    wrist_img = np.array(Image.open(
        os.path.join(frames_root, "hand_camera", f"{frame_idx:05d}.jpg")
    ).convert("RGB"))
    with h5py.File(dataset["traj_h5"], "r") as f:
        joint_pos = f["observation/robot_state/joint_positions"][frame_idx].astype(np.float64)
        gripper_pos = f["observation/robot_state/gripper_position"][frame_idx : frame_idx + 1].astype(np.float64)
    return ext_img, wrist_img, joint_pos, gripper_pos


# ── Core saliency computation ─────────────────────────────────────────────────

def _resize_to_224(img: np.ndarray) -> np.ndarray:
    """Resize to 224×224 for consistent patch alignment with token grid."""
    return cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def _mean_fill_mask(img: np.ndarray, r: int, c: int, grid_size: int) -> np.ndarray:
    """Return a copy of img with patch (r,c) replaced by the per-channel mean."""
    h, w = img.shape[:2]
    ph, pw = h // grid_size, w // grid_size
    masked = img.copy()
    r0, r1 = r * ph, (r + 1) * ph
    c0, c1 = c * pw, (c + 1) * pw
    mean_px = img.mean(axis=(0, 1)).astype(img.dtype)
    masked[r0:r1, c0:c1] = mean_px
    return masked


def _run_infer(policy, example: dict) -> np.ndarray:
    """Run policy.infer() and return actions as float32 (8, 8).

    No attention capture — we only need the predicted action.
    """
    result = policy.infer(example)
    actions = result.get("actions")
    if actions is None:
        raise RuntimeError("policy.infer() returned no 'actions' key")
    return np.asarray(actions, dtype=np.float32)[:8]  # (8, 8)


def compute_occlusion_saliency(
    policy,
    ext_img: np.ndarray,
    wrist_img: np.ndarray,
    joint_pos: np.ndarray,
    gripper_pos: np.ndarray,
    prompt: str,
    grid_size: int = 8,
    progress_cb=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute occlusion saliency maps for both cameras.

    Returns:
        baseline_actions: float32 (8, 8)
        ext_saliency:   float32 (grid_size, grid_size) — exterior camera
        wrist_saliency: float32 (grid_size, grid_size) — wrist camera
    """
    # Resize to 224×224 to align with the model's 16×16 patch tokenisation
    ext_224 = _resize_to_224(ext_img)
    wrist_224 = _resize_to_224(wrist_img)

    base_example = {
        "observation/exterior_image_1_left": ext_224,
        "observation/wrist_image_left": wrist_224,
        "observation/joint_position": joint_pos,
        "observation/gripper_position": gripper_pos,
        "prompt": prompt,
    }

    baseline_actions = _run_infer(policy, base_example)

    total_steps = 2 * grid_size * grid_size
    step = 0

    def _tick(label: str):
        nonlocal step
        step += 1
        if progress_cb is not None:
            progress_cb(step / total_steps, label)

    # ── Exterior camera saliency ──────────────────────────────────────────────
    ext_saliency = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r in range(grid_size):
        for c in range(grid_size):
            masked_ext = _mean_fill_mask(ext_224, r, c, grid_size)
            example = {**base_example, "observation/exterior_image_1_left": masked_ext}
            actions = _run_infer(policy, example)
            ext_saliency[r, c] = float(np.linalg.norm(actions - baseline_actions))
            _tick(f"Exterior {r*grid_size+c+1}/{grid_size**2}")

    # ── Wrist camera saliency ─────────────────────────────────────────────────
    wrist_saliency = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r in range(grid_size):
        for c in range(grid_size):
            masked_wrist = _mean_fill_mask(wrist_224, r, c, grid_size)
            example = {**base_example, "observation/wrist_image_left": masked_wrist}
            actions = _run_infer(policy, example)
            wrist_saliency[r, c] = float(np.linalg.norm(actions - baseline_actions))
            _tick(f"Wrist {r*grid_size+c+1}/{grid_size**2}")

    return baseline_actions, ext_saliency, wrist_saliency


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _upsample(grid: np.ndarray) -> np.ndarray:
    """Upsample (M, M) float32 grid → (224, 224) via bilinear interpolation."""
    return cv2.resize(grid, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def _overlay(img: np.ndarray, hmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a float32 heatmap on an RGB image with Jet colormap."""
    img_s = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    hn = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    color = cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_s, 1 - alpha, color, alpha, 0)


def _jenga_overlay(img: np.ndarray, saliency: np.ndarray, top_k: int) -> np.ndarray:
    """Highlight the top-K most important patches with red borders."""
    img_s = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR).copy()
    grid_size = saliency.shape[0]
    ph = _IMG_SIZE // grid_size
    # Find top-K patches
    flat = saliency.flatten()
    threshold = np.sort(flat)[::-1][min(top_k - 1, len(flat) - 1)]
    for r in range(grid_size):
        for c in range(grid_size):
            if saliency[r, c] >= threshold:
                r0, r1 = r * ph, (r + 1) * ph
                c0, c1 = c * ph, (c + 1) * ph
                cv2.rectangle(img_s, (c0, r0), (c1 - 1, r1 - 1), (255, 50, 50), 2)
    return img_s


def _action_delta_bar(
    baseline_actions: np.ndarray,
    saliency: np.ndarray,
    top_k: int,
    camera_label: str,
) -> None:
    """Bar chart: for the top-K patches, show which action dimensions are most perturbed."""
    import matplotlib.pyplot as plt
    import io

    grid_size = saliency.shape[0]
    flat = saliency.flatten()
    top_indices = np.argsort(flat)[::-1][:top_k]
    # Convert flat index → (r, c)
    patch_rcs = [(i // grid_size, i % grid_size) for i in top_indices]

    # Placeholder: just show the saliency distribution across action steps.
    # For a richer view, we'd need to cache per-patch per-dim deltas.
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.bar(range(len(flat)), np.sort(flat)[::-1], color="steelblue")
    ax.set_xlabel("Patch rank")
    ax.set_ylabel("Action L2 delta")
    ax.set_title(f"{camera_label}: action sensitivity by patch rank")
    ax.axvline(x=top_k - 0.5, color="red", linestyle="--", alpha=0.6, label=f"Top-{top_k} cutoff")
    ax.legend(fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    st.image(buf.read(), use_container_width=True)


# ── GPU helper ────────────────────────────────────────────────────────────────

def _gpu_devices() -> list[str]:
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return [f"cuda:{i}" for i in range(n)] + ["cpu"]
    except Exception:
        return ["cuda:0", "cpu"]


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    """Render the Occlusion Saliency tab."""
    st.markdown("### Image Occlusion Saliency")
    st.caption(
        "Inspired by **Visual Jenga** (arxiv 2503.21770): mask each image patch, "
        "measure how much the robot's predicted action changes. "
        "**High saliency** = load-bearing region. **Low saliency** = the robot ignores it."
    )

    cat = _catalogue()
    if not cat:
        st.error("No local datasets found in `data/example/`.")
        return

    # ── Step 1: Image source ──────────────────────────────────────────────────
    with st.expander("① Image source & frame", expanded=True):
        dataset_name = st.selectbox("Dataset", list(cat.keys()), key="sal_dataset")
        dataset = cat[dataset_name]
        max_frame = dataset["n_frames"] - 1
        frame_idx = st.slider("Frame", 0, max_frame, 0, key="sal_frame")
        camera = st.radio("Ext camera", ["right", "left"], horizontal=True, key="sal_camera")
        try:
            ext_img, wrist_img, joint_pos, gripper_pos = _load_frame(dataset, frame_idx, camera)
            c1, c2 = st.columns(2)
            c1.image(ext_img, caption=f"Exterior ({camera})", use_container_width=True)
            c2.image(wrist_img, caption="Wrist", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load frame: {e}")
            return

    # ── Step 2: Config ────────────────────────────────────────────────────────
    with st.expander("② Settings", expanded=True):
        prompt = st.text_input(
            "Instruction prompt",
            value=dataset["default_prompt"],
            key="sal_prompt",
        )
        grid_label = st.radio(
            "Grid resolution",
            list(_GRID_OPTIONS.keys()),
            index=0,
            key="sal_grid",
        )
        grid_size = _GRID_OPTIONS[grid_label]

        gpu_device = st.selectbox("GPU", _gpu_devices(), key="sal_gpu")
        st.caption(
            f"Will run **{2 * grid_size**2 + 1}** inferences "
            f"({grid_size}×{grid_size} per camera + baseline)."
        )

    # ── Step 3: Run ───────────────────────────────────────────────────────────
    cache_key = f"sal_result_{dataset_name}_{frame_idx}_{grid_size}_{camera}_{prompt[:30]}"

    run_btn = st.button("▶ Run Occlusion Saliency", type="primary", key="sal_run")

    if run_btn:
        from viz.dashboard.inference import load_model  # noqa: PLC0415
        with st.spinner("Loading model…"):
            try:
                ckpt_path = os.path.join(_PROJECT_ROOT, "checkpoints/viz", "pi05_droid_pytorch")
                policy = load_model(ckpt_path, device=gpu_device)
            except Exception as e:
                st.error(f"Model load failed: {e}")
                return

        progress_bar = st.progress(0, text="Starting…")

        def _progress(frac: float, label: str):
            progress_bar.progress(min(frac, 1.0), text=label)

        try:
            baseline_actions, ext_sal, wrist_sal = compute_occlusion_saliency(
                policy=policy,
                ext_img=ext_img,
                wrist_img=wrist_img,
                joint_pos=joint_pos,
                gripper_pos=gripper_pos,
                prompt=prompt,
                grid_size=grid_size,
                progress_cb=_progress,
            )
            progress_bar.progress(1.0, text="Done.")
            st.session_state[cache_key] = {
                "baseline_actions": baseline_actions,
                "ext_saliency": ext_sal,
                "wrist_saliency": wrist_sal,
                "grid_size": grid_size,
                "ext_img": ext_img,
                "wrist_img": wrist_img,
            }
            st.success("Saliency computation complete.")
        except Exception as e:
            st.error(f"Saliency computation failed: {e}")
            return

    # ── Step 4: Visualise ─────────────────────────────────────────────────────
    result = st.session_state.get(cache_key)
    if result is None:
        st.info("Configure settings above and click **▶ Run Occlusion Saliency** to begin.")
        return

    ext_sal = result["ext_saliency"]
    wrist_sal = result["wrist_saliency"]
    baseline_actions = result["baseline_actions"]
    g = result["grid_size"]
    _ext = result["ext_img"]
    _wrist = result["wrist_img"]

    st.markdown("---")
    st.markdown("### Results")

    # ── Heatmap overlays ──────────────────────────────────────────────────────
    st.markdown("#### Saliency heatmaps (Jet: blue=low, red=high importance)")
    col_ext, col_wrist = st.columns(2)
    with col_ext:
        st.caption("**Exterior camera**")
        st.image(_overlay(_ext, _upsample(ext_sal)), caption="Exterior saliency", use_container_width=True)
    with col_wrist:
        st.caption("**Wrist camera**")
        st.image(_overlay(_wrist, _upsample(wrist_sal)), caption="Wrist saliency", use_container_width=True)

    # ── Visual Jenga view ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "#### Visual Jenga — load-bearing patches\n"
        "Patches outlined in red are the regions the robot depends on most. "
        "Remove them and the action prediction changes substantially."
    )
    max_patches = g * g
    top_k = st.slider(
        "Top-K most important patches to highlight",
        min_value=1,
        max_value=max_patches,
        value=max(1, max_patches // 8),
        key="sal_topk",
    )

    col_je, col_jw = st.columns(2)
    with col_je:
        st.image(
            _jenga_overlay(_ext, ext_sal, top_k),
            caption=f"Exterior — top {top_k} load-bearing patches",
            use_container_width=True,
        )
    with col_jw:
        st.image(
            _jenga_overlay(_wrist, wrist_sal, top_k),
            caption=f"Wrist — top {top_k} load-bearing patches",
            use_container_width=True,
        )

    # ── Action sensitivity ranking ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Action sensitivity ranking (sorted by importance)")
    _action_delta_bar(baseline_actions, ext_sal, top_k, "Exterior camera")
    _action_delta_bar(baseline_actions, wrist_sal, top_k, "Wrist camera")

    # ── Baseline action summary ───────────────────────────────────────────────
    with st.expander("Baseline predicted actions (8 steps × 8 dims)"):
        import pandas as pd
        dims = ["j0", "j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
        df = pd.DataFrame(baseline_actions, columns=dims)
        df.index.name = "step"
        st.dataframe(df.style.format("{:.4f}"), use_container_width=True)
