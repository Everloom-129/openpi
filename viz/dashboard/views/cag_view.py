"""Counterfactual Action Guidance (CAG) Tab.

Implements the training-free diagnostic from:
  "Vision Overrides Language in VLAs" (arxiv 2602.17659v1)

Core formula:
    π_CAG(a|o,l) = π_uncond(a|o,∅) + ω · (π_cond(a|o,l) - π_uncond(a|o,∅))

Two inferences per frame:
  1. Conditioned   — policy.infer with real instruction  → cond_action
  2. Unconditioned — policy.infer with empty prompt ""   → uncond_action

Δ = cond - uncond reveals how much language actually steers the action.
Δ ≈ 0 → vision shortcut (language ignored). Large Δ → language is grounding.
"""
from __future__ import annotations

import io
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

matplotlib.use("Agg")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_ACTION_DIM_LABELS = [f"j{i}" for i in range(7)] + ["grip"]
_CAMERA_DIR_MAP = {"right": "varied_camera_2", "left": "varied_camera_1"}


# ── Franka Panda FK ───────────────────────────────────────────────────────────

# Standard DH parameters for Franka Panda (a, d, alpha) — joint angles added at runtime
# Source: Franka Emika technical specs
_FRANKA_DH = np.array([
    # a       d       alpha
    [0.0,     0.333,  0.0        ],
    [0.0,     0.0,   -np.pi / 2  ],
    [0.0,     0.316,  np.pi / 2  ],
    [0.0825,  0.0,    np.pi / 2  ],
    [-0.0825, 0.384, -np.pi / 2  ],
    [0.0,     0.0,    np.pi / 2  ],
    [0.088,   0.107,  np.pi / 2  ],
])


def franka_fk(q: np.ndarray) -> np.ndarray:
    """Forward kinematics for Franka Panda.  Returns end-effector (x, y, z) in metres.

    Uses standard DH convention:
        T = Rz(θ) · Tz(d) · Tx(a) · Rx(α)
    """
    T = np.eye(4)
    for i in range(7):
        a, d, alpha = _FRANKA_DH[i]
        theta = float(q[i])
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        T_i = np.array([
            [ct,    -st * ca,   st * sa,  a * ct],
            [st,     ct * ca,  -ct * sa,  a * st],
            [0.0,    sa,        ca,        d     ],
            [0.0,    0.0,       0.0,       1.0   ],
        ])
        T = T @ T_i
    return T[:3, 3]


def _ee_trajectory(q_init: np.ndarray, velocity_actions: np.ndarray) -> np.ndarray:
    """Integrate joint velocity commands to produce an end-effector trajectory.

    Args:
        q_init:           float64 (7,) — initial joint angles from trajectory.h5
        velocity_actions: float32 (8, 7) — predicted joint velocities per step

    Returns:
        float64 (9, 3) — end-effector XYZ at t=0 (initial) then t=1..8 (after each step)
    """
    q = q_init.copy().astype(np.float64)
    positions = [franka_fk(q)]
    for t in range(8):
        q = q + velocity_actions[t, :7].astype(np.float64)
        positions.append(franka_fk(q))
    return np.stack(positions)  # (9, 3)


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


_OPEN_LOOP_HORIZON = 8


def _load_frame(dataset: dict, frame_idx: int, camera: str = "right"):
    """Return (ext_img, wrist_img, joint_pos, gripper_pos, gt_action).

    gt_action is float32 (8, 8) [joint_velocity×7, gripper] from trajectory.h5,
    NaN-padded near episode end.  None if action data is absent.
    """
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
    gt_action = None
    with h5py.File(dataset["traj_h5"], "r") as f:
        joint_pos = f["observation/robot_state/joint_positions"][frame_idx].astype(np.float64)
        gripper_pos = f["observation/robot_state/gripper_position"][frame_idx : frame_idx + 1].astype(np.float64)
        # GT actions: next 8 joint-velocity + gripper commands
        if "action/joint_velocity" in f:
            traj_len = int(f["action/joint_velocity"].shape[0])
            end = min(frame_idx + _OPEN_LOOP_HORIZON, traj_len)
            n = end - frame_idx
            jv = f["action/joint_velocity"][frame_idx:end].astype(np.float32)   # (n, 7)
            gp = f["action/gripper_position"][frame_idx:end].astype(np.float32)  # (n,) or (n,1)
            if gp.ndim == 1:
                gp = gp[:, None]
            gt_action = np.concatenate([jv, gp], axis=1)  # (n, 8)
            if n < _OPEN_LOOP_HORIZON:
                pad = np.full((_OPEN_LOOP_HORIZON - n, 8), np.nan, dtype=np.float32)
                gt_action = np.concatenate([gt_action, pad], axis=0)
    return ext_img, wrist_img, joint_pos, gripper_pos, gt_action


# ── Inference ─────────────────────────────────────────────────────────────────

def _infer_action(policy, ext_img, wrist_img, joint_pos, gripper_pos, prompt: str) -> np.ndarray:
    """Run policy.infer with the given prompt, return actions (8, 8) float32.

    No attention buffer capture — we only need predicted actions here.
    """
    example = {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": wrist_img,
        "observation/joint_position": joint_pos,
        "observation/gripper_position": gripper_pos,
        "prompt": prompt,
    }
    result = policy.infer(example)
    actions = result.get("actions")
    if actions is None:
        raise RuntimeError("policy.infer() returned no 'actions' key")
    return np.asarray(actions, dtype=np.float32)[:8]  # (8, 8)


def _run_two_branches(
    policy,
    ext_img: np.ndarray,
    wrist_img: np.ndarray,
    joint_pos: np.ndarray,
    gripper_pos: np.ndarray,
    instruction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run conditioned + unconditioned branches.

    Returns:
        cond_action:   float32 (8, 8) — policy response to instruction + vision
        uncond_action: float32 (8, 8) — policy response to vision only (empty prompt)
    """
    cond_action   = _infer_action(policy, ext_img, wrist_img, joint_pos, gripper_pos, instruction)
    uncond_action = _infer_action(policy, ext_img, wrist_img, joint_pos, gripper_pos, "")
    return cond_action, uncond_action


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _delta_heatmap_png(delta: np.ndarray) -> bytes:
    """Render the (8, 8) signed delta matrix as a RdBu_r heatmap PNG.

    delta[step, dim] = cond_action[step, dim] - uncond_action[step, dim]
    Red = language pushes positive; Blue = language pushes negative.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    vmax = max(abs(delta).max(), 1e-6)
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Annotate cells
    for step in range(8):
        for dim in range(8):
            val = delta[step, dim]
            color = "white" if abs(val) > 0.4 * vmax else "#cccccc"
            ax.text(dim, step, f"{val:+.3f}", ha="center", va="center",
                    fontsize=7, color=color)

    # Highlight most language-driven cells with white dashed border
    threshold = 0.5 * vmax
    for step in range(8):
        for dim in range(8):
            if abs(delta[step, dim]) >= threshold:
                rect = plt.Rectangle(
                    (dim - 0.5, step - 0.5), 1, 1,
                    fill=False, edgecolor="white", linewidth=1.5, linestyle="--"
                )
                ax.add_patch(rect)

    ax.set_xticks(range(8))
    ax.set_xticklabels(_ACTION_DIM_LABELS, color="white", fontsize=9)
    ax.set_yticks(range(8))
    ax.set_yticklabels([f"step {i}" for i in range(8)], color="white", fontsize=9)
    ax.set_title("Δ = Cond − Uncond  (language-induced action shift)", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_influence_bars(delta: np.ndarray, key_suffix: str = "") -> None:
    """Plotly bar chart: per-step language influence = ||Δ[step, :]||₂."""
    norms = np.linalg.norm(delta, axis=1)  # (8,)
    norm_max = norms.max() + 1e-8
    colors = [
        f"rgba(76,155,232,{0.3 + 0.7 * (v / norm_max)})"
        for v in norms
    ]
    fig = go.Figure(go.Bar(
        x=[f"Step {i}" for i in range(8)],
        y=norms.tolist(),
        marker_color=colors,
        hovertemplate="Step %{x}: ‖Δ‖₂ = %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Per-step language influence  ‖Δ[step]‖₂",
        height=220,
        margin=dict(l=40, r=20, t=40, b=30),
        yaxis_title="L2 norm of Δ",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"cag_infl_{key_suffix}")


def _render_action_comparison(
    cond: np.ndarray,
    uncond: np.ndarray,
    omega: float,
    key_suffix: str = "",
) -> None:
    """Plotly 2×4 subplot grid: 3 curves × 8 action dims.

    Uncond (orange solid), Cond (blue dashed), CAG guided (green dash-dot).
    """
    guided = uncond + omega * (cond - uncond)
    steps = list(range(8))

    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=_ACTION_DIM_LABELS,
        shared_xaxes=True,
        vertical_spacing=0.22,
        horizontal_spacing=0.08,
    )

    for dim_i, _label in enumerate(_ACTION_DIM_LABELS):
        row, col = dim_i // 4 + 1, dim_i % 4 + 1
        show_legend = dim_i == 0

        fig.add_trace(go.Scatter(
            x=steps, y=uncond[:, dim_i].tolist(),
            mode="lines+markers",
            name="Uncond (vision-only)",
            showlegend=show_legend,
            line=dict(color="#f0a500"),
            marker=dict(size=5),
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=steps, y=cond[:, dim_i].tolist(),
            mode="lines+markers",
            name="Cond (language+vision)",
            showlegend=show_legend,
            line=dict(color="#4c9be8", dash="dash"),
            marker=dict(size=5),
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=steps, y=guided[:, dim_i].tolist(),
            mode="lines+markers",
            name=f"CAG guided (ω={omega:.1f})",
            showlegend=show_legend,
            line=dict(color="#3dcc7a", dash="dashdot"),
            marker=dict(size=5),
        ), row=row, col=col)

    fig.update_layout(
        title=(
            "Action comparison — Uncond (orange) · Cond (blue dashed) · "
            f"CAG guided ω={omega:.1f} (green dash-dot)"
        ),
        height=400,
        margin=dict(l=40, r=20, t=70, b=30),
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"cag_cmp_{key_suffix}")


def _render_3d_trajectories(
    q_init: np.ndarray,
    cond: np.ndarray,
    uncond: np.ndarray,
    omega: float,
    gt_action: np.ndarray | None = None,
    key_suffix: str = "",
) -> None:
    """Plotly 3D interactive chart: end-effector trajectories for all branches.

    Integrates joint velocity predictions (and optionally GT velocities) from
    the real initial joint configuration read from trajectory.h5, applies
    Franka Panda FK, and plots the resulting end-effector (x, y, z) path.
    The chart is fully drag-rotatable and zoomable.
    """
    guided_actions = uncond + omega * (cond - uncond)

    ee_cond   = _ee_trajectory(q_init, cond)          # (9, 3)
    ee_uncond = _ee_trajectory(q_init, uncond)         # (9, 3)
    ee_guided = _ee_trajectory(q_init, guided_actions) # (9, 3)

    step_labels = ["init"] + [f"t={i}" for i in range(1, 9)]

    def _trace(ee, name, color, dash, symbol="circle"):
        return go.Scatter3d(
            x=ee[:, 0].tolist(),
            y=ee[:, 1].tolist(),
            z=ee[:, 2].tolist(),
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=4, dash=dash),
            marker=dict(
                size=[10] + [5] * 8,  # larger marker at start
                color=color,
                symbol=["diamond"] + [symbol] * 8,
            ),
            text=step_labels,
            hovertemplate=(
                "<b>" + name + "</b><br>"
                "Step: %{text}<br>"
                "x=%{x:.4f} m<br>y=%{y:.4f} m<br>z=%{z:.4f} m"
                "<extra></extra>"
            ),
        )

    traces = [
        _trace(ee_uncond, "Uncond (vision-only)",        "#f0a500", "solid"),
        _trace(ee_cond,   "Cond (language+vision)",      "#4c9be8", "dash"),
        _trace(ee_guided, f"CAG guided (ω={omega:.1f})", "#3dcc7a", "dashdot"),
    ]

    # GT trajectory from real trajectory.h5 data (if available and not all-NaN)
    if gt_action is not None:
        valid = ~np.isnan(gt_action).any(axis=1)
        if valid.any():
            # Fill NaN steps with last valid step for FK continuity, mark them separately
            gt_filled = gt_action.copy()
            last_valid = gt_action[valid][-1]
            gt_filled[~valid] = last_valid
            ee_gt = _ee_trajectory(q_init, gt_filled)  # (9, 3)
            # Mask out NaN steps in the display
            for t in range(1, 9):
                if t - 1 < len(valid) and not valid[t - 1]:
                    ee_gt[t] = np.nan
            traces.append(_trace(ee_gt, "GT (from trajectory.h5)", "#e855e8", "dot", "square"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=dict(text="x (m)", font=dict(color="white")), backgroundcolor="#0e1117",
                       gridcolor="#333", showbackground=True, tickfont=dict(color="white")),
            yaxis=dict(title=dict(text="y (m)", font=dict(color="white")), backgroundcolor="#0e1117",
                       gridcolor="#333", showbackground=True, tickfont=dict(color="white")),
            zaxis=dict(title=dict(text="z (m)", font=dict(color="white")), backgroundcolor="#0e1117",
                       gridcolor="#333", showbackground=True, tickfont=dict(color="white")),
            bgcolor="#0e1117",
        ),
        legend=dict(orientation="h", y=1.05, font=dict(color="white")),
        paper_bgcolor="#0e1117",
        font_color="white",
        height=550,
        margin=dict(l=0, r=0, t=50, b=0),
        title=dict(
            text=f"End-effector trajectory (Franka Panda FK)  —  ω={omega:.1f}",
            font=dict(color="white"),
        ),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"cag_3d_{key_suffix}")


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
    """Render the Language Grounding (CAG) tab."""
    st.markdown("### Language Grounding — Counterfactual Action Guidance (CAG)")
    st.caption(
        "From **'Vision Overrides Language in VLAs'** (arxiv 2602.17659): "
        "run two policy branches — conditioned on instruction vs. vision-only (empty prompt) — "
        "and compute the **language-induced action delta Δ = Cond − Uncond**. "
        "Δ ≈ 0 → vision shortcut (language ignored). Large Δ → language is actively grounding behavior."
    )

    cat = _catalogue()
    if not cat:
        st.error("No local datasets found in `data/example/`.")
        return

    # ── Step 1: Image source ──────────────────────────────────────────────────
    with st.expander("① Image source & frame", expanded=True):
        dataset_name = st.selectbox("Dataset", list(cat.keys()), key="cag_dataset")
        dataset = cat[dataset_name]
        max_frame = dataset["n_frames"] - 1
        frame_idx = st.slider("Frame", 0, max_frame, 0, key="cag_frame")
        camera = st.radio("Ext camera", ["right", "left"], horizontal=True, key="cag_camera")
        try:
            ext_img, wrist_img, joint_pos, gripper_pos, gt_action = _load_frame(dataset, frame_idx, camera)
            c1, c2 = st.columns(2)
            c1.image(ext_img, caption=f"Exterior ({camera})", use_container_width=True)
            c2.image(wrist_img, caption="Wrist", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load frame: {e}")
            return

    # ── Step 2: Instruction & settings ───────────────────────────────────────
    with st.expander("② Instruction & settings", expanded=True):
        instruction = st.text_input(
            "Instruction (conditioned branch)",
            value=dataset["default_prompt"],
            key="cag_instruction",
            help="The unconditioned branch always uses an empty prompt.",
        )
        gpu_device = st.selectbox("GPU", _gpu_devices(), key="cag_gpu")

    # ── Step 3: Run ───────────────────────────────────────────────────────────
    cache_key = f"cag_{dataset_name}_{frame_idx}_{camera}_{instruction[:40]}"
    run_btn = st.button("▶ Run CAG Analysis", type="primary", key="cag_run")

    if run_btn:
        from viz.dashboard.inference import load_model  # noqa: PLC0415
        with st.spinner("Loading model…"):
            try:
                ckpt_path = os.path.join(_PROJECT_ROOT, "checkpoints/viz", "pi05_droid_pytorch")
                policy = load_model(ckpt_path, device=gpu_device)
            except Exception as e:
                st.error(f"Model load failed: {e}")
                return

        with st.spinner("Running conditioned branch…"):
            try:
                cond_action, uncond_action = _run_two_branches(
                    policy, ext_img, wrist_img, joint_pos, gripper_pos, instruction
                )
                st.session_state[cache_key] = {
                    "cond_action": cond_action,
                    "uncond_action": uncond_action,
                    "q_init": joint_pos,
                    "gt_action": gt_action,  # may be None near episode end
                }
                st.success("CAG analysis complete.")
            except Exception as e:
                st.error(f"Inference failed: {e}")
                return

    # ── Step 4: Visualise ─────────────────────────────────────────────────────
    result = st.session_state.get(cache_key)
    if result is None:
        st.info("Configure settings above and click **▶ Run CAG Analysis** to begin.")
        return

    cond      = result["cond_action"]    # (8, 8)
    uncond    = result["uncond_action"]  # (8, 8)
    q_init    = result["q_init"]         # (7,) initial joint angles
    gt_action = result.get("gt_action")  # (8, 8) or None
    delta     = cond - uncond            # (8, 8) — language-induced shift

    st.markdown("---")
    st.markdown("### Results")

    # ── Scalar language influence ─────────────────────────────────────────────
    frob = float(np.linalg.norm(delta, "fro"))
    max_possible = float(np.linalg.norm(cond, "fro"))  # rough upper bound
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(
        "Language influence  ‖Δ‖_F",
        f"{frob:.4f}",
        help="Frobenius norm of the action delta. Near 0 = language ignored (vision shortcut).",
    )
    col_m2.metric(
        "‖Cond‖_F",
        f"{max_possible:.4f}",
        help="Magnitude of conditioned action output.",
    )
    col_m3.metric(
        "Δ / ‖Cond‖ (%)",
        f"{100 * frob / (max_possible + 1e-8):.1f}%",
        help="Language influence as percentage of conditioned action magnitude.",
    )

    # ── ω slider (live, no re-inference) ─────────────────────────────────────
    st.markdown("#### Guidance scale ω")
    st.caption(
        "ω=0 → pure vision-only action. ω=1 → standard conditioned action. "
        "ω>1 → amplified language conditioning (CAG)."
    )
    omega = st.slider(
        "ω (guidance scale)",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key="cag_omega",
    )

    # ── Panel A: action comparison line chart ─────────────────────────────────
    st.markdown("#### Panel A — Action trajectories (all dims)")
    _render_action_comparison(cond, uncond, omega, key_suffix=cache_key[-8:])

    # ── Panel B: delta heatmap ────────────────────────────────────────────────
    st.markdown("#### Panel B — Language delta heatmap  Δ = Cond − Uncond")
    st.caption(
        "Red = language shifts action positive. Blue = negative. "
        "White dashed border = strongest language influence (|Δ| > 50% of max)."
    )
    st.image(_delta_heatmap_png(delta), use_container_width=True)

    # ── Panel C: per-step influence bars ──────────────────────────────────────
    st.markdown("#### Panel C — Per-step language influence")
    _render_influence_bars(delta, key_suffix=cache_key[-8:])

    # ── Panel D: 3D end-effector trajectory ──────────────────────────────────
    st.markdown("---")
    st.markdown("#### Panel D — End-effector trajectory in 3D space")
    st.caption(
        "Integrates the predicted joint velocity actions from the initial robot configuration "
        "and applies Franka Panda forward kinematics to show where the end-effector would move. "
        "**Drag to rotate, scroll to zoom.** The ω slider above live-updates this chart."
    )
    _render_3d_trajectories(q_init, cond, uncond, omega, gt_action=gt_action, key_suffix=cache_key[-8:])

    # ── Raw data expander ─────────────────────────────────────────────────────
    with st.expander("Raw action values"):
        import pandas as pd
        tabs_raw = st.tabs(["Cond", "Uncond", "Δ", f"CAG guided (ω={omega:.1f})"])
        guided = uncond + omega * delta
        for tab_r, arr, name in zip(
            tabs_raw,
            [cond, uncond, delta, guided],
            ["Cond", "Uncond", "Δ", "CAG guided"],
        ):
            with tab_r:
                df = pd.DataFrame(arr, columns=_ACTION_DIM_LABELS)
                df.index.name = "step"
                st.dataframe(df.style.format("{:+.4f}"), use_container_width=True)
