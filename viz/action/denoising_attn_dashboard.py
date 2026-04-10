"""Interactive dashboard — action→token attention over denoising steps.

Run:
    uv run streamlit run viz/denoising_attn_dashboard.py

Tabs:
  📈 Attention over steps  — plotly line plots per selected layer (apple-to-apple)
  🖼  Image heatmap         — overlay at selected denoising step, col per layer
  ⏱  Attention timeline    — strip of overlays across all steps, row per layer
  🗂  All layers            — full 18-layer grid at selected step
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
for p in [str(ROOT), str(ROOT / "src"), str(ROOT / "viz")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Constants ─────────────────────────────────────────────────────────────────
DUCK_DIR          = ROOT / "data" / "example" / "duck"
DEFAULT_CKPT      = str(ROOT / "checkpoints" / "viz" / "pi05_droid_pytorch")
OPEN_LOOP_HORIZON = 8
NUM_LAYERS        = 18

EXT_START, EXT_END     = 0,   256
WRIST_START, WRIST_END = 256, 512
TEXT_START             = 768

GROUPS = ["ext_img", "wrist_img", "text", "action_self"]
GROUP_COLOR = {
    "ext_img":    "#4e79a7",
    "wrist_img":  "#f28e2b",
    "text":       "#59a14f",
    "action_self":"#e15759",
}
GROUP_LABEL = {
    "ext_img":    "Ext camera (0:256)",
    "wrist_img":  "Wrist camera (256:512)",
    "text":       "Text tokens",
    "action_self":"Action (self)",
}

# ── Data helpers ──────────────────────────────────────────────────────────────

def load_duck_frame(frame_idx: int) -> dict:
    frames = DUCK_DIR / "frames"
    ext  = np.array(Image.open(frames / "varied_camera_2" / f"{frame_idx:05d}.jpg").convert("RGB"))
    hand = np.array(Image.open(frames / "hand_camera"     / f"{frame_idx:05d}.jpg").convert("RGB"))
    with h5py.File(DUCK_DIR / "trajectory.h5", "r") as f:
        jp  = f["observation/robot_state/joint_positions"][frame_idx].astype(np.float64)
        gp  = f["observation/robot_state/gripper_position"][frame_idx:frame_idx+1].astype(np.float64)
        tl  = f["action/joint_velocity"].shape[0]
        end = min(frame_idx + OPEN_LOOP_HORIZON, tl)
        n   = end - frame_idx
        jv  = f["action/joint_velocity"][frame_idx:end].astype(np.float32)
        g2  = f["action/gripper_position"][frame_idx:end].astype(np.float32)
        gt  = np.concatenate([jv, g2[:, None]], axis=1)
        if n < OPEN_LOOP_HORIZON:
            gt = np.concatenate([gt, np.full((OPEN_LOOP_HORIZON - n, 8), np.nan, np.float32)])
    return {
        "observation/exterior_image_1_left": ext,
        "observation/wrist_image_left":      hand,
        "observation/joint_position":        jp,
        "observation/gripper_position":      gp,
        "prompt":                            "pick up the duck",
        "gt_action":                         gt,
    }


def _resize224(arr: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize((224, 224)))


# ── Policy (cached across reruns) ─────────────────────────────────────────────

@st.cache_resource
def load_policy(checkpoint: str, device: str):
    from openpi.training import config as _cfg
    from openpi.policies import policy_config as _pc
    raw = Path(checkpoint).name.removesuffix("_pytorch")
    candidate = raw
    while candidate:
        try:
            _cfg.get_config(candidate)
            break
        except (ValueError, KeyError):
            idx = candidate.rfind("_")
            if idx == -1:
                candidate = "pi05_droid"
                break
            candidate = candidate[:idx]
    return _pc.create_trained_policy(_cfg.get_config(candidate), checkpoint, pytorch_device=device)


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(policy, example: dict) -> dict:
    """Run one forward pass, capture every NFE step. Stores results in a plain dict."""
    from openpi.models_pytorch import gemma_pytorch as _gpt
    _gpt.enable_suffix_attn_steps_buffer()
    try:
        policy.infer(example)
        steps_buf = list(_gpt.get_suffix_attn_steps_buffer() or [])
    finally:
        _gpt.clear_suffix_attn_steps_buffer()

    if not steps_buf:
        return {}

    first_layer = min(steps_buf[0])
    seq_len = steps_buf[0][first_layer].shape[-1]
    n_text  = max(1, seq_len - TEXT_START - OPEN_LOOP_HORIZON)

    return {
        "steps_buf": steps_buf,
        "seq_len":   seq_len,
        "n_text":    n_text,
        "ext_img":   _resize224(example["observation/exterior_image_1_left"]),
        "wrist_img": _resize224(example["observation/wrist_image_left"]),
    }


# ── Attention helpers ─────────────────────────────────────────────────────────

def group_masses(steps_buf, layer, n_text, action_step=None):
    """Return {group: array(n_steps)} attention mass per token group."""
    text_end = TEXT_START + n_text
    out = {g: [] for g in GROUPS}
    for sd in steps_buf:
        a = sd[layer][0]                            # (n_heads, 8, seq)
        if action_step is not None:
            a = a[:, action_step:action_step+1, :]
        m = a.mean(axis=(0, 1))                     # (seq,)
        out["ext_img"].append(    m[EXT_START:EXT_END].sum())
        out["wrist_img"].append(  m[WRIST_START:WRIST_END].sum())
        out["text"].append(       m[TEXT_START:text_end].sum())
        out["action_self"].append(m[text_end:].sum())
    return {k: np.array(v) for k, v in out.items()}


def heatmap_16x16(step_dict, layer, camera="wrist", action_step=None):
    """16×16 attention heatmap for one camera at one denoising step."""
    a = step_dict[layer][0]                         # (n_heads, 8, seq)
    if action_step is not None:
        a = a[:, action_step:action_step+1, :]
    m = a.mean(axis=(0, 1))                         # (seq,)
    patch = m[WRIST_START:WRIST_END] if camera == "wrist" else m[EXT_START:EXT_END]
    return patch.reshape(16, 16)


def blend_overlay(image_rgb, hmap16, alpha=0.55, cmap="hot", vmin=None, vmax=None):
    """Resize 16×16 heatmap and alpha-blend over an RGB image (H×W×3 uint8)."""
    import matplotlib.cm as mcm
    H, W = image_rgb.shape[:2]
    hmap = cv2.resize(hmap16.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    lo   = hmap.min() if vmin is None else vmin
    hi   = hmap.max() if vmax is None else vmax
    norm = (hmap - lo) / (hi - lo + 1e-9)
    colored = (mcm.get_cmap(cmap)(norm)[:, :, :3] * 255).astype(np.uint8)
    return (image_rgb * (1 - alpha) + colored * alpha).clip(0, 255).astype(np.uint8)


def global_vmax_for(steps_buf, layers, steps, camera, action_step):
    """Compute global max across layers and steps for consistent colorscale."""
    return max(
        heatmap_16x16(steps_buf[s], l, camera, action_step).max()
        for l in layers for s in steps
    )


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Denoising Attention", layout="wide")
st.title("Action → Token Attention over Denoising Steps")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Episode")
    frame_idx  = st.number_input("Duck frame index", 0, 90, 40)
    checkpoint = st.text_input("Checkpoint", DEFAULT_CKPT)
    device     = st.selectbox("Device", ["cuda:0", "cuda:1", "cpu"])
    run_btn    = st.button("▶  Run inference", type="primary", use_container_width=True)

    if run_btn or "inf" not in st.session_state:
        with st.spinner("Running inference — capturing all NFE steps…"):
            example = load_duck_frame(int(frame_idx))
            policy  = load_policy(checkpoint, device)
            inf     = run_inference(policy, example)
        if inf:
            st.session_state["inf"] = inf
            st.success(f"✓  {len(inf['steps_buf'])} NFE steps captured")
        else:
            st.error("No steps captured — check model/checkpoint")

    if "inf" not in st.session_state:
        st.info("Press ▶ Run inference to start.")
        st.stop()

    inf      = st.session_state["inf"]
    n_steps  = len(inf["steps_buf"])
    n_text   = inf["n_text"]
    steps_buf = inf["steps_buf"]
    ext_img   = inf["ext_img"]
    wrist_img = inf["wrist_img"]

    st.divider()
    st.header("Controls")

    sel_layers = st.multiselect(
        "Layers to compare",
        list(range(NUM_LAYERS)),
        default=[0, 4, 8, 12, 17],
    )
    if not sel_layers:
        sel_layers = [8]

    step_slider = st.slider("Denoising step", 0, n_steps - 1, 0)

    action_opt  = st.selectbox("Action step", ["avg all"] + [str(i) for i in range(OPEN_LOOP_HORIZON)])
    action_step = None if action_opt == "avg all" else int(action_opt)

    cam_choice  = st.radio("Camera", ["Wrist", "Ext"], horizontal=True)
    camera      = "wrist" if cam_choice == "Wrist" else "ext"
    cam_img     = wrist_img if camera == "wrist" else ext_img

    alpha       = st.slider("Overlay opacity", 0.1, 0.9, 0.55)
    global_norm = st.checkbox("Global colorscale (apple-to-apple)", value=False,
                              help="Same heatmap scale across all layers and steps")

    st.caption(f"seq_len={inf['seq_len']}  n_text≈{n_text}  steps={n_steps}")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Attention over steps",
    "🖼   Image heatmap",
    "⏱  Attention timeline",
    "🗂  All layers",
])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Plotly line plots, one subplot per selected layer
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.caption(
        "Attention mass allocated to each token group per denoising step. "
        "Dashed line = currently selected step (from sidebar slider)."
    )

    n_cols = len(sel_layers)
    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[f"Layer {l}" for l in sel_layers],
        shared_yaxes=global_norm,
        horizontal_spacing=0.06 / max(n_cols, 1),
    )

    xs = list(range(n_steps))
    for ci, layer in enumerate(sel_layers):
        masses = group_masses(steps_buf, layer, n_text, action_step)
        for g in GROUPS:
            fig.add_trace(
                go.Scatter(
                    x=xs, y=masses[g].tolist(),
                    name=GROUP_LABEL[g],
                    mode="lines",
                    line=dict(color=GROUP_COLOR[g], width=2.5),
                    showlegend=(ci == 0),
                    legendgroup=g,
                    hovertemplate=f"step %{{x}}<br>{GROUP_LABEL[g]}: %{{y:.4f}}<extra></extra>",
                ),
                row=1, col=ci + 1,
            )
        # Vertical marker at selected step
        fig.add_shape(
            type="line", x0=step_slider, x1=step_slider, y0=0, y1=1,
            yref="paper", line=dict(color="gray", dash="dash", width=1.5),
            row=1, col=ci + 1,
        )

    fig.update_layout(
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=80, b=40),
    )
    fig.update_xaxes(title_text="Denoising step (0=most noisy)")
    fig.update_yaxes(title_text="Attention mass", col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Stacked area for representative layer (middle of selection)
    rep = sel_layers[len(sel_layers) // 2]
    st.subheader(f"Normalized budget — layer {rep}")
    masses = group_masses(steps_buf, rep, n_text, action_step)
    stack  = np.stack([masses[g] for g in GROUPS])            # (4, n_steps)
    norm   = stack / stack.sum(axis=0, keepdims=True).clip(1e-9)

    fig2 = go.Figure()
    for gi, g in enumerate(GROUPS):
        fig2.add_trace(go.Scatter(
            x=xs, y=norm[gi].tolist(),
            name=GROUP_LABEL[g],
            stackgroup="one",
            fillcolor=GROUP_COLOR[g],
            line=dict(color=GROUP_COLOR[g], width=0.5),
            hovertemplate=f"{GROUP_LABEL[g]}: %{{y:.2%}}<extra></extra>",
        ))
    fig2.add_vline(x=step_slider, line_dash="dash", line_color="gray", opacity=0.6)
    fig2.update_layout(
        height=300, yaxis=dict(tickformat=".0%", range=[0, 1]),
        xaxis_title="Denoising step (0=most noisy)", yaxis_title="Fraction",
        hovermode="x unified", margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Image heatmap at selected step, columns = selected layers
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader(f"Step {step_slider} of {n_steps - 1}  |  {cam_choice} camera")
    st.caption("Columns = layers selected in sidebar. Left-most column = original image.")

    # Compute global vmax across selected layers at this step if needed
    vmax = global_vmax_for(steps_buf, sel_layers, [step_slider], camera, action_step) \
           if global_norm else None

    cols = st.columns([1] + [1] * len(sel_layers))
    cols[0].image(cam_img, caption="Original", use_container_width=True)

    for ci, layer in enumerate(sel_layers):
        h16     = heatmap_16x16(steps_buf[step_slider], layer, camera, action_step)
        overlay = blend_overlay(cam_img, h16, alpha=alpha, vmax=vmax)
        cols[ci + 1].image(overlay, caption=f"Layer {layer}", use_container_width=True)

    # Per-layer raw heatmap (no image) in a second row for precision
    st.markdown("**Raw 16×16 patch heatmaps** (same colorscale)")
    import matplotlib.pyplot as plt
    import io

    all_hmaps = [heatmap_16x16(steps_buf[step_slider], l, camera, action_step) for l in sel_layers]
    combined_vmax = max(h.max() for h in all_hmaps) + 1e-9

    fig_raw, axes = plt.subplots(1, len(sel_layers), figsize=(3 * len(sel_layers), 3))
    if len(sel_layers) == 1:
        axes = [axes]
    for ax, layer, h16 in zip(axes, sel_layers, all_hmaps):
        im = ax.imshow(h16, cmap="hot", vmin=0, vmax=combined_vmax, interpolation="nearest")
        ax.set_title(f"L{layer}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    buf = io.BytesIO()
    fig_raw.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig_raw)
    st.image(buf.getvalue(), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Attention timeline: strip of overlays across steps, one row per layer
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.caption(
        "Each row = one layer. Each column = one denoising step. "
        "Shows how the attention heatmap evolves as the model denoises."
    )

    # Downsample steps to ≤10 for readability
    max_cols   = 10
    stride     = max(1, n_steps // max_cols)
    show_steps = list(range(0, n_steps, stride))
    if n_steps - 1 not in show_steps:
        show_steps.append(n_steps - 1)

    # Global vmax across all shown steps + layers
    vmax_tl = global_vmax_for(steps_buf, sel_layers, show_steps, camera, action_step) \
              if global_norm else None

    # Header
    header = st.columns(len(show_steps))
    for ci, s in enumerate(show_steps):
        lbl = "first" if s == 0 else ("last" if s == n_steps - 1 else str(s))
        header[ci].markdown(f"<div style='text-align:center;font-size:12px'><b>step {lbl}</b></div>",
                            unsafe_allow_html=True)
    st.markdown("---")

    for layer in sel_layers:
        st.markdown(f"**Layer {layer}**")
        img_cols = st.columns(len(show_steps))
        for ci, s in enumerate(show_steps):
            h16     = heatmap_16x16(steps_buf[s], layer, camera, action_step)
            overlay = blend_overlay(cam_img, h16, alpha=alpha, vmax=vmax_tl)
            img_cols[ci].image(overlay, use_container_width=True)
        st.markdown("")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — All 18 layers at selected step (quick overview grid)
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader(f"All {NUM_LAYERS} layers  |  step {step_slider}  |  {cam_choice} camera")
    st.caption("🔵 = layer currently in the comparison selection. Global colorscale controlled by sidebar.")

    vmax4 = global_vmax_for(steps_buf, list(range(NUM_LAYERS)), [step_slider], camera, action_step) \
            if global_norm else None

    COLS_PER_ROW = 6
    all_layers   = list(range(NUM_LAYERS))
    for row_start in range(0, NUM_LAYERS, COLS_PER_ROW):
        chunk = all_layers[row_start : row_start + COLS_PER_ROW]
        cols  = st.columns(len(chunk))
        for ci, layer in enumerate(chunk):
            h16     = heatmap_16x16(steps_buf[step_slider], layer, camera, action_step)
            overlay = blend_overlay(cam_img, h16, alpha=alpha, vmax=vmax4)
            mark    = "🔵 " if layer in sel_layers else ""
            cols[ci].image(overlay, caption=f"{mark}L{layer}", use_container_width=True)
