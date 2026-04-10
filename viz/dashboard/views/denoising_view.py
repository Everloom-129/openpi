"""Denoising-step attention view.

Shows how action-token attention (to image / text / action tokens) evolves
across the NFE denoising steps of the flow-matching policy.

Two panels:
  Frame level   — line plots + image overlays for the currently selected frame.
  Trajectory    — group-mass traces over all episode frames at a chosen step.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from viz.dashboard.loader import (
    load_images,
    load_suffix_denoising,
    load_suffix_denoising_trajectory,
)

# ── Token layout (matches attn_h5_writer constants) ───────────────────────────
TOTAL_IMAGE_TOKENS = 512
NUM_IMAGE_TOKENS   = 256
TEXT_START_IDX     = 768
NUM_LAYERS         = 18
NUM_ACTION_STEPS   = 8

GROUPS       = ["ext_img", "wrist_img", "text", "action_self"]
GROUP_COLOR  = {
    "ext_img":    "#4e79a7",
    "wrist_img":  "#f28e2b",
    "text":       "#59a14f",
    "action_self":"#e15759",
}
GROUP_LABEL  = {
    "ext_img":    "Ext camera",
    "wrist_img":  "Wrist camera",
    "text":       "Text tokens",
    "action_self":"Action (self)",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _blend_overlay(image_rgb, hmap16x16, alpha=0.55):
    """Alpha-blend a 16×16 heatmap over an RGB image (H×W×3 uint8)."""
    import matplotlib.cm as mcm
    H, W = image_rgb.shape[:2]
    hmap = cv2.resize(hmap16x16.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    lo, hi = hmap.min(), hmap.max()
    norm    = (hmap - lo) / (hi - lo + 1e-9)
    colored = (mcm.get_cmap("hot")(norm)[:, :, :3] * 255).astype(np.uint8)
    return (image_rgb * (1 - alpha) + colored * alpha).clip(0, 255).astype(np.uint8)


def _get_heatmap(action_to_img_step, camera="wrist", action_step=None):
    """Return 16×16 patch array from (8, 512) or (512,) attention row."""
    if action_to_img_step.ndim == 2:
        row = action_to_img_step.mean(axis=0) if action_step is None else action_to_img_step[action_step]
    else:
        row = action_to_img_step   # already (512,)
    patch = row[NUM_IMAGE_TOKENS:TOTAL_IMAGE_TOKENS] if camera == "wrist" else row[:NUM_IMAGE_TOKENS]
    return patch.reshape(16, 16)


# ── Frame-level panel ─────────────────────────────────────────────────────────

def _render_frame_panel(
    layers: list[int],
    images: dict,
    h5_path: str | None = None,
    mem_denoising: dict | None = None,
) -> None:
    """Line plots + image overlays for a single frame.

    Data source priority: mem_denoising (in-memory, Online mode) > h5_path (HDF5, Results mode).
    """

    # Sidebar-style controls inside the panel
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        camera = st.radio("Camera", ["Wrist", "Ext"], horizontal=True, key="dn_cam")
    with col_ctrl2:
        action_opt = st.selectbox("Action step", ["avg all"] + [str(i) for i in range(NUM_ACTION_STEPS)], key="dn_as")
        action_step = None if action_opt == "avg all" else int(action_opt)
    with col_ctrl3:
        alpha       = st.slider("Overlay opacity", 0.1, 0.9, 0.55, key="dn_alpha")
        global_norm = st.checkbox("Global colorscale", False, key="dn_gnorm",
                                  help="Same heatmap scale across all layers (apple-to-apple)")

    cam_key = "wrist" if camera == "Wrist" else "ext"
    cam_img = images.get(cam_key)
    if cam_img is None:
        cam_img = np.full((224, 224, 3), 30, dtype=np.uint8)

    # Resolve data for all selected layers
    layer_data: dict[int, dict] = {}
    for layer in layers:
        if mem_denoising is not None:
            # Online mode: read from in-memory dict keyed by "layer_{i}"
            d = mem_denoising.get(f"layer_{layer}")
        else:
            # Results/Offline mode: read from HDF5
            d = load_suffix_denoising(h5_path, layer) if h5_path else None
        if d is not None:
            layer_data[layer] = d

    if not layer_data:
        if mem_denoising is not None:
            st.info("No denoising data returned by inference — check that `enable_suffix_attn_steps_buffer` ran.")
        else:
            st.info("No `/suffix_denoising` data in this HDF5. Re-run `pipeline.py` to generate it.")
        return

    n_steps = next(iter(layer_data.values()))["n_steps"]
    xs = list(range(n_steps))

    # ── Plotly line plots ─────────────────────────────────────────────────────
    st.markdown("**Attention mass per token group over denoising steps**")
    valid_layers = list(layer_data)
    fig = make_subplots(
        rows=1, cols=len(valid_layers),
        subplot_titles=[f"Layer {l}" for l in valid_layers],
        shared_yaxes=global_norm,
        horizontal_spacing=0.05,
    )
    for ci, layer in enumerate(valid_layers):
        masses = layer_data[layer]["group_masses"]   # (n_steps, 4)
        for gi, g in enumerate(GROUPS):
            fig.add_trace(
                go.Scatter(
                    x=xs, y=masses[:, gi].tolist(),
                    name=GROUP_LABEL[g],
                    mode="lines",
                    line=dict(color=GROUP_COLOR[g], width=2.5),
                    showlegend=(ci == 0),
                    legendgroup=g,
                    hovertemplate=f"step %{{x}}<br>{GROUP_LABEL[g]}: %{{y:.4f}}<extra></extra>",
                ),
                row=1, col=ci + 1,
            )
    fig.update_layout(
        height=350, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
    )
    fig.update_xaxes(title_text="Denoising step (0=most noisy)")
    fig.update_yaxes(title_text="Attention mass", col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── Stacked area (middle layer) ───────────────────────────────────────────
    rep = valid_layers[len(valid_layers) // 2]
    masses = layer_data[rep]["group_masses"]          # (n_steps, 4)
    totals = masses.sum(axis=1, keepdims=True).clip(1e-9)
    norm   = masses / totals

    fig2 = go.Figure()
    for gi, g in enumerate(GROUPS):
        fig2.add_trace(go.Scatter(
            x=xs, y=norm[:, gi].tolist(),
            name=GROUP_LABEL[g], stackgroup="one",
            fillcolor=GROUP_COLOR[g],
            line=dict(color=GROUP_COLOR[g], width=0.5),
            hovertemplate=f"{GROUP_LABEL[g]}: %{{y:.1%}}<extra></extra>",
        ))
    fig2.update_layout(
        title=f"Normalized budget — layer {rep}",
        height=250, hovermode="x unified",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        xaxis_title="Denoising step", yaxis_title="Fraction",
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Image overlays: step slider × layer columns ───────────────────────────
    st.markdown("**Action→image heatmap overlay**")
    step_sel = st.slider("Denoising step", 0, n_steps - 1, 0, key="dn_step_slider")

    # Compute global vmax if needed
    vmax = None
    if global_norm:
        all_vals = []
        for layer in valid_layers:
            a2i = layer_data[layer]["action_to_img_steps"][step_sel]  # (8, 512)
            h16 = _get_heatmap(a2i, cam_key, action_step)
            all_vals.append(h16.max())
        vmax = max(all_vals) if all_vals else None

    img_cols = st.columns([1] + [1] * len(valid_layers))
    img_cols[0].image(cam_img, caption="Original", use_container_width=True)
    for ci, layer in enumerate(valid_layers):
        a2i = layer_data[layer]["action_to_img_steps"][step_sel]   # (8, 512)
        h16 = _get_heatmap(a2i, cam_key, action_step)
        if vmax is not None:
            h16 = h16 / (vmax + 1e-9)
        overlay = _blend_overlay(cam_img, h16, alpha=alpha)
        img_cols[ci + 1].image(overlay, caption=f"L{layer}", use_container_width=True)

    # ── Attention timeline strip ──────────────────────────────────────────────
    with st.expander("Attention timeline (all steps × selected layers)", expanded=False):
        max_cols  = 10
        stride    = max(1, n_steps // max_cols)
        show_steps = list(range(0, n_steps, stride))
        if n_steps - 1 not in show_steps:
            show_steps.append(n_steps - 1)

        vmax_tl = None
        if global_norm:
            all_vals = [
                _get_heatmap(layer_data[l]["action_to_img_steps"][s], cam_key, action_step).max()
                for l in valid_layers for s in show_steps
            ]
            vmax_tl = max(all_vals) if all_vals else None

        hdr = st.columns(len(show_steps))
        for ci, s in enumerate(show_steps):
            lbl = "first" if s == 0 else ("last" if s == n_steps - 1 else str(s))
            hdr[ci].markdown(
                f"<div style='text-align:center;font-size:11px'><b>step {lbl}</b></div>",
                unsafe_allow_html=True,
            )

        for layer in valid_layers:
            st.markdown(f"**Layer {layer}**")
            row_cols = st.columns(len(show_steps))
            for ci, s in enumerate(show_steps):
                a2i = layer_data[layer]["action_to_img_steps"][s]
                h16 = _get_heatmap(a2i, cam_key, action_step)
                if vmax_tl is not None:
                    h16 = h16 / (vmax_tl + 1e-9)
                overlay = _blend_overlay(cam_img, h16, alpha=alpha)
                row_cols[ci].image(overlay, use_container_width=True)

    # ── All-layers grid ───────────────────────────────────────────────────────
    with st.expander("All 18 layers at selected step", expanded=False):
        def _load_layer(l):
            if mem_denoising is not None:
                return mem_denoising.get(f"layer_{l}")
            return load_suffix_denoising(h5_path, l) if h5_path else None

        vmax_all = None
        if global_norm:
            vals = [
                _get_heatmap(_load_layer(l)["action_to_img_steps"][step_sel], cam_key, action_step).max()
                for l in range(NUM_LAYERS)
                if _load_layer(l) is not None
            ]
            vmax_all = max(vals) if vals else None

        COLS_PER_ROW = 6
        for row_start in range(0, NUM_LAYERS, COLS_PER_ROW):
            chunk = list(range(row_start, min(row_start + COLS_PER_ROW, NUM_LAYERS)))
            cols  = st.columns(len(chunk))
            for ci, l in enumerate(chunk):
                d = _load_layer(l)
                if d is None:
                    cols[ci].caption(f"L{l} —")
                    continue
                h16 = _get_heatmap(d["action_to_img_steps"][step_sel], cam_key, action_step)
                if vmax_all is not None:
                    h16 = h16 / (vmax_all + 1e-9)
                overlay = _blend_overlay(cam_img, h16, alpha=alpha)
                mark = "🔵 " if l in layers else ""
                cols[ci].image(overlay, caption=f"{mark}L{l}", use_container_width=True)


# ── Trajectory-level panel ────────────────────────────────────────────────────

def _render_trajectory_panel(
    frame_paths: list[str],
    frame_labels: list[str],
    layers: list[int],
) -> None:
    """Show group-mass evolution across episode frames for selected denoising step."""

    if not frame_paths:
        st.info("No frames available.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        traj_layer = st.selectbox("Layer", layers, key="dn_traj_layer")
    with col_b:
        traj_step_hint = st.radio(
            "Denoising step to compare",
            ["First (most noisy)", "Mid", "Last (clean)"],
            horizontal=True,
            key="dn_traj_step",
        )

    # Load trajectory data (cached — loader hashes the paths tuple)
    traj = load_suffix_denoising_trajectory(tuple(frame_paths), traj_layer)

    if traj is None:
        st.info(
            "No denoising data for these frames. "
            "Re-run `pipeline.py` to generate `/suffix_denoising`."
        )
        return

    n_steps = traj["n_steps"]
    masses  = traj["group_masses"]   # (F, n_steps, 4)
    n_frames = masses.shape[0]

    step_map = {
        "First (most noisy)": 0,
        "Mid":                n_steps // 2,
        "Last (clean)":       n_steps - 1,
    }
    chosen_step = step_map[traj_step_hint]

    x_labels = frame_labels[:n_frames]

    # ── Line chart: 4 groups over episode frames ──────────────────────────────
    st.markdown(
        f"**Attention mass at denoising step {chosen_step} — layer {traj_layer}**  "
        f"(x = episode frame, y = attention mass)"
    )
    fig = go.Figure()
    for gi, g in enumerate(GROUPS):
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=masses[:, chosen_step, gi].tolist(),
            name=GROUP_LABEL[g],
            mode="lines+markers",
            line=dict(color=GROUP_COLOR[g], width=2),
            marker=dict(size=5),
            hovertemplate=f"%{{x}}<br>{GROUP_LABEL[g]}: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        height=350, hovermode="x unified",
        xaxis=dict(title="Episode frame", tickangle=-45, tickfont=dict(size=9)),
        yaxis_title="Attention mass",
        margin=dict(l=40, r=20, t=20, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap: frames × denoising steps for one group ───────────────────────
    st.markdown("**Wrist-camera attention mass — all frames × all denoising steps**")
    wrist_idx = GROUPS.index("wrist_img")
    wrist_mat = masses[:, :, wrist_idx]   # (F, n_steps)

    fig2 = go.Figure(go.Heatmap(
        z=wrist_mat,
        x=[f"step {s}" for s in range(n_steps)],
        y=x_labels,
        colorscale="Hot",
        hovertemplate="frame=%{y}<br>step=%{x}<br>wrist_mass=%{z:.4f}<extra></extra>",
    ))
    fig2.add_vline(x=chosen_step, line_dash="dash", line_color="cyan", opacity=0.7)
    fig2.update_layout(
        height=max(250, 18 * n_frames),
        xaxis_title="Denoising step (0=most noisy)",
        margin=dict(l=120, r=20, t=20, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── Public entry point ─────────────────────────────────────────────────────────

def render(
    available_layers: list[int],
    h5_path: str | None = None,
    mem_images: dict | None = None,
    mem_denoising: dict | None = None,
    all_frame_paths: list[str] | None = None,
    frame_labels: list[str] | None = None,
) -> None:
    """Render the Denoising tab.

    Accepts two data-source modes:
      HDF5 (Results/Offline): pass h5_path — loads data lazily via st.cache_data.
      In-memory (Online):     pass mem_images + mem_denoising from inference.run_inference().

    Args:
        available_layers: Layers to show (from sidebar selection or all 18).
        h5_path:          HDF5 file for the current frame (Results/Offline mode).
        mem_images:       {"exterior": ndarray, "wrist": ndarray} (Online mode).
        mem_denoising:    suffix_denoising dict from run_inference() (Online mode).
        all_frame_paths:  All frame HDF5 paths for this episode (trajectory panel).
        frame_labels:     Human-readable x-axis labels for the trajectory panel.
    """
    layers = available_layers if available_layers else list(range(NUM_LAYERS))

    tab_frame, tab_traj = st.tabs(["🔬 Frame level", "📊 Trajectory level"])

    with tab_frame:
        images = mem_images if mem_images is not None else (load_images(h5_path) if h5_path else {})
        _render_frame_panel(layers, images, h5_path=h5_path, mem_denoising=mem_denoising)

    with tab_traj:
        if all_frame_paths:
            _render_trajectory_panel(
                all_frame_paths,
                frame_labels or [str(i) for i in range(len(all_frame_paths))],
                layers,
            )
        else:
            st.info(
                "Trajectory panel is available in Results (Benchmark) mode "
                "after loading a full episode. Online mode shows frame-level data only."
            )
