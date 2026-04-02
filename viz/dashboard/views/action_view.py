"""Tab 3: Action token attention (suffix).

Shows where each action step attends: image patches, text tokens,
and other action steps (temporal coupling).

Note: this tab only shows data when joint/suffix attention is available.
For now, it provides a fallback view using prefix attention rows corresponding
to the last text token region.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

NUM_IMAGE_TOKENS = 256
TOTAL_IMAGE_TOKENS = 512
TEXT_START_IDX = 768
PATCH_GRID = 16

_ACTION_DIM_LABELS = [f"j{i}" for i in range(7)] + ["grip"]


def _upsample_16(attn_256: np.ndarray) -> np.ndarray:
    """Upsample 16×16 patch grid to 224×224 via kron."""
    grid = attn_256[:NUM_IMAGE_TOKENS].reshape(PATCH_GRID, PATCH_GRID).astype(np.float32)
    return np.kron(grid, np.ones((14, 14), dtype=np.float32))


def render(
    data: dict,
    available_layers: list[int],
) -> None:
    """Render Tab 3: Action View."""

    meta = data.get("meta", {})
    token_texts: list[str] = meta.get("token_texts", [])
    seq_len: int = meta.get("seq_len", 968)
    images = data.get("images", {})
    _ext = images.get("exterior")
    _wrist = images.get("wrist")
    ext_img = _ext if _ext is not None else np.full((224, 224, 3), 30, dtype=np.uint8)
    wrist_img = _wrist if _wrist is not None else np.full((224, 224, 3), 30, dtype=np.uint8)

    # Check for joint/ suffix data
    joint_data = data.get("joint")
    has_joint = joint_data is not None

    if not has_joint:
        st.info(
            "No joint/suffix attention data found. "
            "This tab requires running inference in online mode or storing joint attention maps.\n\n"
            "Showing text-row attention distribution as a proxy."
        )
        _render_text_distribution(data, available_layers, token_texts, ext_img, wrist_img)
        _render_pred_gt_benchmark(data)
        return

    # ── Joint attention rendering ─────────────────────────────────────────────
    col_l, col_h = st.columns([2, 3])
    with col_l:
        layer = st.select_slider("Layer", options=available_layers, key="av_layer")
    with col_h:
        head_opts = ["max", "mean"] + list(range(8))
        head_labels = ["Max", "Mean"] + [f"H{i}" for i in range(8)]
        head_sel_label = st.radio("Head", head_labels, index=0, horizontal=True, key="av_head")
        head_sel = head_opts[head_labels.index(head_sel_label)]

    # Load joint layer data
    j_layer = joint_data.get(f"layer_{layer}", {})
    a2i = j_layer.get("action_to_img")    # (8, 8, 512)
    a2t = j_layer.get("action_to_text")   # (8, 8, n_text)
    a2a = j_layer.get("action_to_action") # (8, 8, 8)

    if a2i is None:
        st.warning(f"No joint data for layer {layer}.")
        return

    def agg(arr):
        if head_sel == "max":
            return arr.max(axis=0)
        if head_sel == "mean":
            return arr.mean(axis=0)
        return arr[int(head_sel)]

    a2i_agg = agg(a2i)   # (8, 512)
    a2t_agg = agg(a2t) if a2t is not None else None   # (8, n_text)
    a2a_agg = agg(a2a) if a2a is not None else None   # (8, 8)

    # ── 8-step camera heatmap grid ────────────────────────────────────────────
    st.markdown("**Action step → camera attention (8 steps)**")
    fig = make_subplots(rows=2, cols=8, subplot_titles=[f"Step {i}" for i in range(8)] * 2,
                        vertical_spacing=0.05, horizontal_spacing=0.02)

    for step in range(8):
        attn_512 = a2i_agg[step]
        ext_h = _upsample_16(attn_512[:NUM_IMAGE_TOKENS])
        wrist_h = _upsample_16(attn_512[NUM_IMAGE_TOKENS:])

        for row_i, (img, hmap) in enumerate([(ext_img, ext_h), (wrist_img, wrist_h)], start=1):
            fig.add_trace(go.Image(z=img, name=f"step{step}"), row=row_i, col=step + 1)
            hmin, hmax = hmap.min(), hmap.max()
            hn = (hmap - hmin) / (hmax - hmin + 1e-8)
            fig.add_trace(
                go.Heatmap(z=hn, colorscale="Jet", opacity=0.45, showscale=False),
                row=row_i, col=step + 1,
            )

    fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0))
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].showticklabels = False
    st.plotly_chart(fig, use_container_width=True)

    # ── Temporal coupling (action→action) ────────────────────────────────────
    if a2a_agg is not None:
        st.markdown("**Temporal coupling (action→action)**")
        fig2 = go.Figure(
            go.Heatmap(
                z=a2a_agg,
                x=[f"Step {i}" for i in range(8)],
                y=[f"Step {i}" for i in range(8)],
                colorscale="Viridis",
                hovertemplate="from=%{y}<br>to=%{x}<br>weight=%{z:.4f}<extra></extra>",
            )
        )
        fig2.update_layout(
            title="Action-to-Action Coupling Matrix",
            height=350,
            margin=dict(l=60, r=20, t=40, b=60),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Language ratio bar ────────────────────────────────────────────────────
    if a2t_agg is not None:
        st.markdown("**Where does each action step draw its information from?**")
        img_frac = a2i_agg.sum(axis=-1)   # (8,)
        txt_frac = a2t_agg.sum(axis=-1)   # (8,)
        act_frac = a2a_agg.sum(axis=-1) if a2a_agg is not None else np.zeros(8)
        total = img_frac + txt_frac + act_frac + 1e-8
        img_pct = 100 * img_frac / total
        txt_pct = 100 * txt_frac / total
        act_pct = 100 * act_frac / total
        steps = [f"Step {i}" for i in range(8)]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Image", x=steps, y=img_pct, marker_color="steelblue"))
        fig3.add_trace(go.Bar(name="Text", x=steps, y=txt_pct, marker_color="coral"))
        fig3.add_trace(go.Bar(name="Action", x=steps, y=act_pct, marker_color="mediumseagreen"))
        fig3.update_layout(
            barmode="stack",
            title="Attention source breakdown per action step (%)",
            height=300,
            margin=dict(l=40, r=20, t=40, b=40),
            yaxis=dict(title="%"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    _render_pred_gt_benchmark(data)


def _render_pred_gt_benchmark(data: dict) -> None:
    """Render Predicted vs Ground-Truth action benchmark for a single frame."""
    pred = data.get("pred_action")   # (N, 8) float32 or None — N may exceed 8
    gt   = data.get("gt_action")     # (8, 8) float32, NaN-padded, or None

    if pred is None and gt is None:
        return

    # Model action chunks can be longer than OPEN_LOOP_HORIZON (e.g. pi0.5=15, pi0=10).
    # Clip both to 8 steps for display.
    if pred is not None:
        pred = pred[:8]
    if gt is not None:
        gt = gt[:8]

    with st.expander("Action Benchmark — Pred vs GT", expanded=True):
        st.caption(
            "Pi0.5 predicted actions vs ground-truth trajectory actions for this frame. "
            "Each subplot = one action dimension; "
            "x-axis = decode step (0–7)."
        )

        steps = list(range(8))

        # ── Per-step L2 bar chart ──────────────────────────────────────────
        if pred is not None and gt is not None:
            valid = ~np.isnan(gt).any(axis=1)   # mask padded rows near episode end
            if valid.any():
                diff = pred[valid] - gt[valid]                  # (n, 8)
                l2_per_step_valid = np.sqrt((diff ** 2).mean(axis=1))  # (n,)
                # pad back to 8 steps with NaN
                l2_per_step = np.full(8, np.nan)
                l2_per_step[np.where(valid)[0]] = l2_per_step_valid

                fig_l2 = go.Figure(go.Bar(
                    x=[f"Step {i}" for i in range(8)],
                    y=l2_per_step.tolist(),
                    marker_color=[
                        f"rgba(76,155,232,{0.4 + 0.6 * (v / (np.nanmax(l2_per_step) + 1e-8))})"
                        if not np.isnan(v) else "rgba(80,80,80,0.3)"
                        for v in l2_per_step
                    ],
                    hovertemplate="Step %{x}: L2=%{y:.4f}<extra></extra>",
                ))
                fig_l2.update_layout(
                    title="L2 error per action step (mean over dims)",
                    height=220,
                    margin=dict(l=40, r=20, t=40, b=30),
                    yaxis_title="L2 error",
                )
                st.plotly_chart(fig_l2, use_container_width=True, key="av_bench_l2")

        # ── Per-dim subplot: pred vs GT ────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=_ACTION_DIM_LABELS,
            shared_xaxes=True,
            vertical_spacing=0.22,
            horizontal_spacing=0.08,
        )
        for dim_i, label in enumerate(_ACTION_DIM_LABELS):
            row, col = dim_i // 4 + 1, dim_i % 4 + 1
            if gt is not None:
                gt_vals = [float(v) if not np.isnan(v) else None for v in gt[:, dim_i]]
                fig.add_trace(go.Scatter(
                    x=steps, y=gt_vals, mode="lines+markers",
                    name="GT" if dim_i == 0 else None,
                    showlegend=(dim_i == 0),
                    line=dict(color="#f0a500"),
                    marker=dict(size=5),
                ), row=row, col=col)
            if pred is not None:
                fig.add_trace(go.Scatter(
                    x=steps, y=pred[:, dim_i].tolist(), mode="lines+markers",
                    name="Pred" if dim_i == 0 else None,
                    showlegend=(dim_i == 0),
                    line=dict(color="#4c9be8", dash="dash"),
                    marker=dict(size=5),
                ), row=row, col=col)

        fig.update_layout(
            title="Pred (blue dashed) vs GT (orange) — all action dims",
            height=380,
            margin=dict(l=40, r=20, t=60, b=30),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True, key="av_bench_detail")


def _render_text_distribution(
    data: dict,
    available_layers: list[int],
    token_texts: list[str],
    ext_img: np.ndarray,
    wrist_img: np.ndarray,
) -> None:
    """Fallback: show per-text-token attention distribution from prefix data."""
    if not available_layers:
        return

    col_l, col_h = st.columns([2, 3])
    with col_l:
        layer = st.select_slider("Layer", options=available_layers, key="av_layer_fallback")
    with col_h:
        head_labels = ["Max", "Mean"] + [f"H{i}" for i in range(8)]
        head_opts: list[str | int] = ["max", "mean"] + list(range(8))
        hl = st.radio("Head", head_labels, index=0, horizontal=True, key="av_head_fallback")
        head_sel = head_opts[head_labels.index(hl)]

    loader_fn = data.get("_load_t2i")
    t2i = None
    if loader_fn is not None:
        t2i = loader_fn(layer)
    else:
        t2i = data.get("prefix", {}).get(f"layer_{layer}", {}).get("text_to_img")

    if t2i is None:
        st.warning("No text_to_img data found.")
        return

    # Aggregate over heads
    if head_sel == "max":
        agg = t2i.max(axis=0)   # (n_text, 512)
    elif head_sel == "mean":
        agg = t2i.mean(axis=0)
    else:
        agg = t2i[int(head_sel)]

    # Per-token total attention to image patches
    img_attn_per_tok = agg.sum(axis=-1)   # (n_text,)
    n_text = agg.shape[0]
    tok_labels = token_texts[:n_text] if len(token_texts) >= n_text else token_texts + [f"tok_{i}" for i in range(len(token_texts), n_text)]
    tok_labels = [t.replace("▁", " ").strip() or f"[{i}]" for i, t in enumerate(tok_labels)]

    fig = go.Figure(
        go.Bar(
            x=tok_labels,
            y=img_attn_per_tok.tolist(),
            marker_color="steelblue",
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Total attention to image patches per text token — Layer {layer}",
        height=300,
        margin=dict(l=40, r=20, t=40, b=80),
        xaxis=dict(tickfont=dict(size=8), tickangle=-45),
        yaxis=dict(title="Sum of attention weights"),
    )
    st.plotly_chart(fig, use_container_width=True)
