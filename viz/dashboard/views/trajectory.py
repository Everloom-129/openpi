"""Trajectory view: multi-frame attention grid for one episode.

Layout:
  Controls  — camera, head aggregation, layer multiselect, frame stride
  Token pills — shared selector across all frames
  Frame strip — small thumbnails of selected frames
  Main grid  — rows = layers, cols = selected frames
  CF section — expander with per-slug grids + Δ vs main
"""
from __future__ import annotations

import io
import os

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from viz.dashboard import loader_results as _rl
from viz.dashboard.loader import (
    list_layers_in_h5,
    load_action_to_img,
    load_gt_action,
    load_images,
    load_meta,
    load_pred_action,
    load_text_to_img,
)
from viz.dashboard.views.grid_heatmap import (
    _ALL_AGG_NAMES,
    _SIMPLE_AGGS,
    _attn_to_heatmap,
    _fig_to_bytes,
    _make_count_fn,
    _make_topk_fn,
    _overlay,
)

DEFAULT_LAYERS = [1, 4, 5, 7, 10]
THUMB_SIZE = 72   # px for thumbnail strip
MAX_FRAMES_WARN = 16  # warn if more frames selected
CHUNK_SIZE = 8   # frames per row in thumbnail strip and trajectory figures


# ── Figure builder ─────────────────────────────────────────────────────────────

def _build_trajectory_figure(
    frames: list[int],
    layers: list[int],
    t2i_data: dict[tuple[int, int], np.ndarray | None],  # (frame, layer) → (8, n_text, 512)
    imgs: dict[int, np.ndarray | None],                   # frame → (H, W, 3) uint8
    tok_idx: int | None,
    camera: str,
    agg_fn,
    row_label_prefix: str = "",
) -> plt.Figure:
    """Build a (layers × frames) matplotlib figure.

    rows = selected layers, cols = selected frames.
    agg_fn is applied to (n_heads, 512) → (512,).
    tok_idx=None means aggregate (mean) over all text tokens.
    """
    n_rows = len(layers)
    n_cols = len(frames)
    cell_w = 1.4
    cell_h = 1.4
    label_frac = 0.45 / (n_cols * cell_w + 0.45)  # fraction for row-label column

    fig = plt.figure(figsize=(n_cols * cell_w + 0.45, n_rows * cell_h + 0.3), dpi=100)
    fig.patch.set_facecolor("#0e1117")
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        wspace=0.04,
        hspace=0.18,
        left=label_frac,
        right=0.99,
        top=0.92,
        bottom=0.02,
    )

    for row_i, layer in enumerate(layers):
        for col_i, frame in enumerate(frames):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            t2i = t2i_data.get((frame, layer))
            img = imgs.get(frame)
            if img is None:
                img = np.full((224, 224, 3), 30, dtype=np.uint8)

            if t2i is None:
                ax.set_facecolor("#1a1a2e")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=7)
            else:
                # tok_idx=None → mean over all text tokens before head aggregation
                heads_512 = t2i.mean(axis=1) if tok_idx is None else t2i[:, tok_idx, :]
                attn_512 = agg_fn(heads_512)  # (n_heads, 512) → (512,)
                hmap = _attn_to_heatmap(attn_512, camera)
                cell_img = _overlay(img, hmap)
                ax.imshow(cell_img, aspect="equal")

            # Column headers (top row only)
            if row_i == 0:
                ax.set_title(f"F{frame}", color="white", fontsize=8, pad=3)

            # Row labels (first column only)
            if col_i == 0:
                prefix = f"{row_label_prefix}\n" if row_label_prefix else ""
                ax.set_ylabel(
                    f"{prefix}L{layer}",
                    color="white", fontsize=8,
                    rotation=0, labelpad=30, va="center",
                )

    return fig


# ── Action temporal analysis ───────────────────────────────────────────────────

def _action_attn_512(a2i: np.ndarray, head_agg: str) -> np.ndarray:
    """Aggregate action→image attention to a single (512,) vector.

    a2i: (n_heads, 8_steps, 512_patches) from load_action_to_img().
    Returns mean over action steps after head aggregation.
    """
    if head_agg == "Max":
        by_head = a2i.max(axis=0)   # (8_steps, 512)
    else:
        by_head = a2i.mean(axis=0)  # (8_steps, 512)
    return by_head.mean(axis=0)     # (512,)


def _concentration(attn_256: np.ndarray) -> float:
    """Focus concentration: 1 − normalized entropy. Range [0, 1]."""
    p = attn_256 / (attn_256.sum() + 1e-10)
    h = -float(np.sum(p * np.log(p + 1e-10)))
    return float(1.0 - h / np.log(256))


def _render_action_temporal(
    root: str,
    outcome: str,
    date: str,
    episode: str,
    selected_frames: list[int],
    all_layers: list[int],
) -> None:
    """Render the Action Token Temporal Analysis expander."""
    with st.expander("Action Token Attention — Temporal Analysis", expanded=True):
        st.caption(
            "Where do **action tokens** attend across the episode? "
            "Each point = one episode frame; attention is averaged over all 8 decode steps. "
            "Hypothesis: early frames attend to the goal, mid-episode to the object, "
            "late frames to both."
        )

        # ── Controls ──────────────────────────────────────────────────────────
        act_c1, act_c2 = st.columns([3, 2])
        with act_c1:
            act_layer = st.selectbox(
                "Layer", options=all_layers,
                index=min(4, len(all_layers) - 1),
                key="traj_act_layer",
            )
        with act_c2:
            act_head_agg = st.radio(
                "Head aggregation", ["Max", "Mean"],
                horizontal=True, key="traj_act_head_agg",
            )

        # ── Load data for every selected frame ────────────────────────────────
        wrist_conc: list[float | None] = []
        ext_conc: list[float | None] = []
        balance: list[float | None] = []
        act_t2i: dict[tuple[int, int], np.ndarray | None] = {}  # for heatmap strip

        wrist_imgs: dict[int, np.ndarray | None] = {}
        with st.spinner("Loading action attention across frames…"):
            for frame in selected_frames:
                path = _rl.h5_path_results(root, outcome, date, episode, frame)
                wrist_imgs[frame] = load_images(path).get("wrist")
                a2i = load_action_to_img(path, act_layer)   # (n_heads, 8, 512) or None
                if a2i is None:
                    wrist_conc.append(None)
                    ext_conc.append(None)
                    balance.append(None)
                    act_t2i[(frame, act_layer)] = None
                    continue

                attn_512 = _action_attn_512(a2i, act_head_agg)
                wrist_conc.append(_concentration(attn_512[256:512]))
                ext_conc.append(_concentration(attn_512[0:256]))
                wrist_sum = float(attn_512[256:512].sum())
                ext_sum   = float(attn_512[0:256].sum())
                balance.append(ext_sum / (wrist_sum + 1e-8))

                # Fake single-token t2i shape (n_heads, 1, 512) so
                # _build_trajectory_figure can render the wrist heatmap strip
                act_t2i[(frame, act_layer)] = a2i.mean(axis=1)[:, np.newaxis, :]  # (n_heads, 1, 512)

        # ── Metrics chart ─────────────────────────────────────────────────────
        valid = [(f, w, e, b) for f, w, e, b in
                 zip(selected_frames, wrist_conc, ext_conc, balance)
                 if w is not None]
        if not valid:
            st.warning("No full-matrix data available for this layer.")
            return

        frames_v, wrist_v, ext_v, balance_v = zip(*valid)

        fig_metrics = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Focus Concentration (higher = more localised)",
                            "Ext / Wrist Attention Balance (>1 = scene-dominant)"),
            vertical_spacing=0.12,
        )
        fig_metrics.add_trace(
            go.Scatter(x=list(frames_v), y=list(wrist_v), mode="lines+markers",
                       name="Wrist conc.", line=dict(color="#f0a500")),
            row=1, col=1,
        )
        fig_metrics.add_trace(
            go.Scatter(x=list(frames_v), y=list(ext_v), mode="lines+markers",
                       name="Ext conc.", line=dict(color="#4c9be8"), opacity=0.6),
            row=1, col=1,
        )
        fig_metrics.add_trace(
            go.Scatter(x=list(frames_v), y=list(balance_v), mode="lines+markers",
                       name="Ext/Wrist ratio", line=dict(color="#9b59b6")),
            row=2, col=1,
        )
        fig_metrics.add_hline(y=1.0, line_dash="dash", line_color="gray",
                              annotation_text="balanced", row=2, col=1)
        fig_metrics.update_layout(
            height=420,
            margin=dict(l=50, r=20, t=50, b=40),
            legend=dict(orientation="h", y=1.06),
            xaxis2=dict(title="Episode frame"),
        )
        st.plotly_chart(fig_metrics, use_container_width=True, key="traj_act_metrics")

        # ── Wrist heatmap strip (action→wrist attention) ──────────────────────
        st.markdown("**Action attention → Wrist camera** (mean over 8 decode steps)")

        # agg_fn: max or mean over heads dim (already collapsed into single-token t2i)
        _agg = (lambda x: x.max(axis=0)) if act_head_agg == "Max" else (lambda x: x.mean(axis=0))

        for chunk_idx, chunk_start in enumerate(range(0, len(selected_frames), CHUNK_SIZE)):
            chunk = selected_frames[chunk_start : chunk_start + CHUNK_SIZE]
            fig_strip = _build_trajectory_figure(
                frames=chunk,
                layers=[act_layer],
                t2i_data=act_t2i,
                imgs=wrist_imgs,
                tok_idx=0,
                camera="wrist",
                agg_fn=_agg,
                row_label_prefix="act",
            )
            caption = f"Action→Wrist — layer {act_layer} — {act_head_agg}" if chunk_idx == 0 else None
            st.image(_fig_to_bytes(fig_strip), caption=caption, use_container_width=True)


# ── Action benchmark (pred vs GT) ─────────────────────────────────────────────

_ACTION_DIM_LABELS = [f"j{i}" for i in range(7)] + ["grip"]


def _render_action_benchmark(
    root: str,
    outcome: str,
    date: str,
    episode: str,
    selected_frames: list[int],
) -> None:
    """Render Predicted vs Ground-Truth action comparison across frames."""
    with st.expander("Action Benchmark — Pred vs GT", expanded=True):
        st.caption(
            "Pi0.5 predicted actions vs ground-truth trajectory actions. "
            "Useful for checking inference determinism and measuring action error."
        )

        # ── Load data ─────────────────────────────────────────────────────────
        pred_data: dict[int, np.ndarray | None] = {}
        gt_data:   dict[int, np.ndarray | None] = {}
        with st.spinner("Loading pred/GT actions…"):
            for frame in selected_frames:
                path = _rl.h5_path_results(root, outcome, date, episode, frame)
                pred_data[frame] = load_pred_action(path)
                gt_data[frame]   = load_gt_action(path)

        has_pred = any(v is not None for v in pred_data.values())
        has_gt   = any(v is not None for v in gt_data.values())

        if not has_pred and not has_gt:
            st.info("No pred/GT action data found — re-run the pipeline to generate it.")
            return

        # ── Per-frame mean L2 error ────────────────────────────────────────
        if has_pred and has_gt:
            frames_v, l2_mean, l2_by_step = [], [], []
            for frame in selected_frames:
                p = pred_data[frame]
                g = gt_data[frame]
                if p is None or g is None:
                    continue
                p = p[: g.shape[0]]  # clip pred to GT length (pi0.5=15, GT=8)
                # Mask NaN GT rows (near episode end)
                valid = ~np.isnan(g).any(axis=1)
                if not valid.any():
                    continue
                diff = (p[valid] - g[valid]) ** 2          # (n_valid, 8)
                l2_step = np.sqrt(diff.mean(axis=1))       # (n_valid,) — one per action step
                frames_v.append(frame)
                l2_mean.append(float(l2_step.mean()))
                l2_by_step.append(l2_step)                 # (n_valid,)

            if frames_v:
                # Top chart: mean L2 over episode frames
                fig_l2 = go.Figure()
                fig_l2.add_trace(go.Scatter(
                    x=frames_v, y=l2_mean, mode="lines+markers",
                    name="Mean L2 (all steps)", line=dict(color="#4c9be8"),
                ))
                # Individual action steps as faint lines
                n_steps = max(len(s) for s in l2_by_step)
                step_colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_steps))
                for step_i in range(n_steps):
                    y_step = [
                        float(s[step_i]) if step_i < len(s) else None
                        for s in l2_by_step
                    ]
                    r, g_c, b, _ = step_colors[step_i]
                    fig_l2.add_trace(go.Scatter(
                        x=frames_v, y=y_step, mode="lines",
                        name=f"step {step_i}",
                        line=dict(color=f"rgba({int(r*255)},{int(g_c*255)},{int(b*255)},0.45)", width=1),
                        showlegend=(step_i < 4),
                    ))
                fig_l2.update_layout(
                    title="L2 error per episode frame",
                    xaxis_title="Episode frame",
                    yaxis_title="L2 error",
                    height=300,
                    margin=dict(l=50, r=20, t=40, b=40),
                    legend=dict(orientation="h", y=1.12),
                )
                st.plotly_chart(fig_l2, use_container_width=True, key="bench_l2")

                # Per-dim error heatmap: rows = action dims, cols = frames
                err_matrix = np.full((8, len(frames_v)), np.nan)
                for fi, frame in enumerate(frames_v):
                    p = pred_data[frame]
                    g_arr = gt_data[frame]
                    if p is None or g_arr is None:
                        continue
                    p = p[: g_arr.shape[0]]  # clip pred to GT length
                    valid = ~np.isnan(g_arr).any(axis=1)
                    if valid.any():
                        err_matrix[:, fi] = np.sqrt(
                            ((p[valid] - g_arr[valid]) ** 2).mean(axis=0)
                        )

                fig_hm = go.Figure(go.Heatmap(
                    z=err_matrix,
                    x=[str(f) for f in frames_v],
                    y=_ACTION_DIM_LABELS,
                    colorscale="Plasma",
                    colorbar=dict(title="L2 err"),
                ))
                fig_hm.update_layout(
                    title="Per-dimension L2 error (rows = action dims, cols = frames)",
                    xaxis_title="Episode frame",
                    yaxis_title="Action dim",
                    height=280,
                    margin=dict(l=60, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_hm, use_container_width=True, key="bench_heatmap")

        # ── Single-frame detail: pred vs GT per dim ────────────────────────
        valid_frames = [f for f in selected_frames
                        if pred_data.get(f) is not None or gt_data.get(f) is not None]
        if not valid_frames:
            return

        detail_frame = st.select_slider(
            "Frame for detail view",
            options=valid_frames,
            value=valid_frames[len(valid_frames) // 2],
            key="bench_detail_frame",
        )
        p_det = pred_data.get(detail_frame)
        g_det = gt_data.get(detail_frame)

        n_steps = (p_det.shape[0] if p_det is not None else
                   g_det.shape[0] if g_det is not None else 8)

        fig_det = make_subplots(
            rows=2, cols=4,
            subplot_titles=_ACTION_DIM_LABELS,
            shared_xaxes=True,
            vertical_spacing=0.18,
            horizontal_spacing=0.08,
        )
        for dim_i, label in enumerate(_ACTION_DIM_LABELS):
            row, col = dim_i // 4 + 1, dim_i % 4 + 1
            steps = list(range(n_steps))
            if g_det is not None:
                gt_vals = g_det[:, dim_i].tolist()
                fig_det.add_trace(go.Scatter(
                    x=steps, y=gt_vals, mode="lines+markers",
                    name="GT" if dim_i == 0 else None,
                    showlegend=(dim_i == 0),
                    line=dict(color="#f0a500"),
                ), row=row, col=col)
            if p_det is not None:
                pred_vals = p_det[:, dim_i].tolist()
                fig_det.add_trace(go.Scatter(
                    x=steps, y=pred_vals, mode="lines+markers",
                    name="Pred" if dim_i == 0 else None,
                    showlegend=(dim_i == 0),
                    line=dict(color="#4c9be8", dash="dash"),
                ), row=row, col=col)

        fig_det.update_layout(
            title=f"Frame {detail_frame} — Pred (blue dashed) vs GT (orange) per action dim",
            height=400,
            margin=dict(l=40, r=20, t=60, b=40),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_det, use_container_width=True, key="bench_detail")


# ── Main render ────────────────────────────────────────────────────────────────

def render(
    root: str,
    outcome: str,
    date: str,
    episode: str,
    available_frames: list[int],
    cf_slugs: list[str],
) -> None:
    """Render the trajectory tab."""

    if not available_frames:
        st.warning("No frames found for this episode.")
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 3, 1])

    with ctrl_c1:
        camera = st.radio("Camera", ["Exterior", "Wrist"], horizontal=True, key="traj_cam")
        camera_key = "exterior" if camera == "Exterior" else "wrist"

        # Exclude "All heads" — trajectory needs a single map per cell
        agg_names = [n for n in _ALL_AGG_NAMES if n != "All heads"]
        agg = st.selectbox("Head aggregation", agg_names, index=agg_names.index("Max"), key="traj_agg")
        if agg == "Top-K Focused":
            topk_k = st.slider("K heads", 1, 8, 4, key="traj_topk")
        elif agg == "Count Above Threshold":
            count_pct = st.slider("Top %", 1, 50, 10, key="traj_pct")

    with ctrl_c2:
        first_h5 = _rl.h5_path_results(root, outcome, date, episode, available_frames[0])
        all_layers = list_layers_in_h5(first_h5)
        default_layers = [l for l in DEFAULT_LAYERS if l in all_layers] or all_layers[:4]
        selected_layers = st.multiselect(
            "Layers", options=all_layers, default=default_layers, key="traj_layers"
        )

    with ctrl_c3:
        stride = st.select_slider(
            "Stride", options=[1, 2, 4, 8, 16], value=8, key="traj_stride"
        )

    if not selected_layers:
        st.info("Select at least one layer.")
        return

    # ── Frame selection ───────────────────────────────────────────────────────
    default_sel = available_frames[::stride] or available_frames[:1]

    # Quick-select buttons write directly to the multiselect's session state key.
    # This is the only reliable way to programmatically update a multiselect in Streamlit,
    # since `default` is ignored once the widget key already exists in session state.
    ms_key = f"traj_frames_{stride}"
    btn_c1, btn_c2, btn_c3, _ = st.columns([1, 1, 2, 8])
    with btn_c1:
        if st.button("All", key="traj_btn_all"):
            st.session_state[ms_key] = available_frames
    with btn_c2:
        if st.button("None", key="traj_btn_none"):
            st.session_state[ms_key] = []
    with btn_c3:
        if st.button(f"Every {stride}", key="traj_btn_stride"):
            st.session_state[ms_key] = available_frames[::stride]

    selected_frames: list[int] = st.multiselect(
        "Select frames to compare:",
        options=available_frames,
        default=default_sel,
        key=ms_key,
    )

    if not selected_frames:
        st.info("Select at least one frame.")
        return

    selected_frames = sorted(selected_frames)

    if len(selected_frames) > MAX_FRAMES_WARN:
        st.warning(f"{len(selected_frames)} frames selected — rendering may be slow. Consider reducing.")

    # ── Token selector ────────────────────────────────────────────────────────
    meta = load_meta(_rl.h5_path_results(root, outcome, date, episode, selected_frames[0]))
    n_real: int = meta.get("n_real_tokens", len(meta.get("token_texts", [])))
    real_texts = meta.get("token_texts", [])[:n_real]

    if not real_texts:
        st.warning("No token texts found.")
        return

    if meta.get("instruction"):
        st.caption(f"**Instruction:** {meta['instruction']}")

    tok_mode = st.radio(
        "Token mode",
        ["Single token", "All text tokens (mean)"],
        horizontal=True,
        key="traj_tok_mode",
    )

    tok_idx: int | None
    selected_label: str

    if tok_mode == "All text tokens (mean)":
        tok_idx = None
        selected_label = "all tokens (mean)"
    else:
        # Deduplicate labels (same logic as grid_heatmap)
        raw_labels = [t.replace("▁", " ").strip() or f"[{i}]" for i, t in enumerate(real_texts)]
        seen: dict[str, int] = {}
        token_labels: list[str] = []
        for lbl in raw_labels:
            if raw_labels.count(lbl) > 1:
                seen[lbl] = seen.get(lbl, 0) + 1
                token_labels.append(f"{lbl}#{seen[lbl]}")
            else:
                token_labels.append(lbl)

        selected_label = st.pills(
            "Token (shared across all frames):",
            options=token_labels,
            default=token_labels[min(3, n_real - 1)],
            selection_mode="single",
            key="traj_tok",
        )
        if selected_label is None:
            st.info("Click a token to visualize its attention.")
            return
        tok_idx = token_labels.index(selected_label)

    # ── Frame thumbnail strip ─────────────────────────────────────────────────
    with st.expander("Frame thumbnails", expanded=True):
        for chunk_start in range(0, len(selected_frames), CHUNK_SIZE):
            chunk = selected_frames[chunk_start : chunk_start + CHUNK_SIZE]
            thumb_cols = st.columns(len(chunk))
            for i, frame in enumerate(chunk):
                path = _rl.h5_path_results(root, outcome, date, episode, frame)
                imgs = load_images(path)
                thumb = imgs.get(camera_key)
                if thumb is not None:
                    thumb_small = cv2.resize(thumb, (THUMB_SIZE, THUMB_SIZE))
                else:
                    thumb_small = np.full((THUMB_SIZE, THUMB_SIZE, 3), 40, dtype=np.uint8)
                with thumb_cols[i]:
                    st.image(thumb_small, caption=f"F{frame}", use_container_width=True)

    # ── Resolve aggregation function ──────────────────────────────────────────
    if agg in _SIMPLE_AGGS:
        agg_fn = _SIMPLE_AGGS[agg]
    elif agg == "Top-K Focused":
        agg_fn = _make_topk_fn(topk_k)
    else:
        agg_fn = _make_count_fn(count_pct)
    if agg_fn is None:  # "All heads" was filtered out, but guard anyway
        agg_fn = lambda x: x.mean(axis=0)

    # ── Load attention data ───────────────────────────────────────────────────
    with st.spinner(f"Loading {len(selected_frames)} × {len(selected_layers)} attention slices…"):
        t2i_data: dict[tuple[int, int], np.ndarray | None] = {}
        imgs_by_frame: dict[int, np.ndarray | None] = {}
        for frame in selected_frames:
            path = _rl.h5_path_results(root, outcome, date, episode, frame)
            imgs_by_frame[frame] = load_images(path).get(camera_key)
            for layer in selected_layers:
                t2i_data[(frame, layer)] = load_text_to_img(path, layer)

    # ── Main trajectory figure ────────────────────────────────────────────────
    with st.spinner("Rendering trajectory grid…"):
        for chunk_idx, chunk_start in enumerate(range(0, len(selected_frames), CHUNK_SIZE)):
            chunk = selected_frames[chunk_start : chunk_start + CHUNK_SIZE]
            fig = _build_trajectory_figure(
                frames=chunk,
                layers=selected_layers,
                t2i_data=t2i_data,
                imgs=imgs_by_frame,
                tok_idx=tok_idx,
                camera=camera_key,
                agg_fn=agg_fn,
            )
            caption = f'Trajectory — "{selected_label}" — {camera} — {agg}' if chunk_idx == 0 else None
            st.image(_fig_to_bytes(fig), caption=caption, use_container_width=True)

    # ── Action Token Temporal Analysis ───────────────────────────────────────
    _render_action_temporal(
        root=root, outcome=outcome, date=date, episode=episode,
        selected_frames=selected_frames,
        all_layers=all_layers,
    )

    # ── Action Benchmark (pred vs GT) ─────────────────────────────────────────
    _render_action_benchmark(
        root=root, outcome=outcome, date=date, episode=episode,
        selected_frames=selected_frames,
    )

    # ── Counterfactual section ────────────────────────────────────────────────
    if not cf_slugs:
        return

    with st.expander(f"Counterfactual comparison ({', '.join(cf_slugs)})", expanded=False):
        st.caption(
            "Same token & frames, different prompt. "
            "Pick a layer, then see each variant's attention and Δ vs main."
        )

        cf_layer = st.selectbox(
            "Layer for CF comparison",
            options=selected_layers,
            index=min(1, len(selected_layers) - 1),
            key="traj_cf_layer",
        )

        # Load main attention for the CF layer (reuse already-loaded if available)
        main_t2i_cf: dict[tuple[int, int], np.ndarray | None] = {
            (f, cf_layer): t2i_data.get((f, cf_layer))
            for f in selected_frames
        }

        for slug in cf_slugs:
            st.markdown(f"**`{slug}`**")
            cf_t2i: dict[tuple[int, int], np.ndarray | None] = {}
            cf_imgs: dict[int, np.ndarray | None] = {}
            for frame in selected_frames:
                cf_path = _rl.h5_path_results(root, outcome, date, episode, frame, slug=slug)
                if os.path.exists(cf_path):
                    cf_imgs[frame] = load_images(cf_path).get(camera_key)
                    cf_t2i[(frame, cf_layer)] = load_text_to_img(cf_path, cf_layer)
                else:
                    # File missing for this frame — use blank
                    cf_imgs[frame] = imgs_by_frame.get(frame)
                    cf_t2i[(frame, cf_layer)] = None

            for chunk_start in range(0, len(selected_frames), CHUNK_SIZE):
                chunk = selected_frames[chunk_start : chunk_start + CHUNK_SIZE]
                fig_cf = _build_trajectory_figure(
                    frames=chunk,
                    layers=[cf_layer],
                    t2i_data=cf_t2i,
                    imgs=cf_imgs,
                    tok_idx=tok_idx,
                    camera=camera_key,
                    agg_fn=agg_fn,
                    row_label_prefix=slug,
                )
                st.image(_fig_to_bytes(fig_cf), use_container_width=True)

        # Δ (main − first slug) heatmap
        st.markdown(f"**Δ main − `{cf_slugs[0]}`**")
        delta_t2i: dict[tuple[int, int], np.ndarray | None] = {}
        for frame in selected_frames:
            main_arr = main_t2i_cf.get((frame, cf_layer))
            cf0_path = _rl.h5_path_results(root, outcome, date, episode, frame, slug=cf_slugs[0])
            cf0_arr = load_text_to_img(cf0_path, cf_layer) if os.path.exists(cf0_path) else None
            if main_arr is not None and cf0_arr is not None:
                # Align n_text dim (may differ if prompts differ in length)
                n = min(main_arr.shape[1], cf0_arr.shape[1])
                delta_t2i[(frame, cf_layer)] = main_arr[:, :n, :] - cf0_arr[:, :n, :]
            else:
                delta_t2i[(frame, cf_layer)] = None

        # For delta, clamp tok_idx to available tokens (None = all tokens, pass through)
        if tok_idx is None:
            delta_tok = None
        else:
            delta_tok = min(tok_idx, min(
                (arr.shape[1] - 1 for arr in delta_t2i.values() if arr is not None),
                default=tok_idx,
            ))

        for chunk_idx, chunk_start in enumerate(range(0, len(selected_frames), CHUNK_SIZE)):
            chunk = selected_frames[chunk_start : chunk_start + CHUNK_SIZE]
            fig_delta = _build_trajectory_figure(
                frames=chunk,
                layers=[cf_layer],
                t2i_data=delta_t2i,
                imgs=imgs_by_frame,
                tok_idx=delta_tok,
                camera=camera_key,
                agg_fn=agg_fn,
                row_label_prefix="Δ",
            )
            caption = "Red = more attention in main, Blue = less" if chunk_idx == 0 else None
            st.image(_fig_to_bytes(fig_delta), caption=caption, use_container_width=True)
