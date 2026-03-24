"""Side-by-side comparison of two checkpoint attention visualizations.

Used by the "Compare (Online)" mode in app.py.  Both models run on the same
input (episode / frame / instruction); this view renders their attention maps
with shared controls so differences are directly visible.
"""
from __future__ import annotations

import io

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ── Constants (mirrors loader.py) ─────────────────────────────────────────────
NUM_IMAGE_TOKENS = 256
PATCH_GRID = 16
DEFAULT_LAYERS = [1, 4, 5, 7, 10]

# Aggregation options (subset of grid_heatmap.py for conciseness)
_AGGS = {
    "Average": lambda x: x.mean(axis=0),
    "Max":     lambda x: x.max(axis=0),
    "Median":  lambda x: np.median(x, axis=0),
    "Std Dev": lambda x: x.std(axis=0),
}


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _get_t2i(data: dict, layer: int) -> np.ndarray | None:
    """Return text_to_img (8, n_text, 512) for a layer, from any data format."""
    fn = data.get("_load_t2i")
    if fn is not None:
        return fn(layer)
    return data.get("prefix", {}).get(f"layer_{layer}", {}).get("text_to_img")


def _get_full(data: dict, layer: int) -> np.ndarray | None:
    """Return full attention matrix (8, seq, seq) for a layer."""
    fn = data.get("_load_full_all")
    if fn is not None:
        return fn(layer)
    return data.get("prefix", {}).get(f"layer_{layer}", {}).get("full")


def _attn_to_heatmap(attn_512: np.ndarray, camera: str) -> np.ndarray:
    start = 0 if camera == "exterior" else NUM_IMAGE_TOKENS
    patches = attn_512[start : start + NUM_IMAGE_TOKENS]
    grid = patches.reshape(PATCH_GRID, PATCH_GRID).astype(np.float32)
    return cv2.resize(grid, (112, 112), interpolation=cv2.INTER_LINEAR)


def _overlay(img: np.ndarray, hmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img_s = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    hmin, hmax = hmap.min(), hmap.max()
    hn = (hmap - hmin) / (hmax - hmin + 1e-8)
    hmap_u8 = (hn * 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_s, 1 - alpha, color, alpha, 0)


def _fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _build_grid_figure(
    t2i_by_layer: dict[int, np.ndarray],
    tok_idx: int,
    camera_img: np.ndarray,
    camera: str,
    layers: list[int],
    agg_fn,
) -> plt.Figure:
    """Compact aggregated grid: one cell per selected layer, max 5 per row."""
    cell = 1.4
    cols_per_row = 5
    n_cols = min(cols_per_row, len(layers))
    n_rows = (len(layers) + cols_per_row - 1) // cols_per_row
    fig = plt.figure(figsize=(n_cols * cell, n_rows * (cell + 0.3)), dpi=100)
    fig.patch.set_facecolor("#0e1117")
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        wspace=0.04, hspace=0.25,
        left=0.02, right=1.0, top=0.93, bottom=0.0,
    )
    for idx, layer in enumerate(layers):
        row_i, col_i = divmod(idx, cols_per_row)
        ax = fig.add_subplot(gs[row_i, col_i])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        t2i = t2i_by_layer.get(layer)
        if t2i is None:
            ax.set_facecolor("#1a1a2e")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=7)
        else:
            attn_512 = agg_fn(t2i[:, tok_idx, :])
            hmap = _attn_to_heatmap(attn_512, camera)
            ax.imshow(_overlay(camera_img, hmap), aspect="equal")
        ax.set_title(f"L{layer}", color="white", fontsize=8, pad=3)
    for idx in range(len(layers), n_rows * n_cols):
        row_i, col_i = divmod(idx, cols_per_row)
        fig.add_subplot(gs[row_i, col_i]).set_visible(False)
    return fig


def _build_heatmap_row(
    data: dict,
    layer: int,
    tok_idx: int,
    head_sel: int | str,
    camera: str,
) -> np.ndarray | None:
    """Return (224, 224) upsampled heatmap for a given camera."""
    t2i = _get_t2i(data, layer)
    if t2i is None:
        return None
    n_text = t2i.shape[1]
    tok_idx = min(tok_idx, n_text - 1)
    if isinstance(head_sel, int):
        agg = t2i[head_sel]
    elif head_sel == "max":
        agg = t2i.max(axis=0)
    else:
        agg = t2i.mean(axis=0)
    attn_512 = agg[tok_idx]
    start = 0 if camera == "exterior" else NUM_IMAGE_TOKENS
    patches = attn_512[start : start + NUM_IMAGE_TOKENS]
    grid = patches.reshape(PATCH_GRID, PATCH_GRID).astype(np.float32)
    return np.kron(grid, np.ones((14, 14), dtype=np.float32))


def _render_heatmap_pair(
    data: dict,
    label: str,
    layer: int,
    tok_idx: int,
    head_sel: int | str,
) -> None:
    """Render exterior + wrist heatmap overlays for one model inside its column."""
    import plotly.graph_objects as go

    images = data.get("images", {})
    _ext = images.get("exterior")
    _wrist = images.get("wrist")
    ext_img = _ext if _ext is not None else np.full((224, 224, 3), 30, dtype=np.uint8)
    wrist_img = _wrist if _wrist is not None else np.full((224, 224, 3), 30, dtype=np.uint8)

    for cam_key, cam_img, cam_label in [
        ("exterior", ext_img, "Exterior"),
        ("wrist",    wrist_img, "Wrist"),
    ]:
        hmap = _build_heatmap_row(data, layer, tok_idx, head_sel, cam_key)
        if hmap is None:
            st.warning(f"No data for layer {layer}.")
            continue
        hmin, hmax = hmap.min(), hmap.max()
        hmap_norm = (hmap - hmin) / (hmax - hmin + 1e-8)
        h, w = cam_img.shape[:2]
        fig = go.Figure()
        fig.add_trace(go.Image(z=cam_img))
        fig.add_trace(go.Heatmap(z=hmap_norm, colorscale="Jet", opacity=0.45, showscale=False))
        fig.update_layout(
            title=dict(text=cam_label, font_size=12),
            margin=dict(l=0, r=0, t=28, b=0),
            xaxis=dict(showticklabels=False, range=[0, w]),
            yaxis=dict(showticklabels=False, range=[h, 0]),
            height=240,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_attn_matrix(data: dict, layer: int, label: str) -> None:
    """Render the full attention matrix for one model."""
    import plotly.graph_objects as go

    full = _get_full(data, layer)
    if full is None:
        st.warning(f"No full attention matrix for layer {layer}.")
        return

    # Average over heads → (seq, seq)
    avg = full.mean(axis=0).astype(np.float32) if full.ndim == 3 else full.astype(np.float32)
    seq = avg.shape[0]

    # Log-scale for visibility
    log_avg = np.log(avg + 1e-10)

    fig = go.Figure(go.Heatmap(
        z=log_avg,
        colorscale="Viridis",
        showscale=False,
        hovertemplate="from=%{y}, to=%{x}<br>log_attn=%{z:.3f}<extra></extra>",
    ))
    # Token region boundaries
    for boundary in [256, 512, 768]:
        if boundary < seq:
            for axis in ["x", "y"]:
                fig.add_shape(
                    type="line",
                    **({axis + "0": boundary, axis + "1": boundary} |
                       ({"y0": 0, "y1": seq} if axis == "x" else {"x0": 0, "x1": seq})),
                    line=dict(color="white", width=1, dash="dot"),
                )
    fig.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=28, b=0),
        xaxis_title="Key position",
        yaxis_title="Query position",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
    )
    st.plotly_chart(fig, use_container_width=True)


def _entropy_curve(data_a: dict, data_b: dict, layers: list[int], tok_idx: int,
                   label_a: str, label_b: str) -> None:
    """Overlay entropy-vs-layer curves for both models."""
    import plotly.graph_objects as go

    def _entropies(data: dict) -> list[float | None]:
        out = []
        for layer in layers:
            t2i = _get_t2i(data, layer)
            if t2i is None:
                out.append(None)
                continue
            tidx = min(tok_idx, t2i.shape[1] - 1)
            head_attn = t2i[:, tidx, :]   # (8, 512)
            ent = -np.sum(head_attn * np.log(head_attn + 1e-10), axis=-1)
            out.append(float(ent.mean()))
        return out

    xs = [str(l) for l in layers]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=_entropies(data_a), mode="lines+markers",
                             name=label_a, line=dict(color="#4FC3F7")))
    fig.add_trace(go.Scatter(x=xs, y=_entropies(data_b), mode="lines+markers",
                             name=label_b, line=dict(color="#FF8A65", dash="dash")))
    fig.update_layout(
        height=200,
        margin=dict(l=40, r=20, t=10, b=40),
        xaxis_title="Layer",
        yaxis_title="Mean entropy (nats)",
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Token helpers ─────────────────────────────────────────────────────────────

def _get_token_labels(data: dict) -> list[str]:
    meta = data.get("meta", {})
    texts = meta.get("token_texts", [])
    n = meta.get("n_real_tokens", len(texts))
    texts = texts[:n]
    raw = [t.replace("▁", " ").strip() or f"[{i}]" for i, t in enumerate(texts)]
    seen: dict[str, int] = {}
    labels: list[str] = []
    for lbl in raw:
        if raw.count(lbl) > 1:
            seen[lbl] = seen.get(lbl, 0) + 1
            labels.append(f"{lbl}#{seen[lbl]}")
        else:
            labels.append(lbl)
    return labels


# ── Main render ───────────────────────────────────────────────────────────────

def render(
    data_a: dict,
    data_b: dict,
    label_a: str,
    label_b: str,
    available_layers: list[int],
) -> None:
    """Render side-by-side checkpoint comparison.

    Parameters
    ----------
    data_a, data_b:
        Slice dicts from ``inference.run_inference()``.
    label_a, label_b:
        Human-readable checkpoint names for column headers.
    available_layers:
        Intersection of layers present in both checkpoints.
    """
    if not available_layers:
        st.warning("No attention layers found in the comparison data.")
        return

    # ── Instruction banner ────────────────────────────────────────────────────
    instr = data_a.get("meta", {}).get("instruction") or data_b.get("meta", {}).get("instruction")
    if instr:
        st.caption(f"**Instruction:** {instr}")

    # ── Column headers ────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    col_a.markdown(f"### 🅐 `{label_a}`")
    col_b.markdown(f"### 🅑 `{label_b}`")
    st.divider()

    # ── Camera images side by side ────────────────────────────────────────────
    with st.expander("📷 Camera images", expanded=False):
        cimg_a, cimg_b = st.columns(2)
        with cimg_a:
            st.markdown(f"**{label_a}**")
            imgs_a = data_a.get("images", {})
            rc1, rc2 = st.columns(2)
            if imgs_a.get("exterior") is not None:
                rc1.image(imgs_a["exterior"], caption="Exterior", use_container_width=True)
            if imgs_a.get("wrist") is not None:
                rc2.image(imgs_a["wrist"], caption="Wrist", use_container_width=True)
        with cimg_b:
            st.markdown(f"**{label_b}**")
            imgs_b = data_b.get("images", {})
            rc1, rc2 = st.columns(2)
            if imgs_b.get("exterior") is not None:
                rc1.image(imgs_b["exterior"], caption="Exterior", use_container_width=True)
            if imgs_b.get("wrist") is not None:
                rc2.image(imgs_b["wrist"], caption="Wrist", use_container_width=True)

    # ── Shared controls ───────────────────────────────────────────────────────
    st.markdown("#### Shared visualization controls")
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 3])

    with ctrl1:
        camera = st.radio("Camera", ["Exterior", "Wrist"], horizontal=True, key="cmp_cam")
        camera_key = "exterior" if camera == "Exterior" else "wrist"

    with ctrl2:
        agg_name = st.radio("Head aggregation", list(_AGGS.keys()), horizontal=True, key="cmp_agg")
        agg_fn = _AGGS[agg_name]

        head_opts = ["max", "mean"] + list(range(8))
        head_labels = ["Max", "Mean"] + [f"H{i}" for i in range(8)]
        head_sel_lbl = st.radio("Head (heatmap)", head_labels, index=0,
                                horizontal=True, key="cmp_head")
        head_sel = head_opts[head_labels.index(head_sel_lbl)]

    with ctrl3:
        default_layers = [l for l in DEFAULT_LAYERS if l in available_layers]
        if not default_layers:
            default_layers = available_layers[:5]
        selected_layers = st.multiselect(
            "Layers", options=available_layers, default=default_layers, key="cmp_layers"
        )

    if not selected_layers:
        st.info("Select at least one layer.")
        return

    layer_single = st.select_slider(
        "Single layer (for Heatmap & Matrix)",
        options=selected_layers,
        value=selected_layers[min(2, len(selected_layers) - 1)],
        key="cmp_layer_single",
    )

    # ── Token selector (shared) ───────────────────────────────────────────────
    labels_a = _get_token_labels(data_a)
    labels_b = _get_token_labels(data_b)
    # Use model A's tokens as primary; fall back to B if A has none
    token_labels = labels_a if labels_a else labels_b

    if not token_labels:
        st.warning("No token text found.")
        return

    default_tok = token_labels[min(3, len(token_labels) - 1)]
    selected_label = st.pills(
        "Token (click to select):",
        options=token_labels,
        default=default_tok,
        selection_mode="single",
        key="cmp_tok",
    )
    if selected_label is None:
        st.info("Click a token above to continue.")
        return
    tok_idx = token_labels.index(selected_label)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_grid, tab_hmap, tab_matrix, tab_entropy = st.tabs([
        "🔲 Grid Heatmap",
        "🖼 Image Heatmap",
        "📊 Attention Matrix",
        "📈 Entropy Curves",
    ])

    # ── Grid Heatmap tab ──────────────────────────────────────────────────────
    with tab_grid:
        images_a = data_a.get("images", {})
        images_b = data_b.get("images", {})
        _raw_a = images_a.get(camera_key)
        _raw_b = images_b.get(camera_key)
        cam_img_a = _raw_a if _raw_a is not None else np.full((224, 224, 3), 30, dtype=np.uint8)
        cam_img_b = _raw_b if _raw_b is not None else np.full((224, 224, 3), 30, dtype=np.uint8)

        t2i_a = {l: _get_t2i(data_a, l) for l in selected_layers}
        t2i_b = {l: _get_t2i(data_b, l) for l in selected_layers}
        t2i_a = {k: v for k, v in t2i_a.items() if v is not None}
        t2i_b = {k: v for k, v in t2i_b.items() if v is not None}

        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown(f"**{label_a}**")
            with st.spinner("Rendering…"):
                if t2i_a:
                    fig_a = _build_grid_figure(t2i_a, tok_idx, cam_img_a,
                                               camera_key, selected_layers, agg_fn)
                    st.image(_fig_to_bytes(fig_a),
                             caption=f"{camera} · token '{selected_label}' · {agg_name}",
                             use_container_width=True)
                else:
                    st.warning("No attention data.")
        with gc2:
            st.markdown(f"**{label_b}**")
            with st.spinner("Rendering…"):
                if t2i_b:
                    fig_b = _build_grid_figure(t2i_b, tok_idx, cam_img_b,
                                               camera_key, selected_layers, agg_fn)
                    st.image(_fig_to_bytes(fig_b),
                             caption=f"{camera} · token '{selected_label}' · {agg_name}",
                             use_container_width=True)
                else:
                    st.warning("No attention data.")

    # ── Image Heatmap tab ─────────────────────────────────────────────────────
    with tab_hmap:
        hc1, hc2 = st.columns(2)
        with hc1:
            st.markdown(f"**{label_a}**  · L{layer_single} · {head_sel_lbl}")
            _render_heatmap_pair(data_a, label_a, layer_single, tok_idx, head_sel)
        with hc2:
            st.markdown(f"**{label_b}**  · L{layer_single} · {head_sel_lbl}")
            _render_heatmap_pair(data_b, label_b, layer_single, tok_idx, head_sel)

    # ── Attention Matrix tab ──────────────────────────────────────────────────
    with tab_matrix:
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"**{label_a}**  · L{layer_single} (avg over heads)")
            _render_attn_matrix(data_a, layer_single, label_a)
        with mc2:
            st.markdown(f"**{label_b}**  · L{layer_single} (avg over heads)")
            _render_attn_matrix(data_b, layer_single, label_b)

    # ── Entropy Curves tab ────────────────────────────────────────────────────
    with tab_entropy:
        st.markdown(
            f"Mean attention entropy (nats) across all heads for token **'{selected_label}'**, "
            "per layer. Lower = more focused."
        )
        _entropy_curve(data_a, data_b, selected_layers, tok_idx, label_a, label_b)
