"""Grid Heatmap: text token → image attention across all layers and heads.

Shows a (n_layers × 8_heads) grid of camera images with attention overlays.
A slider selects the active token. Camera toggle switches exterior/wrist.

Isolated from other views — no shared state.
"""
from __future__ import annotations

import io

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import streamlit as st

NUM_IMAGE_TOKENS = 256
PATCH_GRID = 16
DEFAULT_LAYERS = [1, 4, 5, 7, 10]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _attn_to_heatmap(attn_512: np.ndarray, camera: str) -> np.ndarray:
    """Extract 16×16 patch grid for one camera and upsample to 112×112."""
    if camera == "exterior":
        patches = attn_512[:NUM_IMAGE_TOKENS]
    else:
        patches = attn_512[NUM_IMAGE_TOKENS : 2 * NUM_IMAGE_TOKENS]
    grid = patches.reshape(PATCH_GRID, PATCH_GRID).astype(np.float32)
    up = cv2.resize(grid, (112, 112), interpolation=cv2.INTER_LINEAR)
    return up


def _overlay(img: np.ndarray, hmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend heatmap (112×112 float) on top of img (resized to 112×112)."""
    img_s = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    hmin, hmax = hmap.min(), hmap.max()
    hn = (hmap - hmin) / (hmax - hmin + 1e-8)
    hmap_u8 = (hn * 255).astype(np.uint8)
    color = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_s, 1 - alpha, color, alpha, 0)
    return blended


_SIMPLE_AGGS = {
    "All heads": None,
    "Average":   lambda x: x.mean(axis=0),
    "Median":    lambda x: np.median(x, axis=0),
    "Max":       lambda x: x.max(axis=0),
    "Min":       lambda x: x.min(axis=0),
    "Max - Min": lambda x: x.max(axis=0) - x.min(axis=0),
    "Std Dev":   lambda x: x.std(axis=0),
    "Entropy":   lambda x: -np.sum(
                     (x / (x.sum(axis=0, keepdims=True) + 1e-10))
                     * np.log(x / (x.sum(axis=0, keepdims=True) + 1e-10) + 1e-10),
                     axis=0),
}
_PARAM_AGGS = ["Top-K Focused", "Count Above Threshold"]
_ALL_AGG_NAMES = list(_SIMPLE_AGGS.keys()) + _PARAM_AGGS

_AGG_DESCRIPTIONS = {
    "All heads":             "Shows each head's attention map independently as a grid of 8 columns.",
    "Average":               "Mean attention across all 8 heads. Good general-purpose summary.",
    "Median":                "Median attention across heads. More robust than average — ignores outlier heads that fire on everything.",
    "Max":                   "Maximum attention value across heads per patch. Highlights any patch that at least one head attends to strongly.",
    "Min":                   "Minimum attention across heads. Shows only patches that all heads agree on attending to.",
    "Max - Min":             "Range (max − min) across heads per patch. High values reveal patches where heads strongly disagree — useful for spotting head specialization.",
    "Std Dev":               "Standard deviation across heads per patch. High = heads diverge in attention; low = heads are in consensus.",
    "Entropy":               "Per-patch entropy of attention values across heads (normalized). High entropy = heads attend here roughly equally; low entropy = one head dominates this patch.",
    "Top-K Focused":         "Average of the K least-entropic (most spatially focused) heads only. Filters out diffuse heads to reduce noise.",
    "Count Above Threshold": "Number of heads (out of 8) that rank a patch in their top N% of attention. Intuitive '5/8 heads agree here' reading.",
}


def _make_topk_fn(k: int):
    def fn(x):  # x: (8, 512) or (8, n_text, 512)
        x_flat = x.reshape(x.shape[0], -1)  # (8, *)
        ent = -np.sum(x_flat * np.log(x_flat + 1e-10), axis=-1)  # (8,)
        idx = np.argsort(ent)[:k]
        return x[idx].mean(axis=0)
    return fn


def _make_count_fn(pct: int):
    def fn(x):  # x: (8, 512)
        thresh = np.percentile(x, 100 - pct, axis=-1, keepdims=True)
        return (x >= thresh).sum(axis=0).astype(np.float32)
    return fn


def _build_grid(
    t2i_by_layer: dict[int, np.ndarray],   # layer → (8, n_text, 512)
    tok_idx: int,
    camera_img: np.ndarray,                 # (H, W, 3) uint8
    camera: str,
    layers: list[int],
    agg_fn=None,
    agg_label: str = "All heads",
) -> plt.Figure:
    """Render rows=layers × cols=heads (or aggregated grid) matplotlib figure."""
    cell_size = 1.4  # inches per cell

    if agg_fn is not None:
        # Rows of up to 5 layers each
        cols_per_row = 5
        n_cols = cols_per_row
        n_rows = (len(layers) + cols_per_row - 1) // cols_per_row
        fig = plt.figure(figsize=(n_cols * cell_size, n_rows * (cell_size + 0.3)), dpi=100)
        fig.patch.set_facecolor("#0e1117")
        gs = gridspec.GridSpec(
            n_rows, n_cols,
            figure=fig,
            wspace=0.04,
            hspace=0.25,
            left=0.02, right=1.0,
            top=0.93, bottom=0.0,
        )
        for idx, layer in enumerate(layers):
            row_i, col_i = divmod(idx, cols_per_row)
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.set_xticks([])
            ax.set_yticks([])
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
                cell = _overlay(camera_img, hmap)
                ax.imshow(cell, aspect="equal")
            ax.set_title(f"L{layer}", color="white", fontsize=8, pad=3)

        # Hide unused axes in the last row
        for idx in range(len(layers), n_rows * n_cols):
            row_i, col_i = divmod(idx, cols_per_row)
            fig.add_subplot(gs[row_i, col_i]).set_visible(False)
    else:
        # Full grid: rows=layers × cols=heads
        n_rows, n_cols = len(layers), 8
        fig = plt.figure(figsize=(n_cols * cell_size, n_rows * cell_size), dpi=100)
        fig.patch.set_facecolor("#0e1117")
        gs = gridspec.GridSpec(
            n_rows, n_cols,
            figure=fig,
            wspace=0.04,
            hspace=0.08,
            left=0.06, right=1.0,
            top=0.93, bottom=0.0,
        )
        for row_i, layer in enumerate(layers):
            t2i = t2i_by_layer.get(layer)
            for col_i in range(n_cols):
                ax = fig.add_subplot(gs[row_i, col_i])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if t2i is None:
                    ax.set_facecolor("#1a1a2e")
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, color="gray", fontsize=7)
                else:
                    attn_512 = t2i[col_i, tok_idx, :]
                    hmap = _attn_to_heatmap(attn_512, camera)
                    cell = _overlay(camera_img, hmap)
                    ax.imshow(cell, aspect="auto")
                if col_i == 0:
                    ax.set_ylabel(f"L{layer}", color="white", fontsize=8,
                                  rotation=0, labelpad=22, va="center")
                if row_i == 0:
                    ax.set_title(f"H{col_i}", color="white", fontsize=8, pad=3)

    return fig


def _fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Main render ───────────────────────────────────────────────────────────────

def render(data: dict, available_layers: list[int]) -> None:
    """Render the grid heatmap tab."""

    meta = data.get("meta", {})
    n_real: int = meta.get("n_real_tokens", len(meta.get("token_texts", [])))
    all_texts: list[str] = meta.get("token_texts", [])
    real_texts = all_texts[:n_real]

    images = data.get("images", {})
    _ext = images.get("exterior")
    _wrist = images.get("wrist")
    ext_img = _ext if _ext is not None else np.full((224, 224, 3), 30, dtype=np.uint8)
    wrist_img = _wrist if _wrist is not None else np.full((224, 224, 3), 30, dtype=np.uint8)

    if not available_layers:
        st.warning("No attention layers found.")
        return

    if not real_texts:
        st.warning("No token texts found.")
        return

    # ── Camera + Layer + Aggregation controls ─────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns([2, 3])

    with ctrl_col1:
        camera = st.radio("Camera", ["Exterior", "Wrist"], index=0,
                          horizontal=True, key="gh_cam")
        camera_key = "exterior" if camera == "Exterior" else "wrist"
        camera_img = ext_img if camera_key == "exterior" else wrist_img
        agg = st.radio(
            "Head aggregation",
            _ALL_AGG_NAMES,
            index=0,
            horizontal=True,
            key="gh_agg",
        )
        if agg == "Top-K Focused":
            topk_k = st.slider("K (focused heads)", 1, 8, 4, key="gh_topk")
        elif agg == "Count Above Threshold":
            count_pct = st.slider("Top % threshold", 1, 50, 10, key="gh_pct")
        st.caption(_AGG_DESCRIPTIONS[agg])

    with ctrl_col2:
        default_layers = [l for l in DEFAULT_LAYERS if l in available_layers]
        if not default_layers:
            default_layers = available_layers[:5]
        selected_layers = st.multiselect(
            "Layers",
            options=available_layers,
            default=default_layers,
            key="gh_layers",
        )

    if not selected_layers:
        st.info("Select at least one layer.")
        return

    # ── Clickable token pills ──────────────────────────────────────────────────
    # Deduplicate labels by appending index when two tokens share the same text.
    raw_labels = [t.replace("▁", " ").strip() or f"[{i}]" for i, t in enumerate(real_texts)]
    seen: dict[str, int] = {}
    token_labels: list[str] = []
    for i, lbl in enumerate(raw_labels):
        if raw_labels.count(lbl) > 1:
            seen[lbl] = seen.get(lbl, 0) + 1
            token_labels.append(f"{lbl}#{seen[lbl]}")
        else:
            token_labels.append(lbl)

    default_tok = token_labels[min(3, n_real - 1)]
    has_dupes = any(raw_labels.count(r) > 1 for r in raw_labels)
    label = (
        "Click a token to visualize its attention: (duplicate tokens are suffixed #1, #2, …)"
        if has_dupes
        else "Click a token to visualize its attention:"
    )
    selected_label = st.pills(
        label,
        options=token_labels,
        default=default_tok,
        selection_mode="single",
        key="gh_tok",
    )
    if selected_label is None:
        st.info("Click a token above to visualize its attention.")
        return
    tok_idx = token_labels.index(selected_label)

    # ── Load selected layers ──────────────────────────────────────────────────
    load_fn = data.get("_load_t2i")
    t2i_by_layer: dict[int, np.ndarray] = {}
    for layer in selected_layers:
        if load_fn is not None:
            arr = load_fn(layer)
        else:
            arr = data.get("prefix", {}).get(f"layer_{layer}", {}).get("text_to_img")
        if arr is not None:
            t2i_by_layer[layer] = arr

    # ── Resolve aggregation function ──────────────────────────────────────────
    if agg in _SIMPLE_AGGS:
        agg_fn = _SIMPLE_AGGS[agg]
        agg_label = agg
    elif agg == "Top-K Focused":
        agg_fn = _make_topk_fn(topk_k)
        agg_label = f"Top-{topk_k} Focused"
    else:  # Count Above Threshold
        agg_fn = _make_count_fn(count_pct)
        agg_label = f"Count ≥ Top {count_pct}%"

    # ── Build and display grid ────────────────────────────────────────────────
    with st.spinner("Rendering grid…"):
        fig = _build_grid(t2i_by_layer, tok_idx, camera_img, camera_key, selected_layers,
                          agg_fn=agg_fn, agg_label=agg_label)
        img_bytes = _fig_to_bytes(fig)

    st.image(img_bytes, caption=f'Layer × Head grid — token "{selected_label}" — {camera} — {agg_label}',
             use_container_width=True)

    # ── Per-layer entropy row ─────────────────────────────────────────────────
    with st.expander("Layer entropy (how focused is attention on this token?)"):
        import plotly.graph_objects as go

        entropies_mean = []
        entropies_max_head = []
        for layer in selected_layers:
            t2i = t2i_by_layer.get(layer)
            if t2i is None:
                entropies_mean.append(None)
                entropies_max_head.append(None)
                continue
            head_attn = t2i[:, tok_idx, :]   # (8, 512)
            ent = -np.sum(head_attn * np.log(head_attn + 1e-10), axis=-1)  # (8,)
            entropies_mean.append(float(ent.mean()))
            entropies_max_head.append(float(ent.min()))  # min entropy = most focused head

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=[str(l) for l in selected_layers], y=entropies_mean,
            mode="lines+markers", name="Mean entropy",
            line=dict(color="steelblue"),
        ))
        fig2.add_trace(go.Scatter(
            x=[str(l) for l in selected_layers], y=entropies_max_head,
            mode="lines+markers", name="Most focused head",
            line=dict(color="coral", dash="dash"),
        ))
        fig2.update_layout(
            height=220,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis_title="Layer",
            yaxis_title="Entropy (nats)",
            legend=dict(orientation="h", y=1.1),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
        )
        st.plotly_chart(fig2, use_container_width=True)
