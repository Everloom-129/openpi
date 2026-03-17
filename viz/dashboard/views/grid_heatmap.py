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


_AGG_FNS = {
    "All heads": None,
    "Average":   lambda x: x.mean(axis=0),
    "Max":       lambda x: x.max(axis=0),
    "Min":       lambda x: x.min(axis=0),
}


def _build_grid(
    t2i_by_layer: dict[int, np.ndarray],   # layer → (8, n_text, 512)
    tok_idx: int,
    camera_img: np.ndarray,                 # (H, W, 3) uint8
    camera: str,
    layers: list[int],
    agg: str = "All heads",
) -> plt.Figure:
    """Render rows=layers × cols=heads (or single aggregated col) matplotlib figure."""
    agg_fn = _AGG_FNS[agg]
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
            list(_AGG_FNS.keys()),
            index=0,
            horizontal=True,
            key="gh_agg",
        )

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

    # ── Build and display grid ────────────────────────────────────────────────
    with st.spinner("Rendering grid…"):
        fig = _build_grid(t2i_by_layer, tok_idx, camera_img, camera_key, selected_layers, agg)
        img_bytes = _fig_to_bytes(fig)

    st.image(img_bytes, caption=f'Layer × Head grid — token "{selected_label}" — {camera} — {agg}',
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
