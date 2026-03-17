"""Tab 1: Word → Image attention heatmap.

Shows attention from each text token to exterior/wrist camera patches,
overlaid as a heatmap on the camera images.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


NUM_IMAGE_TOKENS = 256   # 16×16 patches per camera
PATCH_GRID = 16          # patches per side


def _upsample_heatmap(attn_512: np.ndarray, camera: str) -> np.ndarray:
    """Extract one camera's 16×16 patch from 512-dim vector and upsample to 224×224."""
    if camera == "exterior":
        patches = attn_512[:NUM_IMAGE_TOKENS]
    else:
        patches = attn_512[NUM_IMAGE_TOKENS:2 * NUM_IMAGE_TOKENS]

    grid = patches.reshape(PATCH_GRID, PATCH_GRID).astype(np.float32)
    # Upsample 16→224 via np.kron (factor 14)
    upsampled = np.kron(grid, np.ones((14, 14), dtype=np.float32))
    return upsampled  # (224, 224)


def _aggregate_heads(t2i: np.ndarray, head_sel: int | str) -> np.ndarray:
    """Aggregate text_to_img (8, n_text, 512) over heads.

    head_sel: int (specific head), "max", or "mean"
    Returns (n_text, 512).
    """
    if isinstance(head_sel, int):
        return t2i[head_sel]
    if head_sel == "max":
        return t2i.max(axis=0)
    return t2i.mean(axis=0)


def _make_overlay_figure(
    camera_img: np.ndarray,   # (224, 224, 3) uint8
    heatmap: np.ndarray,      # (224, 224) float32
    title: str,
) -> go.Figure:
    """Camera image + heatmap overlay as a Plotly figure."""
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        hmap_norm = (heatmap - hmin) / (hmax - hmin)
    else:
        hmap_norm = heatmap * 0.0

    h, w = camera_img.shape[:2]

    fig = go.Figure()
    # Base image
    fig.add_trace(go.Image(z=camera_img, name="camera"))
    # Heatmap overlay
    fig.add_trace(
        go.Heatmap(
            z=hmap_norm,
            colorscale="Jet",
            opacity=0.45,
            showscale=False,
            hovertemplate="patch_r=%{y}, patch_c=%{x}<br>weight=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font_size=13),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showticklabels=False, range=[0, w]),
        yaxis=dict(showticklabels=False, range=[h, 0]),
        height=260,
        width=280,
    )
    return fig


def render(
    data: dict,
    available_layers: list[int],
) -> None:
    """Render Tab 1: Image Heatmap.

    `data` is either an HDF5-backed loader dict or an in-memory slice dict.
    Expected keys from caller: h5_path (str) for HDF5 mode, or inline arrays.
    """
    if not available_layers:
        st.warning("No attention layers found.")
        return

    meta = data.get("meta", {})
    token_texts: list[str] = meta.get("token_texts", [])
    instruction: str = meta.get("instruction", "")

    if instruction:
        st.caption(f"**Instruction:** {instruction}")

    # ── Controls ──────────────────────────────────────────────────────────────
    col_l, col_h, col_agg = st.columns([2, 3, 2])

    with col_l:
        layer = st.select_slider(
            "Layer",
            options=available_layers,
            value=available_layers[min(2, len(available_layers) - 1)],
            key="ih_layer",
        )

    with col_h:
        head_opts = ["max", "mean"] + list(range(8))
        head_labels = ["Max over heads", "Mean over heads"] + [f"Head {i}" for i in range(8)]
        head_sel_label = st.radio(
            "Head",
            options=head_labels,
            index=0,
            horizontal=True,
            key="ih_head",
        )
        head_sel = head_opts[head_labels.index(head_sel_label)]

    # ── Load attention ─────────────────────────────────────────────────────────
    t2i = data.get("_t2i_cache", {}).get((layer,))
    if t2i is None:
        loader_fn = data.get("_load_t2i")
        if loader_fn is not None:
            t2i = loader_fn(layer)
        else:
            # inline mode
            t2i = data.get("prefix", {}).get(f"layer_{layer}", {}).get("text_to_img")

    if t2i is None:
        st.warning(f"text_to_img not found for layer {layer}.")
        return

    n_text = t2i.shape[1]

    # Align token_texts length
    if len(token_texts) < n_text:
        token_texts = token_texts + [f"tok_{i}" for i in range(len(token_texts), n_text)]
    token_texts = token_texts[:n_text]

    # ── Token selector ────────────────────────────────────────────────────────
    st.markdown("**Select a token:**")

    if "ih_selected_tok" not in st.session_state:
        st.session_state["ih_selected_tok"] = 0

    # Render token pills as buttons (chunks of 20 per row)
    TOKENS_PER_ROW = 20
    for row_start in range(0, n_text, TOKENS_PER_ROW):
        row_toks = token_texts[row_start:row_start + TOKENS_PER_ROW]
        cols = st.columns(len(row_toks))
        for j, (col, tok) in enumerate(zip(cols, row_toks)):
            tok_idx = row_start + j
            is_selected = st.session_state["ih_selected_tok"] == tok_idx
            label = tok.replace("▁", " ").strip() or f"[{tok_idx}]"
            # Truncate long tokens
            display = label[:8] + "…" if len(label) > 9 else label
            btn_type = "primary" if is_selected else "secondary"
            if col.button(display, key=f"tok_{tok_idx}", type=btn_type, use_container_width=True):
                st.session_state["ih_selected_tok"] = tok_idx
                st.rerun()

    tok_idx = st.session_state["ih_selected_tok"]
    tok_idx = min(tok_idx, n_text - 1)

    sel_token_name = token_texts[tok_idx].replace("▁", " ").strip() or f"tok_{tok_idx}"
    st.caption(f"Selected: **{sel_token_name}** (index {tok_idx})")

    # ── Compute heatmap ───────────────────────────────────────────────────────
    agg = _aggregate_heads(t2i, head_sel)   # (n_text, 512)
    attn_512 = agg[tok_idx]                 # (512,)

    ext_hmap = _upsample_heatmap(attn_512, "exterior")
    wrist_hmap = _upsample_heatmap(attn_512, "wrist")

    # ── Load images ───────────────────────────────────────────────────────────
    images = data.get("images", {})
    ext_img = images.get("exterior")
    wrist_img = images.get("wrist")

    if ext_img is None:
        ext_img = np.full((224, 224, 3), 30, dtype=np.uint8)
    if wrist_img is None:
        wrist_img = np.full((224, 224, 3), 30, dtype=np.uint8)

    # ── Plotly figures ────────────────────────────────────────────────────────
    fig_ext = _make_overlay_figure(ext_img, ext_hmap, f"Exterior — '{sel_token_name}'")
    fig_wrist = _make_overlay_figure(wrist_img, wrist_hmap, f"Wrist — '{sel_token_name}'")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_ext, use_container_width=True)
    with c2:
        st.plotly_chart(fig_wrist, use_container_width=True)

    # ── Stats ─────────────────────────────────────────────────────────────────
    with st.expander("Attention stats for selected token"):
        head_attn = t2i[:, tok_idx, :]  # (8, 512)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Max weight", f"{head_attn.max():.4f}")
        col_b.metric("Mean weight", f"{head_attn.mean():.4f}")
        entropy = -np.sum(head_attn * np.log(head_attn + 1e-10), axis=-1).mean()
        col_c.metric("Mean entropy (nats)", f"{entropy:.2f}")
