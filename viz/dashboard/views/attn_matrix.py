"""Tab 2: BertViz-style attention matrix.

Shows the full (seq_len × seq_len) attention heatmap for key layers
that have the full matrix stored, with click-to-inspect row detail.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

NUM_IMAGE_TOKENS = 256
TEXT_START_IDX = 768
FULL_MATRIX_LAYERS = [1, 4, 5, 7, 10]


def _token_label(idx: int, token_texts: list[str], seq_len: int) -> str:
    """Human-readable label for a sequence position."""
    if idx < NUM_IMAGE_TOKENS:
        r, c = divmod(idx, 16)
        return f"ext[{r},{c}]"
    if idx < 2 * NUM_IMAGE_TOKENS:
        r, c = divmod(idx - NUM_IMAGE_TOKENS, 16)
        return f"wrist[{r},{c}]"
    if idx < TEXT_START_IDX:
        r, c = divmod(idx - 2 * NUM_IMAGE_TOKENS, 16)
        return f"dummy[{r},{c}]"
    tok_local = idx - TEXT_START_IDX
    if tok_local < len(token_texts):
        label = token_texts[tok_local].replace("▁", " ").strip()
        return label or f"tok_{tok_local}"
    return f"tok_{tok_local}"


def render(
    data: dict,
    available_layers: list[int],
) -> None:
    """Render Tab 2: Attention Matrix."""

    full_layers = [l for l in available_layers if l in FULL_MATRIX_LAYERS]
    if not full_layers:
        st.warning(
            "No full attention matrices available. "
            "Re-run conversion with key layers {1,4,5,7,10} included."
        )
        return

    meta = data.get("meta", {})
    token_texts: list[str] = meta.get("token_texts", [])
    seq_len: int = meta.get("seq_len", 968)

    # ── Controls ──────────────────────────────────────────────────────────────
    col_l, col_h = st.columns([2, 3])
    with col_l:
        layer = st.selectbox("Layer (full matrix layers only)", full_layers, key="am_layer")

    with col_h:
        head_labels = ["Mean over heads", "Max over heads"] + [f"Head {i}" for i in range(8)]
        head_opts: list[str | int] = ["mean", "max"] + list(range(8))
        head_sel_label = st.radio(
            "Head",
            head_labels,
            index=0,
            horizontal=True,
            key="am_head",
        )
        head_sel = head_opts[head_labels.index(head_sel_label)]

    # ── Load matrix ────────────────────────────────────────────────────────────
    loader_fn = data.get("_load_full_all")
    mat_all = None
    if loader_fn is not None:
        mat_all = loader_fn(layer)   # (8, seq, seq)
    else:
        mat_all = data.get("prefix", {}).get(f"layer_{layer}", {}).get("full")

    if mat_all is None:
        st.warning(f"Full matrix not available for layer {layer}.")
        return

    # Aggregate
    if head_sel == "mean":
        mat = mat_all.mean(axis=0)
    elif head_sel == "max":
        mat = mat_all.max(axis=0)
    else:
        mat = mat_all[int(head_sel)]

    mat = mat.astype(np.float32)
    s = mat.shape[0]

    # ── Display controls ───────────────────────────────────────────────────────
    col_zoom, col_range = st.columns([2, 3])
    with col_zoom:
        view_options = ["Text→Text", "Text→Image", "All"]
        view_mode = st.radio("View region", view_options, index=0, horizontal=True, key="am_view")

    if view_mode == "Text→Text":
        r_start, r_end = TEXT_START_IDX, s
        c_start, c_end = TEXT_START_IDX, s
    elif view_mode == "Text→Image":
        r_start, r_end = TEXT_START_IDX, s
        c_start, c_end = 0, min(512, s)
    else:
        r_start, r_end = 0, s
        c_start, c_end = 0, s

    sub = mat[r_start:r_end, c_start:c_end]

    # Build axis labels (sampled if too large)
    MAX_LABELS = 100

    def make_labels(start, end):
        n = end - start
        if n <= MAX_LABELS:
            return [_token_label(i, token_texts, s) for i in range(start, end)]
        step = max(1, n // MAX_LABELS)
        return [_token_label(i, token_texts, s) if (i - start) % step == 0 else "" for i in range(start, end)]

    row_labels = make_labels(r_start, r_end)
    col_labels = make_labels(c_start, c_end)

    # ── Main heatmap ───────────────────────────────────────────────────────────
    height = min(700, max(300, (r_end - r_start) * 3 + 80))

    fig = go.Figure(
        go.Heatmap(
            z=sub,
            x=col_labels,
            y=row_labels,
            colorscale="Blues",
            hovertemplate="row=%{y}<br>col=%{x}<br>weight=%{z:.5f}<extra></extra>",
            showscale=True,
        )
    )
    fig.update_layout(
        title=f"Attention Matrix — Layer {layer}, {head_sel_label} — {view_mode}",
        height=height,
        margin=dict(l=60, r=20, t=40, b=60),
        xaxis=dict(title="Key tokens", tickfont=dict(size=8)),
        yaxis=dict(title="Query tokens", tickfont=dict(size=8), autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Row inspector (click-to-inspect row) ──────────────────────────────────
    st.markdown("**Inspect a row (query token):**")
    n_rows = r_end - r_start
    row_display = [_token_label(i, token_texts, s) for i in range(r_start, r_end)]
    selected_row_label = st.selectbox("Query token", row_display, index=0, key="am_row_sel")
    sel_local = row_display.index(selected_row_label)

    row_attn = sub[sel_local]   # (n_cols,)
    top_k = min(20, len(row_attn))
    top_idx = np.argsort(row_attn)[::-1][:top_k]
    top_vals = row_attn[top_idx]
    top_labs = [col_labels[i] or f"pos_{c_start + i}" for i in top_idx]

    fig2 = go.Figure(
        go.Bar(
            x=top_labs,
            y=top_vals,
            marker_color="steelblue",
            hovertemplate="%{x}: %{y:.5f}<extra></extra>",
        )
    )
    fig2.update_layout(
        title=f"Top-{top_k} attention targets for query: '{selected_row_label}'",
        height=260,
        margin=dict(l=40, r=20, t=40, b=60),
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(title="Attention weight"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Stats panel ────────────────────────────────────────────────────────────
    with st.expander("Layer statistics"):
        text_rows = mat[TEXT_START_IDX:, TEXT_START_IDX:]
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Tokens", f"{s}")
        col_b.metric("Text tokens", f"{s - TEXT_START_IDX}")
        ent = -np.sum(text_rows * np.log(text_rows + 1e-10), axis=-1).mean()
        col_c.metric("Mean entropy (text rows)", f"{ent:.2f}")
        col_d.metric("Max weight", f"{mat.max():.5f}")
