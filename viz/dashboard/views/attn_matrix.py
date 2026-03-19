"""Tab 2: BertViz-style attention matrix.

Shows the full (seq_len × seq_len) attention heatmap for key layers
that have the full matrix stored, with click-to-inspect row detail.

Token layout (Pi0.5 / DROID):
  [0:256]   ext camera patches
  [256:512] wrist camera patches
  [512:768] zero-padding (dummy)
  [768:768+n_real]  real text tokens (instruction + robot state)
  [768+n_real:s-8]  padding text tokens  (if any)
  [s-8:s]           action suffix tokens
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

NUM_IMAGE_TOKENS = 256
TOTAL_IMAGE_TOKENS = 512   # ext + wrist
TEXT_START_IDX = 768
ACTION_SUFFIX_LEN = 8
FULL_MATRIX_LAYERS = [1, 4, 5, 7, 10]


def _token_label(idx: int, token_texts: list[str], seq_len: int) -> str:
    """Human-readable label for a sequence position."""
    if idx < NUM_IMAGE_TOKENS:
        r, c = divmod(idx, 16)
        return f"ext[{r},{c}]"
    if idx < TOTAL_IMAGE_TOKENS:
        r, c = divmod(idx - NUM_IMAGE_TOKENS, 16)
        return f"wrist[{r},{c}]"
    if idx < TEXT_START_IDX:
        r, c = divmod(idx - TOTAL_IMAGE_TOKENS, 16)
        return f"pad[{r},{c}]"
    tok_local = idx - TEXT_START_IDX
    if tok_local < len(token_texts):
        label = token_texts[tok_local].replace("▁", " ").strip()
        return label or f"tok_{tok_local}"
    # beyond token_texts: action suffix
    action_local = idx - TEXT_START_IDX - len(token_texts)
    if action_local >= 0:
        return f"act_{action_local}"
    return f"tok_{tok_local}"


def _make_token_labels(real_texts: list[str]) -> list[str]:
    """Deduplicate token labels — mirrors grid_heatmap.py."""
    raw = [t.replace("▁", " ").strip() or f"[{i}]" for i, t in enumerate(real_texts)]
    seen: dict[str, int] = {}
    labels: list[str] = []
    for lbl in raw:
        if raw.count(lbl) > 1:
            seen[lbl] = seen.get(lbl, 0) + 1
            labels.append(f"{lbl}#{seen[lbl]}")
        else:
            labels.append(lbl)
    return labels


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
    n_real: int = meta.get("n_real_tokens", len(token_texts))
    real_texts = token_texts[:n_real]
    token_labels = _make_token_labels(real_texts)
    has_dupes = any(token_labels[i] != (real_texts[i].replace("▁", " ").strip() or f"[{i}]")
                    for i in range(len(real_texts)))

    # Derived boundaries (clamped to actual seq_len)
    text_real_end   = min(TEXT_START_IDX + n_real, seq_len)   # end of real text tokens
    action_start    = max(seq_len - ACTION_SUFFIX_LEN, text_real_end)
    action_end      = seq_len

    # ── Controls row 1: Layer + Head ──────────────────────────────────────────
    col_l, col_h = st.columns([2, 3])
    with col_l:
        layer = st.selectbox("Layer (full matrix layers only)", full_layers, key="am_layer")

    with col_h:
        head_labels = ["Mean over heads", "Max over heads"] + [f"Head {i}" for i in range(8)]
        head_opts: list[str | int] = ["mean", "max"] + list(range(8))
        head_sel_label = st.radio(
            "Head", head_labels, index=0, horizontal=True, key="am_head",
        )
        head_sel = head_opts[head_labels.index(head_sel_label)]

    # ── Controls row 2: Row filter + Column filter + Normalize ────────────────
    col_row, col_col, col_norm = st.columns([2, 3, 2])

    with col_row:
        row_options = ["Real text", "All text", "Full seq"]
        row_filter = st.radio(
            "Query rows",
            row_options,
            index=0,
            horizontal=False,
            key="am_row_filter",
            help=(
                "**Real text** — only the instruction + robot-state tokens that were actually used "
                f"(positions {TEXT_START_IDX}–{text_real_end - 1}, n={n_real}).\n\n"
                "**All text** — all positions from TEXT_START onward (includes padding + action).\n\n"
                "**Full seq** — every token including image patches."
            ),
        )

    with col_col:
        col_options = [
            "Full sequence",
            "Image only (ext + wrist)",
            "Ext camera only",
            "Wrist camera only",
            "Real text only",
            "Action suffix only",
        ]
        col_filter = st.radio(
            "Key columns",
            col_options,
            index=0,
            horizontal=False,
            key="am_col_filter",
            help=(
                "**Full sequence** — all key positions.\n\n"
                "**Image only** — ext (0–255) + wrist (256–511).\n\n"
                "**Real text only** — instruction + robot state tokens.\n\n"
                "**Action suffix** — the 8 action tokens at the end of the sequence."
            ),
        )

    with col_norm:
        normalize_rows = st.checkbox(
            "Normalize rows",
            value=False,
            key="am_norm",
            help="Divide each query row by its max. Useful when some tokens have "
                 "very low absolute attention — brings out relative patterns.",
        )

    # ── Load + aggregate matrix ────────────────────────────────────────────────
    loader_fn = data.get("_load_full_all")
    mat_all = None
    if loader_fn is not None:
        mat_all = loader_fn(layer)
    else:
        mat_all = data.get("prefix", {}).get(f"layer_{layer}", {}).get("full")

    if mat_all is None:
        st.warning(f"Full matrix not available for layer {layer}.")
        return

    if head_sel == "mean":
        mat = mat_all.mean(axis=0)
    elif head_sel == "max":
        mat = mat_all.max(axis=0)
    else:
        mat = mat_all[int(head_sel)]

    mat = mat.astype(np.float32)
    s = mat.shape[0]

    # ── Resolve row bounds ────────────────────────────────────────────────────
    if row_filter == "Real text":
        r_start, r_end = TEXT_START_IDX, min(text_real_end, s)
    elif row_filter == "All text":
        r_start, r_end = TEXT_START_IDX, s
    else:  # Full seq
        r_start, r_end = 0, s

    # ── Resolve col bounds ────────────────────────────────────────────────────
    if col_filter == "Full sequence":
        c_start, c_end = 0, s
    elif col_filter == "Image only (ext + wrist)":
        c_start, c_end = 0, min(TOTAL_IMAGE_TOKENS, s)
    elif col_filter == "Ext camera only":
        c_start, c_end = 0, min(NUM_IMAGE_TOKENS, s)
    elif col_filter == "Wrist camera only":
        c_start, c_end = NUM_IMAGE_TOKENS, min(TOTAL_IMAGE_TOKENS, s)
    elif col_filter == "Real text only":
        c_start, c_end = TEXT_START_IDX, min(text_real_end, s)
    else:  # Action suffix only
        c_start, c_end = max(0, action_start), min(action_end, s)

    sub = mat[r_start:r_end, c_start:c_end].copy()

    if normalize_rows:
        row_max = sub.max(axis=1, keepdims=True)
        sub = sub / (row_max + 1e-8)

    # ── Build axis labels (sampled if too large) ──────────────────────────────
    MAX_LABELS = 100

    def make_labels(start, end):
        n = end - start
        if n <= MAX_LABELS:
            return [_token_label(i, token_texts, s) for i in range(start, end)]
        step = max(1, n // MAX_LABELS)
        return [
            _token_label(i, token_texts, s) if (i - start) % step == 0 else ""
            for i in range(start, end)
        ]

    # For real-text rows, use deduplicated token_labels instead of positional labels
    if row_filter == "Real text" and token_labels:
        row_labels = token_labels[: r_end - r_start]
    else:
        row_labels = make_labels(r_start, r_end)

    col_labels = make_labels(c_start, c_end)

    # ── Main heatmap ───────────────────────────────────────────────────────────
    height = min(800, max(300, (r_end - r_start) * 5 + 80))
    colorscale = "Viridis" if normalize_rows else "Blues"
    norm_suffix = " (row-normalized)" if normalize_rows else ""

    fig = go.Figure(
        go.Heatmap(
            z=sub,
            x=col_labels,
            y=row_labels,
            colorscale=colorscale,
            hovertemplate="row=%{y}<br>col=%{x}<br>weight=%{z:.5f}<extra></extra>",
            showscale=True,
        )
    )
    fig.update_layout(
        title=(
            f"Attention Matrix — Layer {layer}, {head_sel_label} — "
            f"rows: {row_filter}  cols: {col_filter}{norm_suffix}"
        ),
        height=height,
        margin=dict(l=80, r=20, t=50, b=60),
        xaxis=dict(title="Key tokens", tickfont=dict(size=8)),
        yaxis=dict(title="Query tokens", tickfont=dict(size=8), autorange="reversed"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Token pills row selector ───────────────────────────────────────────────
    st.markdown("**Inspect a query token:**")

    if row_filter == "Real text" and token_labels:
        pills_label = (
            "Click a token to inspect its attention row: (duplicates are suffixed #1, #2, …)"
            if has_dupes
            else "Click a token to inspect its attention row:"
        )
        default_pill = token_labels[min(3, n_real - 1)]
        selected_pill = st.pills(
            pills_label,
            options=token_labels[: r_end - r_start],
            default=default_pill,
            selection_mode="single",
            key="am_pill",
        )
        if selected_pill is None:
            st.info("Click a token above to inspect its attention row.")
            return
        tok_local = token_labels.index(selected_pill)
        seq_row = TEXT_START_IDX + tok_local
        sel_display = selected_pill
    else:
        row_display = [_token_label(i, token_texts, s) for i in range(r_start, r_end)]
        selected_row_label = st.selectbox("Query token", row_display, index=0, key="am_row_sel")
        tok_local = row_display.index(selected_row_label)
        seq_row = r_start + tok_local
        sel_display = selected_row_label

    # ── Row detail bar chart ───────────────────────────────────────────────────
    if r_start <= seq_row < r_end:
        row_local = seq_row - r_start
        row_attn = sub[row_local]

        top_k = min(20, len(row_attn))
        top_idx = np.argsort(row_attn)[::-1][:top_k]
        top_vals = row_attn[top_idx]
        top_labs = [col_labels[i] or f"pos_{c_start + i}" for i in top_idx]

        fig2 = go.Figure(go.Bar(
            x=top_labs,
            y=top_vals,
            marker_color="steelblue",
            hovertemplate="%{x}: %{y:.5f}<extra></extra>",
        ))
        fig2.update_layout(
            title=f"Top-{top_k} attention targets for '{sel_display}'",
            height=260,
            margin=dict(l=40, r=20, t=40, b=60),
            xaxis=dict(tickfont=dict(size=9)),
            yaxis=dict(title="Attention weight" + (" (normalized)" if normalize_rows else "")),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Region breakdown (always uses raw un-normalized mat) ──────────────
        raw_row = mat[seq_row]   # full-length row

        def _rsum(lo, hi):
            lo, hi = max(0, lo), min(s, hi)
            return float(raw_row[lo:hi].sum()) if hi > lo else 0.0

        ext_s   = _rsum(0, NUM_IMAGE_TOKENS)
        wrist_s = _rsum(NUM_IMAGE_TOKENS, TOTAL_IMAGE_TOKENS)
        pad_s   = _rsum(TOTAL_IMAGE_TOKENS, TEXT_START_IDX)
        text_s  = _rsum(TEXT_START_IDX, text_real_end)
        act_s   = _rsum(action_start, action_end)
        total   = ext_s + wrist_s + pad_s + text_s + act_s + 1e-10

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("→ Ext camera",   f"{ext_s   / total:.1%}")
        c2.metric("→ Wrist camera", f"{wrist_s / total:.1%}")
        c3.metric("→ Padding",      f"{pad_s   / total:.1%}")
        c4.metric("→ Text",         f"{text_s  / total:.1%}")
        c5.metric("→ Action suffix",f"{act_s   / total:.1%}")
    else:
        st.info("Selected token is outside the current view region.")

    # ── Per-token entropy expander ────────────────────────────────────────────
    with st.expander("Per-token entropy across real text tokens"):
        if not real_texts:
            st.info("No real text tokens available.")
        else:
            ent_rows: list[float] = []
            for ti in range(n_real):
                si = TEXT_START_IDX + ti
                if si >= s:
                    break
                row_v = mat[si].astype(np.float64)
                ent_rows.append(float(-np.sum(row_v * np.log(row_v + 1e-10))))

            fig3 = go.Figure(go.Bar(
                x=token_labels[:len(ent_rows)],
                y=ent_rows,
                marker_color="coral",
                hovertemplate="%{x}: %{y:.3f} nats<extra></extra>",
            ))
            fig3.update_layout(
                title="Row entropy per real text token (lower = more focused)",
                height=240,
                margin=dict(l=40, r=20, t=40, b=80),
                xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                yaxis=dict(title="Entropy (nats)"),
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="white",
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── Stats panel ───────────────────────────────────────────────────────────
    with st.expander("Layer statistics"):
        text_end_clamp = min(text_real_end, s)
        text_rows = mat[TEXT_START_IDX:text_end_clamp, TEXT_START_IDX:text_end_clamp]
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        col_a.metric("Seq len", f"{s}")
        col_b.metric("Real text tokens", f"{n_real}")
        col_c.metric("Action start", f"{action_start}")
        if text_rows.size > 0:
            ent = float(-np.sum(text_rows * np.log(text_rows + 1e-10), axis=-1).mean())
            col_d.metric("Mean entropy (real text self-attn)", f"{ent:.2f}")
        else:
            col_d.metric("Mean entropy (real text self-attn)", "N/A")
        col_e.metric("Max weight (full mat)", f"{mat.max():.5f}")
