"""Two-episode comparison view (top-level mode).

Selector
--------
Episodes from both ``success/`` and ``failure/`` for the chosen
dataset+camera+date are gathered and grouped by instruction. Each instruction
becomes a horizontally scrollable row of compact wrist-camera tiles:
  - tile border is **green** for success, **red** for failure;
  - the active selection (up to two) is overlaid with an **orange** outline;
  - a tiny button below toggles selection.

Comparison
----------
Once two episodes are picked, render side-by-side trajectory grids
(reusing ``trajectory._build_trajectory_figure``) plus three diagnostic
panels: per-frame focus concentration, head-aggregation sensitivity, and
a layer × aligned-frame divergence heatmap.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from viz.dashboard import loader as _loader
from viz.dashboard import loader_results as _rl
from viz.dashboard.views.grid_heatmap import (
    _ALL_AGG_NAMES,
    _SIMPLE_AGGS,
    _make_count_fn,
    _make_topk_fn,
)
from viz.dashboard.views.trajectory import (
    _build_trajectory_figure,
    _fig_to_bytes,
)

# Side-by-side view halves the available width per episode, so render
# fewer frames per row than the single-episode trajectory tab.
CHUNK_SIZE = 5
DEFAULT_LAYERS = [0, 4, 8, 12, 16]

THUMB_PX = 88           # rendered tile width/height in CSS pixels
TILE_MIN_PX = 100       # column min-width (so flex doesn't shrink them)
SELECTOR_KEY = "epcmp_selected"   # list[tuple[outcome, date, episode]]
PAGE_KEY = "epcmp_page"           # "select" | "compare"
COMPARE_PAIR_KEY = "epcmp_pair"   # frozen pair while on compare page
N_SAMPLE_FRAMES = 8

OUTCOME_BORDER = {"success": "#3ecf8e", "failure": "#e74c3c"}


# ── Selector helpers ──────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def _episode_meta(root: str, outcome: str, date: str, episode: str) -> dict:
    """Load instruction + thumbnail + frame list for one episode."""
    frames = _rl.list_frames(root, outcome, date, episode)
    if not frames:
        return {"instruction": "(no frames)", "thumb_b64": "", "frames": []}

    h5p = _rl.h5_path_results(root, outcome, date, episode, frames[0])
    meta = _loader.load_meta(h5p)
    imgs = _loader.load_images(h5p)
    instr = (meta.get("instruction") or "").strip() or "(no instruction)"

    thumb = imgs.get("exterior")
    if thumb is None:
        thumb = imgs.get("wrist")
    if thumb is None:
        thumb_arr = np.full((THUMB_PX, THUMB_PX, 3), 40, dtype=np.uint8)
    else:
        thumb_arr = cv2.resize(thumb, (THUMB_PX, THUMB_PX))

    bgr = cv2.cvtColor(thumb_arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    thumb_b64 = base64.b64encode(buf.tobytes()).decode() if ok else ""

    return {"instruction": instr, "thumb_b64": thumb_b64, "frames": frames}


def _ep_key(outcome: str, date: str, episode: str) -> str:
    return f"{outcome}/{date}/{episode}"


def _toggle_selection(key: str) -> None:
    sel: list[str] = st.session_state.get(SELECTOR_KEY, [])
    if key in sel:
        sel.remove(key)
    else:
        if len(sel) >= 2:
            sel = sel[1:]   # FIFO drop
        sel.append(key)
    st.session_state[SELECTOR_KEY] = sel


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Horizontal-scroll rows */
        div[class*="st-key-epcmp-row-"] > div[data-testid="stHorizontalBlock"] {
            overflow-x: auto;
            flex-wrap: nowrap !important;
            padding-bottom: 4px;
        }
        div[class*="st-key-epcmp-row-"] [data-testid="column"] {
            min-width: %(min)dpx;
            flex: 0 0 auto !important;
            padding-left: 2px !important;
            padding-right: 2px !important;
        }
        /* Compact tiles: shrink button padding inside selector rows */
        div[class*="st-key-epcmp-row-"] button[kind] {
            padding: 2px 6px !important;
            min-height: 24px !important;
            font-size: 0.72rem !important;
            line-height: 1.1 !important;
        }
        /* Tile thumbnail with outcome-coloured border */
        .epcmp-thumb {
            width: %(px)dpx; height: %(px)dpx;
            border-radius: 6px;
            display: block;
            object-fit: cover;
            box-sizing: border-box;
        }
        .epcmp-thumb-success { border: 3px solid %(g)s; }
        .epcmp-thumb-failure { border: 3px solid %(r)s; }
        .epcmp-thumb-selected {
            outline: 3px solid #f0a500;
            outline-offset: 1px;
        }
        .epcmp-epname {
            font-family: monospace;
            font-size: 0.68rem;
            color: #ccc;
            margin: 2px 0 0 0;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            width: %(px)dpx;
        }
        </style>
        """ % {
            "min": TILE_MIN_PX, "px": THUMB_PX,
            "g": OUTCOME_BORDER["success"], "r": OUTCOME_BORDER["failure"],
        },
        unsafe_allow_html=True,
    )


def _render_selector(
    root: str, date: str, outcomes: list[str],
) -> list[tuple[str, str, str]]:
    """Render the selector across given outcomes; return selected keys
    as tuples (outcome, date, episode)."""
    if SELECTOR_KEY not in st.session_state:
        st.session_state[SELECTOR_KEY] = []
    _inject_css()

    # Discover episodes across outcomes
    all_eps: list[tuple[str, str, str, dict]] = []  # (outcome, date, ep, info)
    with st.spinner(f"Loading episode previews for date `{date}`…"):
        for oc in outcomes:
            for ep in _rl.list_episodes(root, oc, date):
                info = _episode_meta(root, oc, date, ep)
                if info["frames"]:
                    all_eps.append((oc, date, ep, info))

    if not all_eps:
        st.warning("No episodes found for this dataset/camera/date.")
        return []

    # Group by instruction
    groups: dict[str, list[tuple[str, str, str, dict]]] = {}
    for tup in all_eps:
        groups.setdefault(tup[3]["instruction"], []).append(tup)

    selected: list[str] = st.session_state[SELECTOR_KEY]

    # Header / status
    h1, h2, h3 = st.columns([4, 2, 1])
    with h1:
        st.markdown(
            f"**Selected ({len(selected)}/2):** "
            + (", ".join(f"`{s}`" for s in selected) if selected else "_none_"),
        )
    with h2:
        st.caption(
            f"🟩 success · 🟥 failure · 🟧 selected — total {len(all_eps)} episodes"
        )
    with h3:
        if selected and st.button("Clear", key="epcmp_clear"):
            st.session_state[SELECTOR_KEY] = []
            st.rerun()

    # Render each instruction row
    for gi, (instruction, eps) in enumerate(sorted(groups.items())):
        st.markdown(
            f"<div style='font-size:0.85rem;color:#aaa;margin:8px 0 4px 0'>"
            f"<b>{instruction}</b> · {len(eps)} episode(s)</div>",
            unsafe_allow_html=True,
        )
        row = st.container(key=f"epcmp-row-{gi}")
        with row:
            cols = st.columns(len(eps), gap="small")
            for ci, (oc, dt, ep, info) in enumerate(eps):
                key = _ep_key(oc, dt, ep)
                is_sel = key in selected
                outcome_cls = f"epcmp-thumb-{oc}"
                sel_cls = " epcmp-thumb-selected" if is_sel else ""
                with cols[ci]:
                    st.markdown(
                        f'<img class="epcmp-thumb {outcome_cls}{sel_cls}" '
                        f'src="data:image/png;base64,{info["thumb_b64"]}" '
                        f'title="{oc}: {ep}" />'
                        f'<div class="epcmp-epname">{ep}</div>',
                        unsafe_allow_html=True,
                    )
                    st.button(
                        "✓" if is_sel else "Select",
                        key=f"epcmp-btn-{gi}-{ci}",
                        on_click=_toggle_selection, args=(key,),
                        use_container_width=True,
                        type="primary" if is_sel else "secondary",
                    )

    return [tuple(s.split("/", 2)) for s in selected]  # type: ignore[return-value]


# ── Statistics helpers ────────────────────────────────────────────────────────

def _focus(attn: np.ndarray) -> float:
    p = attn / (attn.sum() + 1e-10)
    h = -float(np.sum(p * np.log(p + 1e-10)))
    return float(1.0 - h / np.log(p.size))


def _attn_per_frame(
    root: str, outcome: str, date: str, episode: str,
    frames: list[int], layer: int, agg_fn,
    tok_idx: int | None, camera: str, attn_type: str,
) -> dict:
    """Load attention slices for one episode/layer.

    camera     — "exterior" or "wrist": which RGB image to overlay on.
    attn_type  — "text" (text→image, requires tok_idx) or "action"
                 (action→image, mean over 8 decode steps).

    Returns dict with:
        attn: {frame → (512,)}                       aggregated to a single map
        imgs: {frame → ndarray}                       chosen-camera image
        t2i:  {(frame, layer) → (n_heads, n_text, 512) | None}
              For attn_type="action" the n_text axis is fake-length-1
              (mean over 8 steps) so trajectory.py's renderer still works.
    """
    out_attn: dict[int, np.ndarray | None] = {}
    out_imgs: dict[int, np.ndarray | None] = {}
    out_t2i: dict[tuple[int, int], np.ndarray | None] = {}
    for f in frames:
        path = _rl.h5_path_results(root, outcome, date, episode, f)
        out_imgs[f] = _loader.load_images(path).get(camera)

        if attn_type == "action":
            a2i = _loader.load_action_to_img(path, layer)   # (n_heads, 8, 512) | None
            if a2i is None:
                out_t2i[(f, layer)] = None
                out_attn[f] = None
                continue
            heads = a2i.mean(axis=1)                        # (n_heads, 512)
            out_t2i[(f, layer)] = heads[:, np.newaxis, :]   # fake (n_heads, 1, 512)
            out_attn[f] = agg_fn(heads)
        else:
            t2i = _loader.load_text_to_img(path, layer)     # (n_heads, n_text, 512)
            out_t2i[(f, layer)] = t2i
            if t2i is None:
                out_attn[f] = None
                continue
            heads = (t2i.mean(axis=1) if tok_idx is None
                     else t2i[:, min(tok_idx, t2i.shape[1] - 1), :])
            out_attn[f] = agg_fn(heads)
    return {"attn": out_attn, "imgs": out_imgs, "t2i": out_t2i}


def _aligned_indices(n_a: int, n_b: int, k: int) -> tuple[list[int], list[int]]:
    k = min(k, n_a, n_b)
    if k <= 0:
        return [], []
    return (
        np.linspace(0, n_a - 1, k).round().astype(int).tolist(),
        np.linspace(0, n_b - 1, k).round().astype(int).tolist(),
    )


# ── Comparison renderer ───────────────────────────────────────────────────────

def _render_comparison(
    root: str,
    a: tuple[str, str, str], b: tuple[str, str, str],
) -> None:
    oa, da, ep_a = a
    ob, db, ep_b = b
    info_a = _episode_meta(root, oa, da, ep_a)
    info_b = _episode_meta(root, ob, db, ep_b)
    frames_a, frames_b = info_a["frames"], info_b["frames"]
    if not frames_a or not frames_b:
        st.error("One of the selected episodes has no frames.")
        return

    first_h5 = _rl.h5_path_results(root, oa, da, ep_a, frames_a[0])
    all_layers = _loader.list_layers_in_h5(first_h5)
    if not all_layers:
        st.error("No attention data in episode A.")
        return

    # ── Controls ──────────────────────────────────────────────────────────
    c0, c1, c2 = st.columns([2, 2, 2])
    with c0:
        attn_type_label = st.radio(
            "Attention type",
            ["text → image", "action → image"],
            horizontal=True, key="epcmp_attn_type",
        )
        attn_type = "action" if attn_type_label.startswith("action") else "text"
    with c1:
        camera = st.radio("Camera (heatmap)", ["exterior", "wrist"],
                          horizontal=True, key="epcmp_cam_h")
    with c2:
        agg_names = [n for n in _ALL_AGG_NAMES if n != "All heads"]
        agg = st.selectbox("Head aggregation", agg_names,
                           index=agg_names.index("Max"), key="epcmp_agg")
    c3, c4 = st.columns([3, 2])
    with c3:
        default_layers = [l for l in DEFAULT_LAYERS if l in all_layers] or all_layers[:5]
        layers = st.multiselect("Layers", all_layers, default=default_layers,
                                key="epcmp_layers")
    with c4:
        n_samples = st.slider("Sample frames", 2, 16, N_SAMPLE_FRAMES,
                              key="epcmp_nsamp")

    if not layers:
        st.info("Select at least one layer.")
        return

    if agg == "Top-K Focused":
        agg_fn = _make_topk_fn(st.slider("K heads", 1, 8, 4, key="epcmp_topk"))
    elif agg == "Count Above Threshold":
        agg_fn = _make_count_fn(st.slider("Top %", 1, 50, 10, key="epcmp_pct"))
    else:
        agg_fn = _SIMPLE_AGGS[agg]

    # ── Token mode (only for text→image) ──────────────────────────────────
    tok_idx: int | None = None
    if attn_type == "action":
        selected_tok_label = "action (mean over 8 steps)"
    else:
        meta_a = _loader.load_meta(first_h5)
        n_real = meta_a.get("n_real_tokens", len(meta_a.get("token_texts", [])))
        real_texts = meta_a.get("token_texts", [])[:n_real]

        tok_mode = st.radio(
            "Token mode", ["All text tokens (mean)", "Single token"],
            horizontal=True, key="epcmp_tokmode",
        )
        selected_tok_label = "all tokens (mean)"
        if tok_mode == "Single token" and real_texts:
            labels = [t.replace("▁", " ").strip() or f"[{i}]"
                      for i, t in enumerate(real_texts)]
            selected_tok_label = st.selectbox(
                "Token", labels, index=min(3, len(labels) - 1), key="epcmp_tok",
            )
            tok_idx = labels.index(selected_tok_label)

    # ── Sample frames evenly ──────────────────────────────────────────────
    si_a, si_b = _aligned_indices(len(frames_a), len(frames_b), n_samples)
    sample_a = [frames_a[i] for i in si_a]
    sample_b = [frames_b[i] for i in si_b]

    pack_a: dict[int, dict] = {}
    pack_b: dict[int, dict] = {}
    with st.spinner("Loading attention slices for both episodes…"):
        for layer in layers:
            pack_a[layer] = _attn_per_frame(root, oa, da, ep_a, sample_a, layer, agg_fn,
                                            tok_idx, camera, attn_type)
            pack_b[layer] = _attn_per_frame(root, ob, db, ep_b, sample_b, layer, agg_fn,
                                            tok_idx, camera, attn_type)

    # In action mode the t2i array is fake-length-1, so the renderer must
    # index axis 1 with 0 instead of None (None means "mean over n_text").
    render_tok_idx = 0 if attn_type == "action" else tok_idx

    # ── Side-by-side trajectory grids ─────────────────────────────────────
    st.markdown(
        f'#### Trajectory grids — `{selected_tok_label}` — {camera} — {agg} '
        f'· *{attn_type_label}*'
    )
    g1, g2 = st.columns(2)
    for col, oc, ep, sample, pack, info in (
        (g1, oa, ep_a, sample_a, pack_a, info_a),
        (g2, ob, ep_b, sample_b, pack_b, info_b),
    ):
        with col:
            badge = "🟩 success" if oc == "success" else "🟥 failure"
            st.markdown(f"**`{ep}`** · {badge} · {len(info['frames'])} frames total")
            st.caption(f"📝 {info['instruction']}")
            t2i_data = {(f, l): pack[l]["t2i"][(f, l)] for l in layers for f in sample}
            imgs = {f: pack[layers[0]]["imgs"][f] for f in sample}
            for cs in range(0, len(sample), CHUNK_SIZE):
                chunk = sample[cs:cs + CHUNK_SIZE]
                fig = _build_trajectory_figure(
                    frames=chunk, layers=layers,
                    t2i_data=t2i_data, imgs=imgs,
                    tok_idx=render_tok_idx, camera=camera, agg_fn=agg_fn,
                )
                st.image(_fig_to_bytes(fig), use_container_width=True)

    # ── Per-frame focus concentration ─────────────────────────────────────
    st.markdown("#### Per-frame focus concentration")
    st.caption(
        "1 − normalized entropy of the aggregated 256-patch attention. "
        "Higher = attention more localised. X-axis is normalised episode progress."
    )
    layer_for_stats = st.selectbox(
        "Layer for stats", layers, index=min(1, len(layers) - 1),
        key="epcmp_stat_layer",
    )

    fig_focus = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=("Exterior half", "Wrist half"),
        horizontal_spacing=0.08,
    )
    for ep, sample, pack, n_total, color in (
        (ep_a, sample_a, pack_a, len(info_a["frames"]), "#4c9be8"),
        (ep_b, sample_b, pack_b, len(info_b["frames"]), "#f0a500"),
    ):
        attn_d = pack[layer_for_stats]["attn"]
        ext_focus, wrist_focus, prog = [], [], []
        for f in sample:
            a_v = attn_d[f]
            if a_v is None:
                continue
            ext_focus.append(_focus(a_v[:256]))
            wrist_focus.append(_focus(a_v[256:512]))
            prog.append(f / max(n_total - 1, 1))
        fig_focus.add_trace(go.Scatter(
            x=prog, y=ext_focus, mode="lines+markers",
            name=f"{ep} ext", line=dict(color=color)), row=1, col=1)
        fig_focus.add_trace(go.Scatter(
            x=prog, y=wrist_focus, mode="lines+markers",
            name=f"{ep} wrist", line=dict(color=color, dash="dash")), row=1, col=2)
    fig_focus.update_layout(
        height=320, margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(title="Episode progress"),
        xaxis2=dict(title="Episode progress"),
    )
    st.plotly_chart(fig_focus, use_container_width=True, key="epcmp_focus")

    # ── Head-aggregation sensitivity ──────────────────────────────────────
    st.markdown("#### Head-aggregation sensitivity")
    st.caption("Per layer: mean cosine distance between Mean- and Max-aggregated heads, averaged over sampled frames.")
    sens: dict[str, list] = {ep_a: [], ep_b: []}
    for layer in layers:
        for ep, sample, pack in ((ep_a, sample_a, pack_a), (ep_b, sample_b, pack_b)):
            ds = []
            for f in sample:
                t2i_arr = pack[layer]["t2i"].get((f, layer))
                if t2i_arr is None:
                    continue
                if attn_type == "action":
                    heads = t2i_arr[:, 0, :]
                else:
                    heads = (t2i_arr.mean(axis=1) if tok_idx is None
                             else t2i_arr[:, min(tok_idx, t2i_arr.shape[1] - 1), :])
                v_mean = heads.mean(axis=0)
                v_max = heads.max(axis=0)
                cs = float(np.dot(v_mean, v_max) / (np.linalg.norm(v_mean) * np.linalg.norm(v_max) + 1e-10))
                ds.append(1.0 - cs)
            sens[ep].append(float(np.mean(ds)) if ds else None)

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(x=layers, y=sens[ep_a], mode="lines+markers",
                                  name=ep_a, line=dict(color="#4c9be8")))
    fig_sens.add_trace(go.Scatter(x=layers, y=sens[ep_b], mode="lines+markers",
                                  name=ep_b, line=dict(color="#f0a500")))
    fig_sens.update_layout(
        height=280, margin=dict(l=50, r=20, t=20, b=40),
        xaxis_title="Layer", yaxis_title="1 − cos(Mean, Max)",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_sens, use_container_width=True, key="epcmp_sens")

    # ── Cross-episode divergence heatmap ──────────────────────────────────
    st.markdown("#### Cross-episode divergence")
    st.caption(
        "Per (layer × aligned sample): cosine distance between A and B aggregated "
        f"attention on the **{camera}** half. Columns are evenly-spaced sample "
        "indices, so episodes of different length still align at progress 0…1."
    )
    half = slice(0, 256) if camera == "exterior" else slice(256, 512)
    n_cols = min(len(sample_a), len(sample_b))
    div = np.full((len(layers), n_cols), np.nan)
    for li, layer in enumerate(layers):
        attn_a, attn_b = pack_a[layer]["attn"], pack_b[layer]["attn"]
        for k in range(n_cols):
            va = attn_a.get(sample_a[k]); vb = attn_b.get(sample_b[k])
            if va is None or vb is None:
                continue
            va_h, vb_h = va[half], vb[half]
            cs = float(np.dot(va_h, vb_h) / (np.linalg.norm(va_h) * np.linalg.norm(vb_h) + 1e-10))
            div[li, k] = 1.0 - cs

    fig_div = go.Figure(go.Heatmap(
        z=div,
        x=[f"k{k}" for k in range(n_cols)],
        y=[f"L{l}" for l in layers],
        colorscale="Viridis",
        colorbar=dict(title="1 − cos"),
    ))
    fig_div.update_layout(
        height=max(220, 36 * len(layers) + 80),
        margin=dict(l=60, r=20, t=20, b=40),
        xaxis_title="Aligned sample index",
        yaxis_title="Layer",
    )
    st.plotly_chart(fig_div, use_container_width=True, key="epcmp_div")


# ── Public entrypoint ────────────────────────────────────────────────────────

def render(root: str, date: str, outcomes: list[str]) -> None:
    """Top-level renderer for the Compare Episodes mode."""
    page = st.session_state.get(PAGE_KEY, "select")

    if page == "compare":
        pair = st.session_state.get(COMPARE_PAIR_KEY)
        if not pair or len(pair) != 2:
            st.session_state[PAGE_KEY] = "select"
            st.rerun()
            return

        back_col, title_col = st.columns([1, 6])
        with back_col:
            if st.button("← Back to selection", key="epcmp_back",
                         use_container_width=True):
                st.session_state[PAGE_KEY] = "select"
                st.rerun()
        a, b = pair[0], pair[1]
        with title_col:
            st.markdown(f"### Comparing `{a[2]}` ({a[0]}) ↔ `{b[2]}` ({b[0]})")
        _render_comparison(root, a, b)
        return

    # ── Selection page ────────────────────────────────────────────────────
    selected = _render_selector(root, date, outcomes)

    n_sel = len(selected)
    with st.sidebar:
        st.markdown("---")
        st.caption(f"**Compare Episodes** — {n_sel}/2 selected")
        if st.button(
            "Confirm & Compare →",
            key="epcmp_confirm",
            type="primary",
            disabled=(n_sel != 2),
            use_container_width=True,
        ):
            st.session_state[COMPARE_PAIR_KEY] = list(selected)
            st.session_state[PAGE_KEY] = "compare"
            st.rerun()

    st.markdown("---")
    if n_sel < 2:
        st.info(f"Select **two** episodes above, then click **Confirm & Compare** in the sidebar ({n_sel}/2).")
    else:
        st.success("Two episodes selected — click **Confirm & Compare** in the sidebar to continue.")
