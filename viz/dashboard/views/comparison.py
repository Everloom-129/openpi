"""Tab 4: Side-by-side comparison.

Three modes:
  - Checkpoint A vs B  (same episode/frame, different checkpoint)
  - Episode A vs B     (same checkpoint, two different episodes)
  - Frame A vs B       (same checkpoint+episode, two frame sliders)
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from viz.dashboard import loader as _loader
from viz.dashboard.views import image_heatmap as _ih


def _build_data(checkpoint: str, episode: str, frame: int, attn_h5_root: str) -> dict:
    """Build a data dict wired to the HDF5 loader functions."""
    path = _loader.h5_path(checkpoint, episode, frame, attn_h5_root)
    meta = _loader.load_meta(path)
    images = _loader.load_images(path)
    avail = _loader.list_layers_in_h5(path)

    def load_t2i(layer):
        return _loader.load_text_to_img(path, layer)

    def load_full_all(layer):
        return _loader.load_full_matrix_all_heads(path, layer)

    return {
        "meta": meta,
        "images": images,
        "_load_t2i": load_t2i,
        "_load_full_all": load_full_all,
        "_available_layers": avail,
        "_path": path,
    }


def _render_half(
    label: str,
    checkpoint: str,
    episode: str,
    frame: int,
    attn_h5_root: str,
    key_suffix: str,
) -> None:
    """Render one half of the comparison view."""
    st.markdown(f"### {label}")
    st.caption(f"`{checkpoint}` / `{episode}` / frame `{frame}`")

    data = _build_data(checkpoint, episode, frame, attn_h5_root)
    avail = data["_available_layers"]

    if not avail:
        st.warning(f"No data found at `{data['_path']}`.")
        return

    # Override session state keys to avoid collision between the two columns
    # by temporarily monkeypatching the key prefix — we do this with st.session_state
    # namespacing via key_suffix passed to sub-renders.
    # Simple approach: just render the heatmap view with unique widget keys.
    _render_heatmap_compact(data, avail, key_suffix)


def _render_heatmap_compact(data: dict, available_layers: list[int], key_suffix: str) -> None:
    """Compact version of image_heatmap.render with namespaced widget keys."""
    meta = data.get("meta", {})
    token_texts: list[str] = meta.get("token_texts", [])
    instruction: str = meta.get("instruction", "")

    if instruction:
        st.caption(f"**Instruction:** {instruction}")

    col_l, col_h = st.columns([2, 3])
    with col_l:
        layer = st.select_slider(
            "Layer",
            options=available_layers,
            value=available_layers[min(2, len(available_layers) - 1)],
            key=f"cmp_layer_{key_suffix}",
        )
    with col_h:
        head_opts = ["max", "mean"] + list(range(8))
        head_labels = ["Max", "Mean"] + [f"H{i}" for i in range(8)]
        hl = st.radio(
            "Head",
            head_labels,
            index=0,
            horizontal=True,
            key=f"cmp_head_{key_suffix}",
        )
        head_sel = head_opts[head_labels.index(hl)]

    # Load t2i
    load_fn = data.get("_load_t2i")
    t2i = load_fn(layer) if load_fn else None
    if t2i is None:
        st.warning("No attention data.")
        return

    n_text = t2i.shape[1]
    if len(token_texts) < n_text:
        token_texts = token_texts + [f"tok_{i}" for i in range(len(token_texts), n_text)]
    token_texts = token_texts[:n_text]

    # Token selector (compact: selectbox instead of pills)
    tok_labels = [f"{i}: {t.replace('▁', ' ').strip() or f'[{i}]'}" for i, t in enumerate(token_texts)]
    sel = st.selectbox("Token", tok_labels, key=f"cmp_tok_{key_suffix}")
    tok_idx = int(sel.split(":")[0])

    # Aggregate
    if head_sel == "max":
        agg = t2i.max(axis=0)
    elif head_sel == "mean":
        agg = t2i.mean(axis=0)
    else:
        agg = t2i[int(head_sel)]

    attn_512 = agg[tok_idx]

    # Images
    images = data.get("images", {})
    _ext = images.get("exterior")
    _wrist = images.get("wrist")
    ext_img = _ext if _ext is not None else np.full((224, 224, 3), 30, dtype=np.uint8)
    wrist_img = _wrist if _wrist is not None else np.full((224, 224, 3), 30, dtype=np.uint8)

    ext_h = _ih._upsample_heatmap(attn_512, "exterior")
    wrist_h = _ih._upsample_heatmap(attn_512, "wrist")

    tok_name = token_texts[tok_idx].replace("▁", " ").strip() or f"tok_{tok_idx}"
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _ih._make_overlay_figure(ext_img, ext_h, f"Ext — '{tok_name}'"),
            use_container_width=True,
            key=f"cmp_ext_{key_suffix}_{tok_idx}_{layer}",
        )
    with c2:
        st.plotly_chart(
            _ih._make_overlay_figure(wrist_img, wrist_h, f"Wrist — '{tok_name}'"),
            use_container_width=True,
            key=f"cmp_wrist_{key_suffix}_{tok_idx}_{layer}",
        )


def render(
    attn_h5_root: str,
    default_checkpoint: str,
    checkpoints: list[str],
) -> None:
    """Render Tab 4: Comparison view."""

    if not checkpoints:
        st.info("No checkpoints found in the attention HDF5 directory.")
        return

    mode = st.radio(
        "Comparison mode",
        ["Checkpoint A vs B", "Episode A vs B", "Frame A vs B"],
        horizontal=True,
        key="cmp_mode",
    )

    episodes = _loader.list_episodes(default_checkpoint or checkpoints[0], attn_h5_root)
    default_episode = episodes[0] if episodes else "episode_0"
    frames = _loader.list_frames(default_checkpoint, default_episode, attn_h5_root)
    default_frame = frames[0] if frames else 0

    if mode == "Checkpoint A vs B":
        ckpt_a = st.selectbox("Checkpoint A", checkpoints, index=0, key="cmp_ckpt_a")
        ckpt_b_opts = [c for c in checkpoints if c != ckpt_a] or checkpoints
        ckpt_b = st.selectbox("Checkpoint B", ckpt_b_opts, index=0, key="cmp_ckpt_b")
        ep = st.selectbox("Episode", episodes or ["episode_0"], key="cmp_ep_shared")
        frm_opts = frames or [0]
        frm = st.selectbox("Frame", frm_opts, key="cmp_frm_shared")

        col_a, col_b = st.columns(2)
        with col_a:
            _render_half(f"Checkpoint {ckpt_a}", ckpt_a, ep, frm, attn_h5_root, "a")
        with col_b:
            _render_half(f"Checkpoint {ckpt_b}", ckpt_b, ep, frm, attn_h5_root, "b")

    elif mode == "Episode A vs B":
        ckpt = st.selectbox("Checkpoint", checkpoints, key="cmp_ckpt_shared")
        eps_a = _loader.list_episodes(ckpt, attn_h5_root) or ["episode_0"]
        ep_a = st.selectbox("Episode A", eps_a, key="cmp_ep_a")
        ep_b = st.selectbox("Episode B", eps_a, index=min(1, len(eps_a) - 1), key="cmp_ep_b")
        frms = _loader.list_frames(ckpt, ep_a, attn_h5_root) or [0]
        frm = st.selectbox("Frame", frms, key="cmp_frm_ep")

        col_a, col_b = st.columns(2)
        with col_a:
            _render_half(f"Episode {ep_a}", ckpt, ep_a, frm, attn_h5_root, "a")
        with col_b:
            _render_half(f"Episode {ep_b}", ckpt, ep_b, frm, attn_h5_root, "b")

    else:  # Frame A vs B
        ckpt = st.selectbox("Checkpoint", checkpoints, key="cmp_ckpt_frm")
        ep = st.selectbox("Episode", _loader.list_episodes(ckpt, attn_h5_root) or ["episode_0"], key="cmp_ep_frm")
        frms = _loader.list_frames(ckpt, ep, attn_h5_root) or [0]
        col_fa, col_fb = st.columns(2)
        with col_fa:
            frm_a = st.selectbox("Frame A", frms, key="cmp_frm_a")
        with col_fb:
            frm_b = st.selectbox("Frame B", frms, index=min(1, len(frms) - 1), key="cmp_frm_b")

        col_a, col_b = st.columns(2)
        with col_a:
            _render_half(f"Frame {frm_a}", ckpt, ep, frm_a, attn_h5_root, "a")
        with col_b:
            _render_half(f"Frame {frm_b}", ckpt, ep, frm_b, attn_h5_root, "b")
