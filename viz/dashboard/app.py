"""Pi0.5 Attention Visualization Dashboard.

Launch:
    streamlit run viz/dashboard/app.py
    # or from project root with uv:
    uv run streamlit run viz/dashboard/app.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow running from the project root without installing the package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from viz.dashboard import loader as _loader
from viz.dashboard.views import action_view, attn_matrix, comparison, counterfactual, grid_heatmap, image_heatmap

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pi0.5 Attention Viz",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ATTN_H5_ROOT = os.path.join(_PROJECT_ROOT, "attn_h5")
CHECKPOINT_ROOT = os.path.join(_PROJECT_ROOT, "checkpoints/viz")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Pi0.5 Attention")
    st.markdown("---")

    mode = st.radio("Mode", ["Offline (HDF5)", "Online (Inference)"], index=0)

    if mode == "Offline (HDF5)":
        checkpoints = _loader.list_checkpoints(ATTN_H5_ROOT)
        if not checkpoints:
            st.error(
                f"No HDF5 data found in `{ATTN_H5_ROOT}`.\n\n"
                "Run conversion first:\n"
                "```\npython viz/convert_npy_to_h5.py\n```"
            )
            st.stop()

        checkpoint_a = st.selectbox("Checkpoint", checkpoints, key="ckpt_a")

        episodes_a = _loader.list_episodes(checkpoint_a, ATTN_H5_ROOT)
        if not episodes_a:
            st.error(f"No episodes found for checkpoint `{checkpoint_a}`.")
            st.stop()

        episode = st.selectbox("Episode", episodes_a, key="episode")
        frames = _loader.list_frames(checkpoint_a, episode, ATTN_H5_ROOT)
        frame_idx = st.selectbox("Frame", frames if frames else [0], key="frame")

        st.markdown("---")
        st.caption(f"HDF5 root: `{ATTN_H5_ROOT}`")

    else:
        # Online mode
        from viz.dashboard import inference as _inf

        online_checkpoints = _inf.list_online_checkpoints(CHECKPOINT_ROOT)
        if not online_checkpoints:
            st.warning(f"No checkpoints found in `{CHECKPOINT_ROOT}`.")

        online_ckpt = st.selectbox(
            "Checkpoint",
            online_checkpoints or ["checkpoints/viz/pi05_droid_pytorch"],
            key="online_ckpt",
        )
        online_ckpt_path = os.path.join(CHECKPOINT_ROOT, online_ckpt)

        st.markdown("**Input**")
        instruction_text = st.text_input(
            "Instruction",
            value="place the duck toy into the pink bowl",
            key="online_instruction",
        )

        dataset_choice = st.selectbox(
            "Dataset", ["duck", "pineapple"], key="online_dataset"
        )
        max_frames = {"duck": 90, "pineapple": 90}
        online_frame = st.slider(
            "Frame", 0, max_frames.get(dataset_choice, 90), 0, key="online_frame"
        )

        gpu_device = st.selectbox("GPU device", ["cuda:0", "cuda:1", "cpu"], key="gpu_device")

        run_btn = st.button("▶ Run Inference", type="primary")

        if run_btn:
            with st.spinner("Loading model and running inference…"):
                import sys as _sys
                _sys.path.insert(0, os.path.join(_PROJECT_ROOT, "viz"))
                if dataset_choice == "duck":
                    from attn_map import load_duck_example
                    example = load_duck_example(camera="left", index=online_frame)
                    example["prompt"] = instruction_text
                else:
                    from attn_pipeline import load_toy_example
                    from pathlib import Path as _Path
                    example = load_toy_example(
                        data_dir=_Path(os.path.join(_PROJECT_ROOT, "data/visualization/aawr_pineapple")),
                        index=online_frame,
                        camera="right",
                    )
                    example["prompt"] = instruction_text
                try:
                    policy = _inf.load_model(online_ckpt_path, device=gpu_device)
                    slice_dict = _inf.run_inference(policy, example)
                    st.session_state["online_data"] = slice_dict
                    st.success("Inference complete!")
                except Exception as e:
                    st.error(f"Inference failed: {e}")


# ── Main content area ──────────────────────────────────────────────────────────

if mode == "Offline (HDF5)":
    h5_path = _loader.h5_path(checkpoint_a, episode, frame_idx, ATTN_H5_ROOT)

    meta = _loader.load_meta(h5_path)
    images = _loader.load_images(h5_path)
    available_layers = _loader.list_layers_in_h5(h5_path)

    if not available_layers:
        st.error(f"No data found at `{h5_path}`. Check your conversion output.")
        st.stop()

    # Wire loader functions into the data dict
    def _make_load_t2i(path):
        def _fn(layer):
            return _loader.load_text_to_img(path, layer)
        return _fn

    def _make_load_full_all(path):
        def _fn(layer):
            return _loader.load_full_matrix_all_heads(path, layer)
        return _fn

    data = {
        "meta": meta,
        "images": images,
        "_load_t2i": _make_load_t2i(h5_path),
        "_load_full_all": _make_load_full_all(h5_path),
    }

    st.header(f"Checkpoint `{checkpoint_a}` · Episode `{episode}` · Frame `{frame_idx}`")
    if meta.get("instruction"):
        st.caption(f"**Instruction:** {meta['instruction']}")

    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "🔲 Grid Heatmap",
        "🖼 Image Heatmap",
        "📊 Attention Matrix",
        "🤖 Action View",
        "⚖ Compare",
    ])

    with tab0:
        grid_heatmap.render(data, available_layers)

    with tab1:
        image_heatmap.render(data, available_layers)

    with tab2:
        attn_matrix.render(data, available_layers)

    with tab3:
        action_view.render(data, available_layers)

    with tab4:
        comparison.render(
            attn_h5_root=ATTN_H5_ROOT,
            default_checkpoint=checkpoint_a,
            checkpoints=checkpoints,
        )

else:
    # Online mode
    st.header("Online Inference Mode")

    if "online_data" not in st.session_state:
        st.info("Configure the sidebar and click **▶ Run Inference** to begin.")
        st.stop()

    data = st.session_state["online_data"]
    available_layers = sorted(
        int(k.split("_")[1])
        for k in data.get("prefix", {})
        if k.startswith("layer_")
    )

    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "🔲 Grid Heatmap",
        "🖼 Image Heatmap",
        "📊 Attention Matrix",
        "🤖 Action View",
        "🔀 Counterfactual",
    ])

    with tab0:
        grid_heatmap.render(data, available_layers)
    with tab1:
        image_heatmap.render(data, available_layers)
    with tab2:
        attn_matrix.render(data, available_layers)
    with tab3:
        action_view.render(data, available_layers)
    with tab4:
        counterfactual.render(
            attn_h5_root=ATTN_H5_ROOT,
            checkpoints=_loader.list_checkpoints(ATTN_H5_ROOT),
            default_checkpoint="",
        )
    