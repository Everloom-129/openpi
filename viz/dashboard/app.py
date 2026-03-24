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
from viz.dashboard import loader_results as _rl
from viz.dashboard.views import action_view, attn_matrix, cag_view, ckpt_compare, comparison, counterfactual, grid_heatmap, image_heatmap, image_saliency, trajectory


def _save_online_h5(slice_dict: dict, h5_path: str) -> None:
    """Write an online-inference slice_dict to HDF5.

    Works purely from the already-processed slice_dict — no raw buffer needed.
    Schema mirrors attn_h5_writer so the file is loadable by loader.py.
    """
    import h5py
    import numpy as np

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    meta = slice_dict.get("meta", {})
    images = slice_dict.get("images", {})
    prefix = slice_dict.get("prefix", {})

    with h5py.File(h5_path, "w") as f:
        # /meta
        mg = f.create_group("meta")
        for k, v in meta.items():
            if isinstance(v, list):
                mg.attrs[k] = [s.encode() if isinstance(s, str) else s for s in v]
            elif isinstance(v, str):
                mg.attrs[k] = v
            else:
                mg.attrs[k] = v

        # /images
        ig = f.create_group("images")
        for cam, img in images.items():
            if img is not None:
                ig.create_dataset(cam, data=img, compression="gzip", compression_opts=4)

        # /prefix/layer_{i}/text_to_img + full
        pg = f.create_group("prefix")
        for layer_key, layer_data in prefix.items():
            lg = pg.create_group(layer_key)
            for arr_key, arr in layer_data.items():
                if arr is not None:
                    lg.create_dataset(
                        arr_key,
                        data=np.asarray(arr, dtype=np.float32),
                        compression="gzip",
                        compression_opts=4,
                    )

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pi0.5 Attention Viz",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ATTN_H5_ROOT = os.path.join(_PROJECT_ROOT, "attn_h5")
CHECKPOINT_ROOT = os.path.join(_PROJECT_ROOT, "checkpoints/viz")
RESULTS_ROOT = os.environ.get(
    "RESULTS_ROOT",
    "/data3/tonyw/toy_cube_benchmark/pi05_vis/cube_gold",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Pi0.5 Attention")
    st.markdown("---")

    mode = st.radio("Mode", ["Offline (HDF5)", "Results (Benchmark)", "Online (Inference)", "Compare (Online)"], index=0)

    if mode == "Offline (HDF5)":
        checkpoints = _loader.list_checkpoints(ATTN_H5_ROOT)
        if not checkpoints:
            st.error(
                f"No HDF5 data found in `{ATTN_H5_ROOT}`.\n\n"
                "Run the batch pipeline first:\n"
                "```\npython viz/pipeline.py <DATA_ROOT> <RESULTS_ROOT>\n```\n"
                "Or use **Online (Inference)** mode and click **💾 Save to HDF5**."
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

    elif mode == "Results (Benchmark)":
        res_camera = st.radio("Camera", ["right", "left"], horizontal=True, key="res_camera")
        _res_root = os.path.join(RESULTS_ROOT, res_camera)

        outcomes = _rl.list_outcomes(_res_root)
        if not outcomes:
            st.error(f"No data found in `{_res_root}`.\n\nSet `RESULTS_ROOT` env var or run the pipeline first.")
            st.stop()

        res_outcome = st.selectbox("Outcome", outcomes, key="res_outcome")

        res_dates = _rl.list_dates(_res_root, res_outcome)
        if not res_dates:
            st.error("No dates found.")
            st.stop()
        res_date = st.selectbox("Date", res_dates, key="res_date")

        res_episodes = _rl.list_episodes(_res_root, res_outcome, res_date)
        if not res_episodes:
            st.error("No episodes found.")
            st.stop()
        ep_labels = [
            f"{'✓ ' if _rl.is_complete(_res_root, res_outcome, res_date, ep) else '○ '}{ep}"
            for ep in res_episodes
        ]
        res_ep_idx = st.selectbox(
            "Episode", range(len(res_episodes)),
            format_func=lambda i: ep_labels[i], key="res_ep_idx"
        )
        res_episode = res_episodes[res_ep_idx]

        res_all_frames = _rl.list_frames(_res_root, res_outcome, res_date, res_episode)
        if not res_all_frames:
            st.error("No frames found.")
            st.stop()

        res_frame = st.selectbox("Frame (single-frame tabs)", res_all_frames, key="res_frame")
        res_cf_slugs = _rl.list_cf_slugs(_res_root, res_outcome, res_date, res_episode, res_all_frames[0])

        st.markdown("---")
        st.caption(f"Camera: **{res_camera}** · `{res_outcome}/{res_date}/{res_episode}`")
        if res_cf_slugs:
            st.caption(f"CF variants: {', '.join(f'`{s}`' for s in res_cf_slugs)}")

    elif mode == "Online (Inference)":
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
        input_source = st.radio(
            "Source", ["Episode", "Upload Image"], horizontal=True, key="online_input_source",
        )

        if input_source == "Upload Image":
            uploaded_exterior = st.file_uploader(
                "Exterior camera image", type=["jpg", "jpeg", "png"], key="upload_ext",
            )
            uploaded_wrist = st.file_uploader(
                "Wrist camera image (optional)", type=["jpg", "jpeg", "png"], key="upload_wrist",
            )
            instruction_text = st.text_input("Instruction", key="online_instruction_upload")

            # Show previews
            if uploaded_exterior:
                st.image(uploaded_exterior, caption="Exterior", width=200)
            if uploaded_wrist:
                st.image(uploaded_wrist, caption="Wrist", width=200)

            # Placeholders for episode-only variables
            online_camera = "right"
            online_frame = 0
            _ep_dir = None
            _has_recordings = False
        else:
            uploaded_exterior = None
            uploaded_wrist = None

            _EXAMPLE_DIR = os.path.join(_PROJECT_ROOT, "data/example")

            # List all episode subdirectories under data/example/
            _example_episodes = sorted(
                d for d in os.listdir(_EXAMPLE_DIR)
                if os.path.isdir(os.path.join(_EXAMPLE_DIR, d))
            ) if os.path.isdir(_EXAMPLE_DIR) else []

            online_episode = st.selectbox("Episode", _example_episodes or ["(none)"], key="online_episode")
            online_camera = st.radio(
                "Ext camera", ["right", "left"], horizontal=True, key="online_camera"
            )

            # Auto-detect episode structure and metadata
            _ep_dir = os.path.join(_EXAMPLE_DIR, online_episode)
            _has_recordings = os.path.isdir(os.path.join(_ep_dir, "recordings", "frames"))
            _frames_root = os.path.join(_ep_dir, "recordings", "frames") if _has_recordings else os.path.join(_ep_dir, "frames")
            _hand_dir = os.path.join(_frames_root, "hand_camera")
            _n_frames = len([f for f in os.listdir(_hand_dir) if f.endswith(".jpg")]) if os.path.isdir(_hand_dir) else 1
            _max_frame = max(0, _n_frames - 1)
            _instr_path = os.path.join(_ep_dir, "instruction.txt")
            _default_instr = open(_instr_path).read().strip() if os.path.exists(_instr_path) else ""

            instruction_text = st.text_input(
                "Instruction", value=_default_instr, key="online_instruction",
            )
            online_frame = st.slider("Frame", 0, _max_frame, 0, key="online_frame")

        def _available_gpu_devices() -> list[str]:
            try:
                import pynvml
                pynvml.nvmlInit()
                n = pynvml.nvmlDeviceGetCount()
                pynvml.nvmlShutdown()
                return ["Auto"] + [f"cuda:{i}" for i in range(n)] + ["cpu"]
            except Exception:
                return ["Auto", "cpu"]

        gpu_device_sel = st.selectbox("GPU device", _available_gpu_devices(), key="gpu_device")

        run_btn = st.button("▶ Run Inference", type="primary")

        if run_btn:
            with st.spinner("Loading model and running inference…"):
                import sys as _sys
                _sys.path.insert(0, os.path.join(_PROJECT_ROOT, "viz"))
                from pathlib import Path as _Path
                from PIL import Image as _PILImage

                if input_source == "Upload Image":
                    if not uploaded_exterior:
                        st.error("Please upload at least an exterior camera image.")
                        st.stop()
                    ext_img = np.array(_PILImage.open(uploaded_exterior).convert("RGB"))
                    if uploaded_wrist:
                        wrist_img = np.array(_PILImage.open(uploaded_wrist).convert("RGB"))
                    else:
                        # Use a black placeholder if no wrist image provided
                        wrist_img = np.zeros_like(ext_img)
                    example = {
                        "observation/exterior_image_1_left": ext_img,
                        "observation/wrist_image_left": wrist_img,
                        "observation/joint_position": np.zeros(7, dtype=np.float64),
                        "observation/gripper_position": np.zeros(1, dtype=np.float64),
                        "prompt": instruction_text,
                    }
                elif _has_recordings:
                    # DROID format: recordings/frames/{camera}/
                    from pipeline import load_example as _load_example
                    example = _load_example(
                        data_dir=_Path(_ep_dir),
                        index=online_frame,
                        camera=online_camera,
                    )
                    example["prompt"] = instruction_text
                else:
                    # Duck format: frames/{camera}/ directly
                    from attn_map import load_duck_example
                    example = load_duck_example(camera=online_camera, index=online_frame)
                    example["prompt"] = instruction_text
                try:
                    if gpu_device_sel == "Auto":
                        from attn_map import select_best_gpu as _sbg
                        _dev_id = _sbg()
                        gpu_device = f"cuda:{_dev_id}" if isinstance(_dev_id, int) else str(_dev_id)
                    else:
                        gpu_device = gpu_device_sel
                    policy = _inf.load_model(online_ckpt_path, device=gpu_device)
                    slice_dict = _inf.run_inference(policy, example)
                    st.session_state["online_data"] = slice_dict
                    st.success("Inference complete!")
                except Exception as e:
                    st.error(f"Inference failed: {e}")

    else:
        # ── Compare (Online) mode sidebar ─────────────────────────────────────
        from viz.dashboard import inference as _inf

        _cmp_checkpoints = _inf.list_online_checkpoints(CHECKPOINT_ROOT)
        if not _cmp_checkpoints:
            st.warning(f"No checkpoints found in `{CHECKPOINT_ROOT}`.")

        st.markdown("**Checkpoint A**")
        cmp_ckpt_a = st.selectbox(
            "Checkpoint A",
            _cmp_checkpoints or ["pi05_droid_pytorch"],
            key="cmp_ckpt_a",
            label_visibility="collapsed",
        )
        st.markdown("**Checkpoint B**")
        # Default B to second checkpoint if available, else same as A
        _cmp_default_b_idx = 1 if len(_cmp_checkpoints) > 1 else 0
        cmp_ckpt_b = st.selectbox(
            "Checkpoint B",
            _cmp_checkpoints or ["pi0_droid_pytorch"],
            index=_cmp_default_b_idx,
            key="cmp_ckpt_b",
            label_visibility="collapsed",
        )

        st.markdown("**Shared input**")
        _EXAMPLE_DIR = os.path.join(_PROJECT_ROOT, "data/example")
        _cmp_episodes = sorted(
            d for d in os.listdir(_EXAMPLE_DIR)
            if os.path.isdir(os.path.join(_EXAMPLE_DIR, d))
        ) if os.path.isdir(_EXAMPLE_DIR) else []

        cmp_episode = st.selectbox("Episode", _cmp_episodes or ["(none)"], key="cmp_episode")
        cmp_camera = st.radio("Ext camera", ["right", "left"], horizontal=True, key="cmp_camera")

        _cmp_ep_dir = os.path.join(_EXAMPLE_DIR, cmp_episode)
        _cmp_has_rec = os.path.isdir(os.path.join(_cmp_ep_dir, "recordings", "frames"))
        _cmp_frames_root = (
            os.path.join(_cmp_ep_dir, "recordings", "frames")
            if _cmp_has_rec else os.path.join(_cmp_ep_dir, "frames")
        )
        _cmp_hand_dir = os.path.join(_cmp_frames_root, "hand_camera")
        _cmp_n_frames = (
            len([f for f in os.listdir(_cmp_hand_dir) if f.endswith(".jpg")])
            if os.path.isdir(_cmp_hand_dir) else 1
        )
        _cmp_max_frame = max(0, _cmp_n_frames - 1)
        _cmp_instr_path = os.path.join(_cmp_ep_dir, "instruction.txt")
        _cmp_default_instr = (
            open(_cmp_instr_path).read().strip() if os.path.exists(_cmp_instr_path) else ""
        )

        cmp_instruction = st.text_input(
            "Instruction", value=_cmp_default_instr, key="cmp_instruction"
        )
        cmp_frame = st.slider("Frame", 0, _cmp_max_frame, 0, key="cmp_frame")

        def _cmp_gpu_devices() -> list[str]:
            try:
                import pynvml
                pynvml.nvmlInit()
                n = pynvml.nvmlDeviceGetCount()
                pynvml.nvmlShutdown()
                return ["Auto"] + [f"cuda:{i}" for i in range(n)] + ["cpu"]
            except Exception:
                return ["Auto", "cpu"]

        _gpus = _cmp_gpu_devices()
        st.markdown("**GPU assignment**")
        cg1, cg2 = st.columns(2)
        cmp_gpu_a = cg1.selectbox("A", _gpus, key="cmp_gpu_a", label_visibility="visible")
        cmp_gpu_b = cg2.selectbox("B", _gpus, key="cmp_gpu_b", label_visibility="visible")

        cmp_run_btn = st.button("▶ Run Both", type="primary", key="cmp_run")

        if cmp_run_btn:
            import sys as _sys
            _sys.path.insert(0, os.path.join(_PROJECT_ROOT, "viz"))
            from pathlib import Path as _Path

            def _resolve_gpu(sel: str) -> str:
                if sel != "Auto":
                    return sel
                try:
                    from attn_map import select_best_gpu as _sbg
                    d = _sbg()
                    return f"cuda:{d}" if isinstance(d, int) else str(d)
                except Exception:
                    return "cuda:0"

            def _load_ep(ep_dir: str, frame: int, camera: str, instr: str) -> dict:
                has_rec = os.path.isdir(os.path.join(ep_dir, "recordings", "frames"))
                if has_rec:
                    from pipeline import load_example as _lex
                    ex = _lex(data_dir=_Path(ep_dir), index=frame, camera=camera)
                else:
                    from attn_map import load_duck_example
                    ex = load_duck_example(camera=camera, index=frame)
                ex["prompt"] = instr
                return ex

            with st.spinner(f"Running inference on **{cmp_ckpt_a}**…"):
                try:
                    _ex = _load_ep(_cmp_ep_dir, cmp_frame, cmp_camera, cmp_instruction)
                    _pol_a = _inf.load_model(
                        os.path.join(CHECKPOINT_ROOT, cmp_ckpt_a),
                        device=_resolve_gpu(cmp_gpu_a),
                    )
                    st.session_state["cmp_data_a"] = _inf.run_inference(_pol_a, _ex)
                    st.session_state["cmp_label_a"] = cmp_ckpt_a
                except Exception as _e:
                    st.error(f"Checkpoint A failed: {_e}")

            with st.spinner(f"Running inference on **{cmp_ckpt_b}**…"):
                try:
                    _ex = _load_ep(_cmp_ep_dir, cmp_frame, cmp_camera, cmp_instruction)
                    _pol_b = _inf.load_model(
                        os.path.join(CHECKPOINT_ROOT, cmp_ckpt_b),
                        device=_resolve_gpu(cmp_gpu_b),
                    )
                    st.session_state["cmp_data_b"] = _inf.run_inference(_pol_b, _ex)
                    st.session_state["cmp_label_b"] = cmp_ckpt_b
                except Exception as _e:
                    st.error(f"Checkpoint B failed: {_e}")

            if "cmp_data_a" in st.session_state and "cmp_data_b" in st.session_state:
                st.success("Both models done — see main area.")


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

elif mode == "Results (Benchmark)":
    res_h5 = _rl.h5_path_results(_res_root, res_outcome, res_date, res_episode, res_frame)

    meta = _loader.load_meta(res_h5)
    images = _loader.load_images(res_h5)
    available_layers = _loader.list_layers_in_h5(res_h5)

    if not available_layers:
        st.error(f"No attention data at `{res_h5}`.")
        st.stop()

    def _make_load_t2i_res(path):
        def _fn(layer):
            return _loader.load_text_to_img(path, layer)
        return _fn

    def _make_load_full_res(path):
        def _fn(layer):
            return _loader.load_full_matrix_all_heads(path, layer)
        return _fn

    data = {
        "meta": meta,
        "images": images,
        "_load_t2i": _make_load_t2i_res(res_h5),
        "_load_full_all": _make_load_full_res(res_h5),
        "pred_action": _loader.load_pred_action(res_h5),
        "gt_action": _loader.load_gt_action(res_h5),
    }

    st.header(f"{res_outcome} · `{res_episode}` · Frame `{res_frame}` · {res_camera} cam")
    if meta.get("instruction"):
        st.caption(f"**Instruction:** {meta['instruction']}")

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔲 Grid Heatmap",
        "🖼 Image Heatmap",
        "📊 Attention Matrix",
        "🤖 Action View",
        "⚖ Compare",
        "📈 Trajectory",
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
            default_checkpoint="",
            checkpoints=_loader.list_checkpoints(ATTN_H5_ROOT),
        )
    with tab5:
        trajectory.render(
            root=_res_root,
            outcome=res_outcome,
            date=res_date,
            episode=res_episode,
            available_frames=res_all_frames,
            cf_slugs=res_cf_slugs,
        )

elif mode == "Online (Inference)":
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

    # ── Save to HDF5 ──────────────────────────────────────────────────────────
    _h5_save_path = os.path.join(
        ATTN_H5_ROOT, "online",
        st.session_state.get("online_episode", "episode"),
        f"{st.session_state.get('online_frame', 0):05d}.h5",
    )
    _col_info, _col_btn = st.columns([6, 2])
    _col_info.caption(f"Save path: `{_h5_save_path}`")
    if _col_btn.button("💾 Save to HDF5", key="online_save_h5"):
        try:
            _save_online_h5(data, _h5_save_path)
            st.success(f"Saved to `{_h5_save_path}`")
        except Exception as _e:
            st.error(f"Save failed: {_e}")

    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔲 Grid Heatmap",
        "🖼 Image Heatmap",
        "📊 Attention Matrix",
        "🤖 Action View",
        "🔀 Counterfactual",
        "🧩 Occlusion Saliency",
        "📐 Language Grounding (CAG)",
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
    with tab5:
        image_saliency.render()
    with tab6:
        cag_view.render()

else:
    # ── Compare (Online) mode ─────────────────────────────────────────────────
    st.header("Compare (Online) — Side-by-Side Checkpoint Comparison")

    _have_a = "cmp_data_a" in st.session_state
    _have_b = "cmp_data_b" in st.session_state

    if not (_have_a and _have_b):
        st.info(
            "Select two checkpoints in the sidebar, choose an episode and frame, "
            "then click **▶ Run Both**."
        )
        if _have_a and not _have_b:
            st.warning("Checkpoint A is ready but B failed or hasn't run yet.")
        elif _have_b and not _have_a:
            st.warning("Checkpoint B is ready but A failed or hasn't run yet.")
        st.stop()

    _cmp_data_a = st.session_state["cmp_data_a"]
    _cmp_data_b = st.session_state["cmp_data_b"]
    _cmp_label_a = st.session_state.get("cmp_label_a", "Checkpoint A")
    _cmp_label_b = st.session_state.get("cmp_label_b", "Checkpoint B")

    # Available layers = union; view handles missing layers gracefully
    def _layers_in(d: dict) -> list[int]:
        return sorted(
            int(k.split("_")[1])
            for k in d.get("prefix", {})
            if k.startswith("layer_")
        )

    _layers_a = _layers_in(_cmp_data_a)
    _layers_b = _layers_in(_cmp_data_b)
    _cmp_all_layers = sorted(set(_layers_a) | set(_layers_b))

    ckpt_compare.render(
        data_a=_cmp_data_a,
        data_b=_cmp_data_b,
        label_a=_cmp_label_a,
        label_b=_cmp_label_b,
        available_layers=_cmp_all_layers,
    )
