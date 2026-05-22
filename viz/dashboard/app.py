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
from viz.dashboard.views import action_view, attn_matrix, cag_view, ckpt_compare, comparison, counterfactual, dataset_browser, denoising_view, episode_compare, grid_heatmap, image_heatmap, image_saliency, trajectory
from viz.dashboard.views import gr00t_attn as gr00t_attn_view


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
# Layout: DATA_ATTN_ROOT/{ckpt}/{dataset}/{left,right}/{success,failure}/{date}/{episode}/...
# Default DATA_ATTN_ROOT = grandparent of RESULTS_ROOT (i.e. strip ckpt+dataset).
DATA_ATTN_ROOT = os.environ.get(
    "DATA_ATTN_ROOT",
    os.path.dirname(os.path.dirname(RESULTS_ROOT)),
)
_DEFAULT_CKPT = os.path.basename(os.path.dirname(RESULTS_ROOT))
_DEFAULT_DATASET = os.path.basename(RESULTS_ROOT)


def _list_checkpoints(data_root: str) -> list[str]:
    if not os.path.isdir(data_root):
        return []
    return sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )


def _select_side(side: str, default_ckpt: str, default_dataset: str) -> dict | None:
    """Render a self-contained selector for one side (A or B) of a cross-comparison.

    Returns dict with keys: ckpt, dataset, camera, outcome, date, episode, root.
    Returns None if any level has no data.
    """
    _ckpts = _list_checkpoints(DATA_ATTN_ROOT)
    if not _ckpts:
        st.error(f"No checkpoints in `{DATA_ATTN_ROOT}`.")
        return None
    _idx = _ckpts.index(default_ckpt) if default_ckpt in _ckpts else 0
    ckpt = st.selectbox(f"{side} · Checkpoint", _ckpts, index=_idx, key=f"x_{side}_ckpt")

    _datasets = _list_datasets_for_ckpt(os.path.join(DATA_ATTN_ROOT, ckpt))
    if not _datasets:
        st.warning(f"{side}: no datasets under `{ckpt}`.")
        return None
    _di = _datasets.index(default_dataset) if default_dataset in _datasets else 0
    dataset = st.selectbox(f"{side} · Dataset", _datasets, index=_di, key=f"x_{side}_ds")

    base = os.path.join(DATA_ATTN_ROOT, ckpt, dataset)
    _cams = [c for c in ("right", "left") if os.path.isdir(os.path.join(base, c))]
    if not _cams:
        st.warning(f"{side}: no camera data in `{base}`.")
        return None
    camera = st.radio(f"{side} · Camera", _cams, horizontal=True, key=f"x_{side}_cam")
    root = os.path.join(base, camera)

    _ocs = _rl.list_outcomes(root)
    if not _ocs:
        st.warning(f"{side}: no outcomes in `{root}`.")
        return None
    outcome = st.selectbox(f"{side} · Outcome", _ocs, key=f"x_{side}_oc")

    _dates = _rl.list_dates(root, outcome)
    if not _dates:
        st.warning(f"{side}: no dates.")
        return None
    date = st.selectbox(f"{side} · Date", _dates, key=f"x_{side}_date")

    _eps = _rl.list_episodes(root, outcome, date)
    if not _eps:
        st.warning(f"{side}: no episodes.")
        return None
    episode = st.selectbox(f"{side} · Episode", _eps, key=f"x_{side}_ep")

    return {"ckpt": ckpt, "dataset": dataset, "camera": camera, "outcome": outcome,
            "date": date, "episode": episode, "root": root}


def _list_datasets_for_ckpt(ckpt_root: str) -> list[str]:
    if not os.path.isdir(ckpt_root):
        return []
    out = []
    for d in sorted(os.listdir(ckpt_root)):
        dp = os.path.join(ckpt_root, d)
        if os.path.isdir(dp) and any(
            os.path.isdir(os.path.join(dp, c)) for c in ("left", "right")
        ):
            out.append(d)
    return out

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Pi0.5 Attention")
    st.markdown("---")

    mode = st.radio("Mode", ["Offline (HDF5)", "Results (Benchmark)", "Compare Episodes", "Compare Checkpoints", "Compare (Cross)", "Online (Inference)", "Online (GR00T)", "Online (Upload)", "Online (Dataset)", "Compare (Online)"], index=0)

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
        # Checkpoint selector: scans DATA_ATTN_ROOT for ckpt subdirs.
        _ckpts = _list_checkpoints(DATA_ATTN_ROOT)
        _ckpt_idx = _ckpts.index(_DEFAULT_CKPT) if _DEFAULT_CKPT in _ckpts else 0
        res_ckpt = st.selectbox("Checkpoint", _ckpts or [_DEFAULT_CKPT],
                                index=_ckpt_idx, key="res_ckpt")
        _res_parent = os.path.join(DATA_ATTN_ROOT, res_ckpt)

        _available_ds = _list_datasets_for_ckpt(_res_parent)
        _ds_idx = _available_ds.index(_DEFAULT_DATASET) if _DEFAULT_DATASET in _available_ds else 0
        res_dataset = st.selectbox("Dataset", _available_ds or [_DEFAULT_DATASET],
                                   index=_ds_idx, key="res_dataset")
        _effective_res_root = os.path.join(_res_parent, res_dataset)

        # Camera radio: only show cameras that actually exist for this dataset
        _available_cameras = [c for c in ("right", "left") if os.path.isdir(os.path.join(_effective_res_root, c))]
        if not _available_cameras:
            st.error(f"No camera data found in `{_effective_res_root}`.")
            st.stop()
        res_camera = st.radio("Camera", _available_cameras, horizontal=True, key="res_camera")
        _res_root = os.path.join(_effective_res_root, res_camera)

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

    elif mode == "Compare Episodes":
        # Dataset selector: same logic as Results mode
        _res_parent = os.path.dirname(RESULTS_ROOT)
        _default_ds = os.path.basename(RESULTS_ROOT)
        _available_ds = []
        if os.path.isdir(_res_parent):
            for _d in sorted(os.listdir(_res_parent)):
                _dp = os.path.join(_res_parent, _d)
                if os.path.isdir(_dp) and any(
                    os.path.isdir(os.path.join(_dp, c)) for c in ("left", "right")
                ):
                    _available_ds.append(_d)
        _ds_idx = _available_ds.index(_default_ds) if _default_ds in _available_ds else 0
        cmp_dataset = st.selectbox("Dataset", _available_ds or [_default_ds],
                                   index=_ds_idx, key="cmp_ep_dataset")
        _cmp_effective_root = os.path.join(_res_parent, cmp_dataset)

        _cmp_cameras = [c for c in ("right", "left")
                        if os.path.isdir(os.path.join(_cmp_effective_root, c))]
        if not _cmp_cameras:
            st.error(f"No camera data found in `{_cmp_effective_root}`.")
            st.stop()
        cmp_camera_ep = st.radio("Camera", _cmp_cameras, horizontal=True, key="cmp_ep_camera")
        _cmp_root = os.path.join(_cmp_effective_root, cmp_camera_ep)

        _cmp_outcomes = _rl.list_outcomes(_cmp_root)
        if not _cmp_outcomes:
            st.error(f"No data found in `{_cmp_root}`.")
            st.stop()

        # Date selector spans union of dates across outcomes
        _all_dates: set[str] = set()
        for _oc in _cmp_outcomes:
            _all_dates.update(_rl.list_dates(_cmp_root, _oc))
        _dates_sorted = sorted(_all_dates, reverse=True)
        if not _dates_sorted:
            st.error("No dated runs found.")
            st.stop()
        cmp_ep_date = st.selectbox("Date", _dates_sorted, key="cmp_ep_date")

        # Outcome filter (multi)
        cmp_ep_outcome_sel = st.multiselect(
            "Outcomes", _cmp_outcomes, default=_cmp_outcomes, key="cmp_ep_outcomes",
        )

        st.markdown("---")
        st.caption(f"Camera: **{cmp_camera_ep}** · `{cmp_dataset}` · `{cmp_ep_date}`")

    elif mode == "Compare Checkpoints":
        _ckpts = _list_checkpoints(DATA_ATTN_ROOT)
        if len(_ckpts) < 2:
            st.error(f"Need at least 2 checkpoints in `{DATA_ATTN_ROOT}`; found {len(_ckpts)}.")
            st.stop()
        _idx_a = _ckpts.index(_DEFAULT_CKPT) if _DEFAULT_CKPT in _ckpts else 0
        cc_ckpt_a = st.selectbox("Checkpoint A", _ckpts, index=_idx_a, key="cc_ckpt_a")
        _idx_b = next((i for i, c in enumerate(_ckpts) if c != cc_ckpt_a), 0)
        cc_ckpt_b = st.selectbox("Checkpoint B",
                                 [c for c in _ckpts if c != cc_ckpt_a],
                                 index=0, key="cc_ckpt_b")

        # Dataset must exist under both ckpts
        ds_a = set(_list_datasets_for_ckpt(os.path.join(DATA_ATTN_ROOT, cc_ckpt_a)))
        ds_b = set(_list_datasets_for_ckpt(os.path.join(DATA_ATTN_ROOT, cc_ckpt_b)))
        _shared_ds = sorted(ds_a & ds_b)
        if not _shared_ds:
            st.error(f"No dataset is present under both `{cc_ckpt_a}` and `{cc_ckpt_b}`.")
            st.stop()
        _ds_idx = _shared_ds.index(_DEFAULT_DATASET) if _DEFAULT_DATASET in _shared_ds else 0
        cc_dataset = st.selectbox("Dataset", _shared_ds, index=_ds_idx, key="cc_dataset")

        cc_root_a = os.path.join(DATA_ATTN_ROOT, cc_ckpt_a, cc_dataset)
        cc_root_b = os.path.join(DATA_ATTN_ROOT, cc_ckpt_b, cc_dataset)

        # Camera must exist on both sides
        _cc_cameras = [c for c in ("right", "left")
                       if os.path.isdir(os.path.join(cc_root_a, c))
                       and os.path.isdir(os.path.join(cc_root_b, c))]
        if not _cc_cameras:
            st.error("No shared camera between the two checkpoints for this dataset.")
            st.stop()
        cc_camera = st.radio("Camera", _cc_cameras, horizontal=True, key="cc_camera")
        cc_root_a = os.path.join(cc_root_a, cc_camera)
        cc_root_b = os.path.join(cc_root_b, cc_camera)

        # Shared outcome / date / episode
        _oc_a = set(_rl.list_outcomes(cc_root_a))
        _oc_b = set(_rl.list_outcomes(cc_root_b))
        _shared_oc = sorted(_oc_a & _oc_b)
        if not _shared_oc:
            st.error("No shared outcome between the two checkpoints.")
            st.stop()
        cc_outcome = st.selectbox("Outcome", _shared_oc, key="cc_outcome")

        _d_a = set(_rl.list_dates(cc_root_a, cc_outcome))
        _d_b = set(_rl.list_dates(cc_root_b, cc_outcome))
        _shared_d = sorted(_d_a & _d_b, reverse=True)
        if not _shared_d:
            st.error("No shared date between the two checkpoints.")
            st.stop()
        cc_date = st.selectbox("Date", _shared_d, key="cc_date")

        _e_a = set(_rl.list_episodes(cc_root_a, cc_outcome, cc_date))
        _e_b = set(_rl.list_episodes(cc_root_b, cc_outcome, cc_date))
        _shared_e = sorted(_e_a & _e_b)
        if not _shared_e:
            st.error("No shared episode between the two checkpoints for this date.")
            st.stop()
        cc_episode = st.selectbox("Episode", _shared_e, key="cc_episode")

        st.markdown("---")
        st.caption(f"**A**: `{cc_ckpt_a}` · **B**: `{cc_ckpt_b}` · `{cc_dataset}/{cc_camera}/{cc_outcome}/{cc_date}/{cc_episode}`")

    elif mode == "Compare (Cross)":
        st.markdown("**Side A**")
        x_side_a = _select_side("A", _DEFAULT_CKPT, _DEFAULT_DATASET)
        st.markdown("---")
        st.markdown("**Side B**")
        x_side_b = _select_side("B", _DEFAULT_CKPT, _DEFAULT_DATASET)
        if x_side_a and x_side_b:
            st.markdown("---")
            st.caption(
                f"**A**: `{x_side_a['ckpt']}/{x_side_a['dataset']}/{x_side_a['camera']}` · `{x_side_a['episode']}`  \n"
                f"**B**: `{x_side_b['ckpt']}/{x_side_b['dataset']}/{x_side_b['camera']}` · `{x_side_b['episode']}`"
            )

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
        _has_lerobot = os.path.isfile(os.path.join(_ep_dir, "meta", "info.json"))

        _lerobot_ep_idx = 0  # default; overridden below for LeRobot
        if _has_lerobot:
            # ── LeRobot format (RoboCasa) ─────────────────────────────────
            import sys as _sys
            _sys.path.insert(0, os.path.join(_PROJECT_ROOT, "viz"))
            from robocasa_loader import load_episode_metadata as _load_ep_meta
            _lr_episodes = _load_ep_meta(_ep_dir)
            _lr_ep_labels = [
                f"ep {e['episode_index']}: {e.get('tasks', [''])[0][:60]}"
                for e in _lr_episodes
            ]
            _lr_sel = st.selectbox("LeRobot episode", _lr_ep_labels, key="lr_episode")
            _lerobot_ep_idx = _lr_episodes[_lr_ep_labels.index(_lr_sel)]["episode_index"]
            _lr_ep_info = next(e for e in _lr_episodes if e["episode_index"] == _lerobot_ep_idx)
            _max_frame = max(0, _lr_ep_info["length"] - 1)
            _default_instr = _lr_ep_info.get("tasks", [""])[0]
        else:
            # ── DROID / duck format ───────────────────────────────────────
            _frames_root = os.path.join(_ep_dir, "recordings", "frames") if _has_recordings else os.path.join(_ep_dir, "frames")
            _hand_dir = os.path.join(_frames_root, "hand_camera")
            _n_frames = len([f for f in os.listdir(_hand_dir) if f.endswith(".jpg")]) if os.path.isdir(_hand_dir) else 1
            _max_frame = max(0, _n_frames - 1)
            # Clamp to trajectory.h5 length when present (images may outnumber traj rows)
            _traj_path = os.path.join(_ep_dir, "trajectory.h5")
            if _has_recordings and os.path.exists(_traj_path):
                import h5py as _h5py
                with _h5py.File(_traj_path, "r") as _f:
                    _traj_len = _f["observation/robot_state/joint_positions"].shape[0]
                _max_frame = min(_max_frame, _traj_len - 1)
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
                if _has_lerobot:
                    # LeRobot format (RoboCasa)
                    from robocasa_loader import load_robocasa_example as _load_rc
                    example = _load_rc(
                        lerobot_root=_Path(_ep_dir),
                        episode_index=_lerobot_ep_idx,
                        frame_index=online_frame,
                        ext_camera=online_camera,
                    )
                elif _has_recordings:
                    # DROID format: recordings/frames/{camera}/
                    from pipeline import load_example as _load_example
                    example = _load_example(
                        data_dir=_Path(_ep_dir),
                        index=online_frame,
                        camera=online_camera,
                    )
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

    elif mode == "Online (GR00T)":
        # ── Online (GR00T) sidebar ──────────────────────────────────────────
        # Talks to a GR00T server launched with `ATTN=1` (serve_gr00t_attn.py).
        # We only need a zmq client here; no model load in the dashboard process.
        from viz.dashboard import inference_gr00t as _gi

        st.caption(
            "Server must be running with attention capture: "
            "`CUDA_VISIBLE_DEVICES=0 ATTN=1 bash viz_sim/run_gr00t_server.sh`"
        )
        gr00t_host = st.text_input("Host", value="localhost", key="gr00t_host")
        gr00t_port = st.number_input(
            "Port", min_value=1, max_value=65535, value=5555, step=1, key="gr00t_port",
        )
        gr00t_prompt = st.text_input(
            "Instruction", value="pick up the red cube", key="gr00t_prompt",
        )

        st.markdown("**Images** (ext + wrist) — uploaded images are letterboxed to 180×320.")
        gr00t_ext_file = st.file_uploader(
            "Exterior camera", type=["jpg", "jpeg", "png"], key="gr00t_ext_img"
        )
        gr00t_wrist_file = st.file_uploader(
            "Wrist camera", type=["jpg", "jpeg", "png"], key="gr00t_wrist_img"
        )

        gr00t_run = st.button("▶ Run GR00T Inference", type="primary", key="gr00t_run")

        if gr00t_run:
            from PIL import Image as _PIL
            import io as _io

            if gr00t_ext_file is None or gr00t_wrist_file is None:
                st.error("Upload both ext and wrist images.")
            else:
                with st.spinner("Connecting and running inference…"):
                    try:
                        ext_img = np.asarray(_PIL.open(_io.BytesIO(gr00t_ext_file.read())).convert("RGB"))
                        wrist_img = np.asarray(_PIL.open(_io.BytesIO(gr00t_wrist_file.read())).convert("RGB"))

                        # Letterbox to 180×320 to match the model's training res.
                        # Reuse the helper from run_policy_sim_gr00t.py.
                        import sys as _sys
                        _sys.path.insert(0, os.path.join(_PROJECT_ROOT, "viz_sim"))
                        from run_policy_sim_gr00t import _resize_with_pad as _pad

                        ext_padded = _pad(ext_img, 180, 320)
                        wrist_padded = _pad(wrist_img, 180, 320)

                        # Build the nested obs the GR00T DROID embodiment expects.
                        # State dims set to neutral defaults — the dashboard isn't
                        # tracking a real robot. (eef pose at origin, gripper open,
                        # joint pose at home pose mean from statistics.json.)
                        obs = {
                            "video": {
                                "exterior_image_1_left": ext_padded[None, None, ...],
                                "wrist_image_left":      wrist_padded[None, None, ...],
                            },
                            "state": {
                                "eef_9d": np.array(
                                    [[[0.5, 0.0, 0.3, 1, 0, 0, 0, 1, 0]]],
                                    dtype=np.float32),
                                "gripper_position": np.array([[[0.0]]], dtype=np.float32),
                                "joint_position": np.array(
                                    [[[0.01, 0.28, -0.02, -1.95, -0.03, 2.23, 0.10]]],
                                    dtype=np.float32),
                            },
                            "language": {
                                "annotation.language.language_instruction":
                                    [[gr00t_prompt]],
                            },
                        }

                        client = _gi.get_client(host=gr00t_host, port=int(gr00t_port))
                        slice_dict = _gi.run_inference(client, obs, instruction=gr00t_prompt)
                        st.session_state["gr00t_data"] = slice_dict
                        st.success("Inference complete!")
                    except Exception as e:
                        st.error(f"Inference failed: {e}")
                        st.exception(e)

    elif mode == "Online (Upload)":
        # ── Online (Upload) mode sidebar ──────────────────────────────────────
        from viz.dashboard import inference as _inf

        _upl_checkpoints = _inf.list_online_checkpoints(CHECKPOINT_ROOT)
        if not _upl_checkpoints:
            st.warning(f"No checkpoints found in `{CHECKPOINT_ROOT}`.")

        upl_ckpt = st.selectbox(
            "Checkpoint",
            _upl_checkpoints or ["pi05_droid_pytorch"],
            key="upl_ckpt",
        )
        upl_ckpt_path = os.path.join(CHECKPOINT_ROOT, upl_ckpt)

        st.markdown("**Upload Images**")
        uploaded_exterior = st.file_uploader(
            "Exterior camera image", type=["jpg", "jpeg", "png"], key="upload_ext",
        )
        uploaded_wrist = st.file_uploader(
            "Wrist camera image (optional)", type=["jpg", "jpeg", "png"], key="upload_wrist",
        )

        # Show previews
        if uploaded_exterior:
            st.image(uploaded_exterior, caption="Exterior", width=200)
        if uploaded_wrist:
            st.image(uploaded_wrist, caption="Wrist", width=200)

        upl_instruction = st.text_input("Instruction", key="upl_instruction")

        # Editable robot state (reviewer feedback: zeros may not be ideal)
        with st.expander("Robot state (advanced)", expanded=False):
            st.caption("Joint positions (7-DoF, radians). Defaults to zeros (neutral).")
            _jp_cols = st.columns(7)
            upl_joint_pos = np.array([
                _jp_cols[i].number_input(f"j{i}", value=0.0, format="%.3f", key=f"upl_jp_{i}")
                for i in range(7)
            ], dtype=np.float64)
            upl_gripper = st.number_input(
                "Gripper position (0=open, 1=closed)", value=0.0,
                min_value=0.0, max_value=1.0, step=0.1, key="upl_gripper",
            )

        def _upl_gpu_devices() -> list[str]:
            try:
                import pynvml
                pynvml.nvmlInit()
                n = pynvml.nvmlDeviceGetCount()
                pynvml.nvmlShutdown()
                return ["Auto"] + [f"cuda:{i}" for i in range(n)] + ["cpu"]
            except Exception:
                return ["Auto", "cpu"]

        upl_gpu_sel = st.selectbox("GPU device", _upl_gpu_devices(), key="upl_gpu")

        upl_run_btn = st.button("▶ Run Inference", type="primary", key="upl_run")

        if upl_run_btn:
            if not uploaded_exterior:
                st.error("Please upload at least an exterior camera image.")
            else:
                with st.spinner("Loading model and running inference…"):
                    from PIL import Image as _PILImage

                    ext_img = np.array(_PILImage.open(uploaded_exterior).convert("RGB"))
                    if uploaded_wrist:
                        wrist_img = np.array(_PILImage.open(uploaded_wrist).convert("RGB"))
                    else:
                        wrist_img = np.zeros_like(ext_img)
                    example = {
                        "observation/exterior_image_1_left": ext_img,
                        "observation/wrist_image_left": wrist_img,
                        "observation/joint_position": upl_joint_pos,
                        "observation/gripper_position": np.array([upl_gripper], dtype=np.float64),
                        "prompt": upl_instruction,
                    }
                    try:
                        if upl_gpu_sel == "Auto":
                            from attn_map import select_best_gpu as _sbg
                            _dev_id = _sbg()
                            gpu_device = f"cuda:{_dev_id}" if isinstance(_dev_id, int) else str(_dev_id)
                        else:
                            gpu_device = upl_gpu_sel
                        policy = _inf.load_model(upl_ckpt_path, device=gpu_device)
                        slice_dict = _inf.run_inference(policy, example)
                        st.session_state["upload_data"] = slice_dict
                        st.success("Inference complete!")
                    except Exception as e:
                        st.error(f"Inference failed: {e}")

    elif mode == "Online (Dataset)":
        # ── Online (Dataset) mode sidebar ─────────────────────────────────────
        from viz.dashboard import inference as _inf

        _DEFAULT_DS_ROOT = "/mnt/sda/edward/projects/toy_cube_benchmark/cube_gold"
        ds_data_root = st.text_input(
            "DATA_ROOT",
            value=st.session_state.get("ds_data_root_val", _DEFAULT_DS_ROOT),
            key="ds_data_root_input",
        )
        st.session_state["ds_data_root_val"] = ds_data_root

        _ds_checkpoints = _inf.list_online_checkpoints(CHECKPOINT_ROOT)
        if not _ds_checkpoints:
            st.warning(f"No checkpoints found in `{CHECKPOINT_ROOT}`.")
        ds_ckpt = st.selectbox(
            "Checkpoint",
            _ds_checkpoints or ["pi05_droid_pytorch"],
            key="ds_ckpt",
        )
        ds_ckpt_path = os.path.join(CHECKPOINT_ROOT, ds_ckpt)

        def _ds_gpu_devices() -> list[str]:
            try:
                import pynvml
                pynvml.nvmlInit()
                n = pynvml.nvmlDeviceGetCount()
                pynvml.nvmlShutdown()
                return ["Auto"] + [f"cuda:{i}" for i in range(n)] + ["cpu"]
            except Exception:
                return ["Auto", "cpu"]

        ds_gpu = st.selectbox("GPU device", _ds_gpu_devices(), key="ds_gpu")

        st.markdown("---")
        if st.button("🗑 Clear selection", key="ds_clear"):
            for k in ("ds_selected_episode", "ds_selected_frame", "online_data"):
                st.session_state.pop(k, None)
            st.rerun()

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

    _joint = {}
    for _l in available_layers:
        _a2i = _loader.load_action_to_img(h5_path, _l)
        if _a2i is not None:
            _joint[f"layer_{_l}"] = {"action_to_img": _a2i}

    data = {
        "meta":           meta,
        "images":         images,
        "_load_t2i":      _make_load_t2i(h5_path),
        "_load_full_all": _make_load_full_all(h5_path),
        "joint":          _joint or None,
        "pred_action":    _loader.load_pred_action(h5_path),
        "gt_action":      _loader.load_gt_action(h5_path),
    }

    st.header(f"Checkpoint `{checkpoint_a}` · Episode `{episode}` · Frame `{frame_idx}`")
    if meta.get("instruction"):
        st.caption(f"**Instruction:** {meta['instruction']}")

    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔲 Grid Heatmap",
        "🖼 Image Heatmap",
        "📊 Attention Matrix",
        "🤖 Action View",
        "⚖ Compare",
        "🧠 Denoising",
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

    with tab5:
        denoising_view.render(
            available_layers=available_layers,
            h5_path=h5_path,
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

    _joint_res = {}
    for _l in available_layers:
        _a2i = _loader.load_action_to_img(res_h5, _l)
        if _a2i is not None:
            _joint_res[f"layer_{_l}"] = {"action_to_img": _a2i}

    data = {
        "meta":           meta,
        "images":         images,
        "_load_t2i":      _make_load_t2i_res(res_h5),
        "_load_full_all": _make_load_full_res(res_h5),
        "pred_action":    _loader.load_pred_action(res_h5),
        "gt_action":      _loader.load_gt_action(res_h5),
        "joint":          _joint_res or None,
    }

    st.header(f"{res_outcome} · `{res_episode}` · Frame `{res_frame}` · {res_camera} cam")
    if meta.get("instruction"):
        st.caption(f"**Instruction:** {meta['instruction']}")

    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔲 Grid Heatmap",
        "🖼 Image Heatmap",
        "📊 Attention Matrix",
        "🤖 Action View",
        "⚖ Compare",
        "📈 Trajectory",
        "🧠 Denoising",
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
    with tab6:
        _all_res_h5 = [
            _rl.h5_path_results(_res_root, res_outcome, res_date, res_episode, f)
            for f in res_all_frames
        ]
        denoising_view.render(
            available_layers=available_layers,
            h5_path=res_h5,
            all_frame_paths=_all_res_h5,
            frame_labels=[str(f) for f in res_all_frames],
        )

elif mode == "Compare Episodes":
    st.header(f"🆚 Compare Episodes — `{cmp_dataset}` · {cmp_camera_ep} · {cmp_ep_date}")
    if not cmp_ep_outcome_sel:
        st.info("Select at least one outcome (success / failure) in the sidebar.")
        st.stop()
    try:
        episode_compare.render(
            root=_cmp_root,
            date=cmp_ep_date,
            outcomes=cmp_ep_outcome_sel,
        )
    except Exception as _e:
        import traceback as _tb
        st.error(f"Compare Episodes render failed: **{type(_e).__name__}**: {_e}")
        st.code(_tb.format_exc(), language="text")

elif mode == "Compare Checkpoints":
    st.header(f"🔀 Compare Checkpoints — `{cc_ckpt_a}` ↔ `{cc_ckpt_b}` · `{cc_dataset}` · {cc_camera}")
    episode_compare.render_checkpoint_compare(
        root_a=cc_root_a, root_b=cc_root_b,
        label_a=cc_ckpt_a, label_b=cc_ckpt_b,
        outcome=cc_outcome, date=cc_date, episode=cc_episode,
    )

elif mode == "Compare (Cross)":
    if not (x_side_a and x_side_b):
        st.info("Pick a checkpoint/dataset/episode on both sides in the sidebar.")
        st.stop()
    st.header(f"🔀 Cross Compare — A: `{x_side_a['ckpt']}/{x_side_a['dataset']}` ↔ B: `{x_side_b['ckpt']}/{x_side_b['dataset']}`")
    label_a = f"{x_side_a['ckpt']}/{x_side_a['dataset']}/{x_side_a['episode']}"
    label_b = f"{x_side_b['ckpt']}/{x_side_b['dataset']}/{x_side_b['episode']}"
    episode_compare._render_comparison(
        root_a=x_side_a['root'],
        a=(x_side_a['outcome'], x_side_a['date'], x_side_a['episode']),
        b=(x_side_b['outcome'], x_side_b['date'], x_side_b['episode']),
        root_b=x_side_b['root'],
        label_a=label_a, label_b=label_b,
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

    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔲 Grid Heatmap",
        "🖼 Image Heatmap",
        "📊 Attention Matrix",
        "🤖 Action View",
        "🔀 Counterfactual",
        "🧩 Occlusion Saliency",
        "📐 Language Grounding (CAG)",
        "🧠 Denoising",
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
    with tab7:
        denoising_view.render(
            available_layers=available_layers,
            mem_images=data.get("images"),
            mem_denoising=data.get("suffix_denoising"),
        )

elif mode == "Online (GR00T)":
    st.header("Online GR00T Mode")

    if "gr00t_data" not in st.session_state:
        st.info(
            "1. Launch the server: `CUDA_VISIBLE_DEVICES=0 ATTN=1 bash viz_sim/run_gr00t_server.sh`\n"
            "2. Upload an exterior + wrist image in the sidebar.\n"
            "3. Click **▶ Run GR00T Inference**.\n\n"
            "The view shows Qwen3-VL backbone attention. DiT (action-head) "
            "attention is not captured yet — see task #8."
        )
        st.stop()

    gr00t_attn_view.render(st.session_state["gr00t_data"])

elif mode == "Online (Upload)":
    # ── Online (Upload) mode ──────────────────────────────────────────────────
    st.header("Online (Upload) — Drag & Drop Images")

    if "upload_data" not in st.session_state:
        st.info("Upload images in the sidebar and click **▶ Run Inference** to begin.")
        st.stop()

    data = st.session_state["upload_data"]
    available_layers = sorted(
        int(k.split("_")[1])
        for k in data.get("prefix", {})
        if k.startswith("layer_")
    )

    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔲 Grid Heatmap",
        "🖼 Image Heatmap",
        "📊 Attention Matrix",
        "🤖 Action View",
        "🔀 Counterfactual",
        "🧩 Occlusion Saliency",
        "📐 Language Grounding (CAG)",
        "🧠 Denoising",
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
    with tab7:
        denoising_view.render(
            available_layers=available_layers,
            mem_images=data.get("images"),
            mem_denoising=data.get("suffix_denoising"),
        )

elif mode == "Online (Dataset)":
    # ── Online (Dataset) mode ─────────────────────────────────────────────────
    st.header("Online (Dataset) — Browse & Infer")

    dataset_browser.render(ds_data_root, ds_ckpt_path, ds_gpu)

    # Attention tabs are shown only on the results page
    if st.session_state.get("ds_page") == "results" and "online_data" in st.session_state:
        data = st.session_state["online_data"]
        available_layers = sorted(
            int(k.split("_")[1])
            for k in data.get("prefix", {})
            if k.startswith("layer_")
        )
        tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🔲 Grid Heatmap",
            "🖼 Image Heatmap",
            "📊 Attention Matrix",
            "🤖 Action View",
            "🔀 Counterfactual",
            "🧩 Occlusion Saliency",
            "📐 Language Grounding (CAG)",
            "🧠 Denoising",
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
        with tab7:
            denoising_view.render(
                available_layers=available_layers,
                mem_images=data.get("images"),
                mem_denoising=data.get("suffix_denoising"),
            )

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
