# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo is a fork/extension of the [openpi](https://github.com/Physical-Intelligence/openpi) codebase — open-source Vision-Language-Action (VLA) models from Physical Intelligence — extended with attention visualization tools. The main additions live in `viz/`.

Models supported: **π₀** (flow-based), **π₀-FAST** (autoregressive + FAST tokenizer), **π₀.₅** (upgraded generalization).

## Testing PRs

When helping test a PR, always use **real data** from the PR author's intended data source (HuggingFace, a provided dataset path, the project's download scripts, etc.). **Never propose synthetic/fake data** as a substitute — it only validates that Python runs, not that the actual data pipeline works correctly.

## Package Manager

This project uses **uv** exclusively. Always prefix Python commands with `uv run`.

> **Important**: `uv run python` and `uv pip` both resolve to the conda Python 3.7 environment on this machine, NOT the project's `.venv` (Python 3.11). Use `.venv/bin/python` directly when you need to bypass this. Never run `uv pip install -e .` (dot) as it installs into conda Python 3.7.

```bash
# Install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync # usually only need to run this once

# Run any script
uv run python viz/convert_npy_to_h5.py ...
uv run streamlit run viz/dashboard/app.py
```

## Key Commands

```bash
# Run the attention visualization dashboard
uv run streamlit run viz/dashboard/app.py

# Or use the launch script (sets RESULTS_ROOT per user)
bash viz/start_app.sh

# Run batch attention pipeline (single GPU)
uv run python viz/pipeline.py <DATA_ROOT> <RESULTS_ROOT>

# Run batch attention pipeline (multi-GPU)
uv run python viz/pipeline_mp.py <DATA_ROOT> <RESULTS_ROOT> --gpus 0,1

# Run tests
uv run pytest src/

# Lint / format
uv run ruff check .
uv run ruff format .
```

## Architecture

### Visualization Pipeline

The `viz/` directory is the primary area of active development. It has a three-layer architecture:

**1. Data Layer** — Results HDF5 files
Batch pipeline writes one HDF5 per inference under `RESULTS_ROOT`:
```  # todo ckpt
RESULTS_ROOT/{ckpt}/{left,right}/{success,failure}/{date}/{episode}/{frame:05d}/{frame:05d}.h5
                                                                         {frame:05d}_{cf_key}.h5
```

Each HDF5 file contains:
- `/meta` — prefix_len, seq_len, instruction, token_texts/ids, n_real_tokens
- `/images` — exterior + wrist RGB images (224×224, gzip-4)
- `/prefix/layer_{i}/` — `text_to_img` float32(8, n_text, 512) + `full` float32(8, seq, seq) for all 18 layers
- `/suffix/layer_{i}/` — `action_to_img` float32(n_heads, 8, 512) — action-token → image attention
- `/gt_action` — float32(8, 8): ground-truth [joint_velocity×7, gripper×1] from trajectory.h5, NaN-padded near episode end
- `/pred_action` — float32(N, 8): full predicted action chunk from policy.infer(). N varies by model (pi0.5=15, pi0=10). Dashboard clips to first 8 steps (OPEN_LOOP_HORIZON) for display.

**2. Loader Layer** — `viz/dashboard/loader.py`
Streamlit-cached functions for efficient data access. Key constants:
```python
NUM_IMAGE_TOKENS = 256      # 16×16 patches per camera
TOTAL_IMAGE_TOKENS = 512    # ext (0:256) + wrist (256:512)
TEXT_START_IDX = 768        # where text tokens begin in sequence
NUM_LAYERS = 18
```
Key loaders: `load_meta`, `load_images`, `load_text_to_img`, `load_full_matrix_all_heads`,
`load_action_to_img`, `load_gt_action`, `load_pred_action`.

Results-specific path helpers live in `viz/dashboard/loader_results.py`.

**3. Visualization Layer** — `viz/dashboard/views/`
Modular tabs, each independently customizable:
- `grid_heatmap.py` — text→image attention as 16×16 grids per camera
- `image_heatmap.py` — attention overlaid on RGB images
- `attn_matrix.py` — full sequence-level attention matrix
- `action_view.py` — action→image heatmap grid (8 steps × 2 cameras), temporal coupling, attention source breakdown, pred vs GT benchmark. Requires `data["joint"]` dict; falls back to text-proxy view when absent.
- `trajectory.py` — multi-frame attention grid + action benchmark across episode
- `comparison.py` / `counterfactual.py` — multi-checkpoint or counterfactual prompt comparisons

### Token Layout (Pi0.5 / DROID)

```
[0:256]        [256:512]     [512:768]    [768:N]       [N:N+8]
ext_camera     wrist_camera  zero_padding text_tokens   action_tokens
   256              256          256         ~100            8
```

`TEXT_START_IDX = 768` is the same for both π₀ and π₀.₅. What differs is the **content and length of the text tokens** (N - 768):

#### π₀.₅ text format (state in discrete language tokens)
`PaligemmaTokenizer.tokenize(prompt, state=state_array)` →
```
"Task: {instruction}, State: {s0} {s1} ... {s7};\nAction: "
```
~100 tokens (instruction + 8 discretized joint-state numbers + header)

#### π₀ text format (state is a continuous suffix token, NOT in text)
`PaligemmaTokenizer.tokenize(prompt, state=None)` →
```
"{instruction}\n"
```
~20–50 tokens (instruction only)

**Impact on visualization**: Token labels (`token_texts` in `/meta`) must match the format the model actually used. The current viz code (`inference.py`, `attn_h5_writer.py`) generates labels using the π₀.₅ format even for π₀ checkpoints — this is a known bug. Always check `"pi05" in config_name` before choosing the tokenizer call.

This layout is critical for all slicing/indexing in the visualization code.

### Dashboard Modes

`viz/dashboard/app.py` wires sidebar controls to loader and views.

- **Offline (HDF5)**: Browse pre-computed attention from `attn_h5/` (legacy format). Populates `data["joint"]` from `/suffix/` HDF5 group (action→image only; no action→text or temporal coupling).
- **Results (Benchmark)**: Browse batch pipeline output from `RESULTS_ROOT`. Camera (left/right) is selected in the sidebar — resolves to `RESULTS_ROOT/{camera}/`. Includes Trajectory tab with action benchmark. Also populates `data["joint"]`.
- **Online (Inference)**: Live inference on example episodes from `data/example/`. Episodes are auto-discovered; structure (DROID `recordings/frames/` vs duck `frames/`) is detected automatically. GPU list is detected via `pynvml`; "Auto" selects the GPU with most free memory. Captures both prefix and suffix attention; `gt_action` available for DROID episodes (has trajectory.h5), None for duck format.

`RESULTS_ROOT` is set via environment variable in `viz/start_app.sh` (per-user). Default layout:
```
/path/to/results/cube_gold/
├── right/   ← RESULTS_ROOT/right
└── left/    ← RESULTS_ROOT/left
```

### Attention Capture — RAM Buffer

**Do not write npy files.** All attention capture uses the in-RAM buffer in `src/openpi/models_pytorch/gemma_pytorch.py`:

```python
# Prefix attention (PaliGemma forward, Case 1)
_gpt.enable_attn_buffer()One of the selected episodes has no frames.
    _gpt.clear_attn_buffer()

# Suffix attention (action-token forward, Case 2)
_gpt.enable_suffix_attn_buffer()
try:
    result = policy.infer(example)
    suffix_buf = _gpt.get_suffix_attn_buffer()  # dict[layer_idx, ndarray(1,n_heads,8,k)]
finally:
    _gpt.clear_suffix_attn_buffer()
```

`viz/attn_h5_writer.write_attn_h5_from_buffer(attn_buffer, h5_path, ..., suffix_attn_buffer, gt_action, pred_action)` converts the buffer directly to HDF5.

Both buffers should be enabled together in a single `policy.infer()` call (see `pipeline.py`).
`viz/dashboard/inference.py` enables both and returns `{"prefix": ..., "joint": ..., "pred_action": ..., "gt_action": ...}`.

This same pattern is used in:
- `viz/pipeline.py` / `viz/pipeline_mp.py` — batch offline inference
- `viz/dashboard/inference.py` — online inference (captures both prefix + suffix)
- `viz/dashboard/views/counterfactual.py` — counterfactual prompt inference

### Action Denoising Analysis — `viz/action/`

Standalone analysis scripts for action→token attention across denoising steps:
- `export_denoising_spreadsheet.py` — episode-fair averaging over a dataset; outputs Excel + line/grid plots. Run via `bash viz/export_attn_grid.sh`.
- `analyze_variance_by_outcome.py` — within-episode std of action self-attention, split by success/failure; Mann-Whitney U test + strip/box plots.
- `plot_denoising_attn.py` — single-inference figure: action→image and action→text attention across all NFE steps.
- `denoising_attn_dashboard.py` — interactive Streamlit dashboard for per-step attention.
- `example_suffix_attn.py` — minimal example capturing suffix attention for one frame.

All scripts use `Path(__file__).resolve().parents[2]` to reach the repo root (one extra level vs. `viz/` scripts).

### Batch Pipeline

`viz/pipeline.py` (single-process) and `viz/pipeline_mp.py` (multi-GPU, episode-level parallelism) run offline inference. Both use `load_example` from `pipeline.py` which loads DROID-format episodes (trajectory.h5 + recordings/frames/). Completed episodes are marked with `pi05.md`. Counterfactual prompts are configured via `viz/config/counterfactual.yaml`.

`viz/convert_npy_to_h5.py` exists only for converting legacy npy captures — do not use for new work.

### Example Data

Local example episodes live in `data/example/`:
- `duck/` — uses `frames/{camera}/` directly; no instruction.txt
- `aawr_pineapple/` — DROID format (`recordings/frames/`); no instruction.txt
- `place-pattern/` — DROID format; has `instruction.txt`

The dashboard auto-detects the format by checking for `recordings/frames/`.

### Source Model Code

`src/openpi/` contains the upstream model implementations:
- `models/` — JAX implementations (PaliGemma backbone + action expert)
- `models_pytorch/` — PyTorch equivalents; `gemma_pytorch.py` owns the attention buffer
- `policies/` — Pi0, Pi0-FAST, Pi0.5 policy definitions; `policy.infer()` returns `{"actions": ndarray(N, 8)}` where N is the full chunk size (pi0.5=15, pi0=10). Use `actions[:8]` for the OPEN_LOOP_HORIZON=8 steps that are actually executed.
- `training/` — training configs and data loaders
- `serving/` — policy server for remote inference

## Closed-Loop Sim (`viz_sim/`)

Two parallel sim setups, one per model family. Both render a robosuite Panda
into a single cv2 canvas (sim view + ext/wrist tiles + prompt strip).

**pi0 / pi0.5** (openpi, websocket on port 8000):
- `run_pi0_policy_server.sh` — launches `serve_policy_attn.py` in the openpi `.venv`.
  Defaults to `CONFIG=pi05_droid`; override with `CONFIG=pi05_libero` (or `pi0_droid`,
  `pi0_aloha_towel`, etc.) plus matching `CKPT=` and `PROMPT=`. Download/convert
  helpers: `scripts/get_<config>_torch.sh` (e.g. `get_pi05_libero_torch.sh`).
- `install_pi0_client_in_sim_env.sh` — installs `openpi-client` into `robocasa_sim`
- `run_pi0_policy_sim.py` — supports 3 configs via `--config`. Action spec was
  reverse-engineered from each ckpt's `assets/.../norm_stats.json` (see
  `docs/0428-robocasa-actionspace.md`). Per-ckpt setup:

| `--config` | robot | controller | model dims used | gripper convention |
|---|---|---|---|---|
| `pi05_droid` / `pi0_droid` | Panda (fixed) | `JOINT_VELOCITY` (7) | 8: `[joint_vel(7), gripper(1)]` | DROID `[0,1]` → binarize at 0.5 → `±1` |
| `pi05_libero` | Panda (fixed) | `OSC_POSE` delta | 7: `[Δeef_pos(3), Δeef_rot(3), gripper(1)]` | passthrough (already `±1`) |
| `pi05_robocasa365` | **PandaOmron** (mobile base) | `OSC_POSE` arm + `JOINT_POSITION` torso + `JOINT_VELOCITY` base (HYBRID_MOBILE_BASE composite, 12-D) | 12 (layout B): `[eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]` where `base_motion = [base_x, base_y, base_yaw, torso_z]` | passthrough (already `±1`) |

  **As of 2026-04-29 the robocasa pipeline is wired end-to-end** — there's no
  more "DROID-obs hack". `pi05_robocasa365` is a real `TrainConfig` registered
  in `src/openpi/training/config.py` and uses
  `src/openpi/policies/robocasa_policy.py:RobocasaInputs/RobocasaOutputs`
  (mirroring upstream `robocasa-benchmark/openpi`'s
  `examples/robocasa/main.py`). Server boots with `CONFIG=pi05_robocasa365`
  in `run_pi0_policy_server.sh`.

  - **State (16-D)** sent under `observation/state`:
    `[eef_pos_rel(3), eef_rot_rel(4 quat xyzw), base_pos(3), base_rot(4 quat), gripper_qpos(2)]`
    — eef-first ordering, matching upstream's `np.concatenate` order. Built
    by `make_robocasa_obs` in `viz_sim/run_pi0_policy_sim.py` from
    robosuite's native keys (`robot0_base_to_eef_pos/quat`,
    `robot0_base_pos/quat`, `robot0_gripper_qpos`).
  - **Cameras**: `robot0_agentview_left` (PandaOmron-mounted, the training
    camera) with fallback to plain `agentview` when running native
    robosuite tasks that don't register the robocasa-only camera. Both
    images go through `_resize_with_pad` to 224×224 (letterbox, matches
    upstream `image_tools.resize_with_pad`).
  - **Action (12-D layout B)** is the gym-wrapper layout from
    `robocasa/utils/env_utils.py:convert_action` and
    `robocasa/utils/lerobot_utils.py:ACTION_KEY_ORDERING_HDF5`. The
    `norm_stats.json` shipped with the converted ckpt matches this
    ordering (eef std ≈ 0.32, gripper std ≈ 0.99 Bernoulli, etc.).
    `RobocasaOutputs` returns `actions[:, :12]`.
  - **Slot remap on the sim side is non-trivial.** The model emits layout B,
    but raw robosuite PandaOmron's composite controller flattens action in
    the order it inits part_controllers — and `robot.py:957-958` *appends*
    `right_gripper` AFTER the body_parts dict. So the actual robosuite
    env_action layout is `[arm 6, torso 1, base 3, right_gripper 1, cmode 1]`
    (gripper at index 10, not 6). `run_pi0_policy_sim.py` and
    `viz_sim/eval_perturb.py` apply this remap explicitly:
    ```
    env[0:6]  ← model[0:6]   # arm OSC_POSE (eef pos+rot)
    env[6]    ← model[10]    # torso        ← base_motion[3]
    env[7:10] ← model[7:10]  # base x/y/yaw ← base_motion[0:3]
    env[10]   ← model[6]     # right_gripper ← gripper
    env[11]   ← model[11]    # control_mode
    ```
    Forgetting this swap causes the model's *gripper* signal (∈[-1.8, +2.2]
    bimodal) to drive the *torso* — which is exactly what we observed
    before the fix (torso shooting up/down at every chunk boundary).
  - **`--block_base`** in `run_pi0_policy_sim.py` and `eval_perturb.py`
    zeros torso + base and forces `control_mode = -1` (arm-only). Use it
    to isolate manipulator behavior from base/torso noise during
    debugging.

  **Outstanding (filed at
  [robocasa-benchmark/openpi#3](https://github.com/robocasa-benchmark/openpi/issues/3))**:
  the model is still OOD on plain robosuite tasks because (a) `robot0_agentview_left`
  doesn't register without `gym.make("robocasa/...")`, so we fall back to
  the world-fixed `agentview`; (b) tasks like Lift / PickPlaceSingle / Door
  aren't in the robocasa365 multitask training set; (c) image-rotation
  comment in upstream's `main.py` ("rotate 180") isn't backed by code —
  unclear if it should be applied. The eef block of the model output is
  visibly hallucinated when these conditions aren't met.

  **DROID action convention**: pi0/pi0.5 DROID checkpoints output `joint_velocity`
  (7) for the arm, with `gripper` (1) appended. The runtime sim path uses
  robosuite's `JOINT_VELOCITY` controller — the 7 arm dims are commanded
  joint velocities, not positions/deltas. Earlier versions of this doc
  conflated `joint_position` (a separate norm-stats key in some training
  bundles) with the controller convention; the actual runtime spec is
  velocity. Gripper passes through DROID's `[0, 1]` convention with a 0.5
  binarization to robosuite's `±1`.

**GR00T-N1.7-DROID** (NVIDIA, ZMQ REQ/REP on port 5555):
- `run_gr00t_server.sh` — launches `gr00t/eval/run_gr00t_server.py` in the
  Isaac-GR00T uv venv. Defaults `MODEL=nvidia/GR00T-N1.7-DROID`,
  `TAG=OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`. First-time setup:
  `cd third_party/Isaac-GR00T && uv sync --all-extras`. Pin the GPU with
  `CUDA_VISIBLE_DEVICES=N` so it doesn't fight the openpi server / sim.
- `install_gr00t_client_in_sim_env.sh` — installs `pyzmq`, `msgpack`, `scipy`,
  `Pillow` into `robocasa_sim`
- `gr00t_client.py` — vendored minimal ZMQ/msgpack client (so the sim env
  doesn't need to import `gr00t`). Mirrors `gr00t/policy/server_client.py`.
- `run_policy_sim_gr00t.py` — supports `--robot {Panda, PandaOmron}`. Obs
  is nested:
  `{video.{exterior_image_1_left, wrist_image_left}, state.{eef_9d, gripper_position, joint_position}, language.annotation.language.language_instruction}`,
  images letterboxed to 180×320 with `resize_with_pad`. Video
  `delta_indices` is queried from the server at startup (N1.7-DROID uses
  `[0]`, registry default was `[-15, 0]`); frame buffer is sized
  accordingly. `control_freq=15` to match upstream
  `examples/DROID/main_gr00t.py:DROID_CONTROL_FREQUENCY`. Camera key
  fallback: `robot0_agentview_left → agentview`. Per-step action diagnostic
  saved on exit (npz + png), same plumbing as the pi0.5 path.

  **Action space — the "RELATIVE" gotcha (verified against the model's
  bundled `statistics.json`):** despite the embodiment tag containing
  "RELATIVE_JOINT", `action/joint_position` is in **ABSOLUTE joint angles
  (radians)**. Stats: `min ≈ [-2.78, -1.64, -2.74, -2.95, -2.78, 0.18, -2.90]`,
  `max ≈ [2.74, 1.66, 2.74, -0.19, 2.78, 4.40, 2.90]`,
  `mean ≈ [0.01, 0.28, -0.02, -1.95, -0.03, 2.23, 0.10]` — Panda joint
  limits with the home pose at the mean. The separate
  `relative_action/joint_position` stats key (range ~±0.35 rad) is what
  *real* deltas look like; the runtime tensor is not in that space.
  Upstream `main_gr00t.py` works because the real DROID
  `RobotEnv(action_space="joint_position")` also treats the value as an
  absolute target. Earlier client-side `current_qpos + Δ` anchoring was
  wrong; client now sends `target_qpos = action[:7]` directly.

  **Robosuite `JOINT_POSITION` `input_type` gotcha (the runaway-rotation
  bug):** `JointPositionController` defaults to `input_type="delta"`
  (`third_party/robocasa/robosuite/robosuite/controllers/parts/generic/joint_pos.py:107`).
  In delta mode, `set_goal()` does `self.goal_qpos = current_qpos +
  scale_action(action)` per tick — sending GR00T's absolute ±3-rad
  targets there commands a ~3-rad delta per control cycle, joints
  saturate rate limits and integrate indefinitely (the "position fed in
  as velocity" runaway). `_droid_arm_cfg()` now sets
  `input_type="absolute"` (and explicitly `impedance_mode="fixed"`,
  which absolute mode requires per `joint_pos.py:172`). In absolute mode,
  `set_goal()` does `self.goal_qpos = action` directly
  (`joint_pos.py:227-228`) — no scaling, no accumulation — and the PD
  law `τ = Kp·(action − qpos) + Kd·(−qvel)` servoes to the target.
  Note: this is a robosuite-wide controller flag (delta vs absolute);
  the pi0.5-DROID path keeps the default `delta` mode because pi0.5 *does*
  output deltas — they're using the same controller class with opposite
  `input_type`.

  **PandaOmron mode caveat.** GR00T-N1.7-DROID has no base/torso/cmode
  head, so on `--robot PandaOmron` the script always pins torso=0,
  base=0, control_mode=-1 (effectively forced `--block_base`); it can
  drive arm + gripper only. Native robosuite tasks (Lift,
  PickPlaceSingle, PnPCounterToCab without `gym.make("robocasa/...")`)
  are OOD vs DROID training; expect plausible motion but low task
  success — same OOD caveat as `pi05_robocasa365` on plain robosuite envs.

  **Action chunk:** `joint_position` (T, 7) sent straight through to the
  arm controller; `gripper_position` (T, 1) is absolute in [0, 1],
  binarized at 0.5 → GRIP `±1`. `eef_9d` is consumed only on the state
  side with the DROID rotation correction (`R @ DROID_EEF_ROTATION_CORRECT`,
  top 2 rows = 6D rep; mirrors `compute_eef_9d` in
  `examples/DROID/main_gr00t.py`). World-frame match to DROID is
  approximate. No live attention overlay (GR00T server doesn't expose it).

### Action diagnostic plots

`viz_sim/diagnose_actions.py` is a standalone module that turns a recorded
`(T, D)` model-action stream into a per-dim diagnostic figure: time series
+ histogram per dim, with ±1 clip lines (red) and a training-distribution
mean ± std band (green) overlaid from `norm_stats.json`. Right-side text
boxes annotate `μ`, `σ`, `clip%`, and `[CONST]` when σ is ~0.

`run_pi0_policy_sim.py` records `model_action` every step into a buffer
and saves both `.npz` and `.png` to `--diag_dir` (default
`viz_sim/results/action_diag/`) on any exit path — clean end, `q` keypress,
or SIGINT (signal handler sets a flag that the loop checks; no full-loop
re-indent). Disable with `--no_diag`.

Use this to spot dims that saturate (clip% high), sit constant (`[CONST]`),
or drift far from the training band — the easiest first-pass test of "is
the policy responding at all" vs "stuck in mean-action mode".

Standalone re-plot from a saved `.npz`:
```bash
.venv/bin/python viz_sim/diagnose_actions.py path.npz \
    --config pi05_robocasa365 \
    --norm_stats checkpoints/viz/pi05_robocasa365_pytorch/assets/droid/norm_stats.json
```

### Perturbation eval (`viz_sim/perturb_orch.sh` + `eval_perturb.py`)

5 robosuite tasks × 7 layer-7 KV-perturbation conditions × N episodes.
Conditions are passed as `obs["_perturb"]={mode,camera,layer:7}` and the
server applies the requested scaling at inference time.

```bash
# Full sweep (defaults: 10 eps, seed_base=1000)
bash viz_sim/perturb_orch.sh

# Arm-only debug variant (torso+base zeroed, gripper still active)
BLOCK_BASE=1 EPISODES=3 bash viz_sim/perturb_orch.sh
```

Outputs land in `/mnt/sda/edward/projects/robocasa_365_perturb/{task}/{condition}/ep_NNN.npz`.
The orch starts the `pi05_robocasa365` server on GPU 2, runs every
`(task, condition)` cell sequentially (one server, no restarts), then
calls `render_perturb_video.py` and `build_perturb_report.py` at the end.

Both `eval_perturb.py` and `perturb_orch.sh` were ported on 2026-04-29 to
match the new robocasa schema (was previously running the same
DROID-obs / DROID-config hack as the legacy sim path). Same caveat as
above re: tasks being OOD — baseline success rate is expected to be low
on native robosuite envs until `gym.make("robocasa/...")` is wired in.

### Eval data & comparison animations

Per-episode rollouts written by `viz_sim/eval_runner.py` live at
`/mnt/sda/edward/projects/robocasa_365/{model}/{task}/ep_NNN.npz`. Each npz
contains `success`, `final_reward`, `steps`, `prompt`, subsampled
`attn_stacks (40, 18, 8, 512)`, `ext_frames`, `wrist_frames`.

After each eval finishes `_summary.json`, `eval_runner.py` automatically calls
`render_example_video.render_for_task(args.task)` to (re)build
`results/video/{task}.webp` — an animated WebP with 3 columns (one per
pi0.5 ckpt) × 2 rows (ext / wrist) and attention overlay (layer 7, head-mean).
Missing ckpts render as a blank panel, so the file updates incrementally
across a sweep. Standalone invocation:
`uv run python viz_sim/render_example_video.py [--task Lift]`.

`viz_sim/build_combined_viz.py` aggregates the per-task npz dumps into
`results/combined_summary.json`, per-task `combined_{task}.png`, and
`combined.html`. Run it at the end of orchestration scripts.

## Baselines (`baseline/`)

External-paper reproductions and side investigations that build on the core
openpi stack but live in their own subtree so they can be deleted/rebased
without touching `src/openpi/` or `viz/`.

### `baseline/delock/` — DeLock (LIBERO post-training lock-in)

Paper: https://suninghuang19.github.io/delock_page/

The DeLock mechanism is two pieces, both wired end-to-end here:
1. **Visual-encoder weight-drift L2 regularizer** during JAX training
   (`scripts/train.py:loss_fn`, `TrainConfig.vis_reg_lambda`).
2. **Contrastive Prompt Guidance (CPG)** at inference — JAX
   (`Pi0.sample_actions_cpg`, `Policy.infer_cpg_jax`) and PyTorch
   (`pi0_pytorch.sample_actions_cpg`, `Policy.infer_cpg`) both implemented.

Three TrainConfigs registered in `src/openpi/training/config.py`:
- `pi05_libero_delock`     — LoRA r=16/32 + λ=1e-4, bs=16, 10k steps (paper App. B + 3090 OOM cut)
- `pi05_libero_delock_lambda0` — same as above but vis_reg_lambda=0.0 (ablation)
- `pi05_libero_budget_matched` — full-FT bs=16 × 10k (vanilla at the DeLock budget; not yet trained)

Headline `libero_spatial` results (50 trials × 10 tasks, JAX-direct serve):

| Ckpt | Suite total | Notes |
|---|---|---|
| `pi05_libero` (public, 30k bs=256) | 97.6 % | full-FT, 7.68 M samples |
| `pi05_libero_delock` (10k bs=16) | 66.2 % | LoRA + λ=1e-4, 160 k samples |
| `pi05_libero_delock_lambda0` (10k bs=16) | 62.6 % | LoRA only, 160 k samples |

The 31 pp gap vs vanilla is overwhelmingly the 48× compute-budget gap, not
the DeLock mechanism. vis-reg helps slightly (+3.6 pp) at λ=1e-4. The
budget-matched comparison is the still-missing control.

JAX-side attention capture is in `baseline/delock/attn/`:
- `dump_libero_observations.py` — dump t=0 obs per task (run inside the libero docker)
- `jax_attn_capture.py` — load a JAX ckpt, run prefix forward, write per-layer
  text→image attention HDF5. Implementation note: extracts linen params from
  the nnx-bridged llm via `nnx.split` and applies a fresh
  `gemma.Module(return_attn=True)` — bit-identical to the production forward.
  See `src/openpi/models/gemma.py` `return_attn` flag.
- `render_attn_diff.py`, `render_attn_entropy.py` — static figures
- `dashboard.py` — streamlit dashboard for per-task, per-layer, per-head
  inspection across all captured ckpts. Launch:
  `.venv/bin/python -m streamlit run baseline/delock/attn/dashboard.py --server.port 8503`

Full status, recipe, and known limitations: `baseline/delock/readme.md`.

## GPU Requirements

- Inference only: >8 GB (RTX 4090 sufficient)
- Batch pipeline: ~16 GB per worker; pipeline_mp.py limits to 2 workers/GPU
- LoRA fine-tuning: >22.5 GB
- Full fine-tuning: >70 GB (A100/H100)

## Tests

Tests are co-located with source under `src/openpi/`. The `manual` pytest marker gates tests requiring manual setup/hardware. The test suite auto-falls back to CPU if no GPU is detected.
