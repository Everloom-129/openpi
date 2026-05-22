# viz_sim — closed-loop policy sims with live attention

Two parallel sim setups, one per model family. Both render a robosuite Panda
into a single cv2 canvas (sim view + ext/wrist tiles + prompt strip), and
both run the policy as a separate process so the heavy ML deps stay out of
the sim conda env.

| Model           | Transport             | Server launcher           | Sim client                   |
|-----------------|-----------------------|---------------------------|------------------------------|
| pi0 / pi0.5     | websocket, port 8000  | `run_pi0_policy_server.sh`    | `run_pi0_policy_sim.py`          |
| GR00T-N1.7-DROID| ZMQ REQ/REP, port 5555| `run_gr00t_server.sh`     | `run_policy_sim_gr00t.py`    |

> **Always update this README when you change anything in `viz_sim/`.**

## Environments

- **`robocasa_sim` conda env** — sim side for both models. mujoco 3.3.1, numpy 2.x.
- **openpi `.venv`** — pi0.5 server side. Use `bash viz_sim/run_pi0_policy_server.sh`.
- **Isaac-GR00T uv venv** (Python 3.10, CUDA 12.8) — GR00T server side.
  First-time: `cd third_party/Isaac-GR00T && uv sync --all-extras`.

One-time install of sim-side client deps:

```bash
bash viz_sim/install_pi0_client_in_sim_env.sh        # openpi-client (websocket)
bash viz_sim/install_gr00t_client_in_sim_env.sh  # pyzmq + msgpack + scipy + Pillow
```

## pi0 / pi0.5 (openpi)

Default config is `pi05_droid`. Other openpi checkpoints are served by the
same launcher — pass `CONFIG=` and a matching converted-pytorch directory.
Download/convert helpers under `scripts/`:

| Config         | Download script                          | Default `--prompt`                                |
|----------------|------------------------------------------|---------------------------------------------------|
| `pi05_droid`   | `scripts/get_pi05_droid_torch.sh`        | "pick up the cube"                                |
| `pi05_libero`  | `scripts/get_pi05_libero_torch.sh`       | "pick up the alphabet soup and place it in the basket" |
| `pi0_droid`    | `scripts/get_pi0_droid_torch.sh`         | "pick up the cube"                                |

Example (pi05-libero):

```bash
bash scripts/get_pi05_libero_torch.sh                   # one-time
CONFIG=pi05_libero PROMPT="pick up the alphabet soup" \
    bash viz_sim/run_pi0_policy_server.sh
```

Pass `--config` to `run_pi0_policy_sim.py` so the sim-side obs and action
layout match the served checkpoint. Currently supported:

| `--config` | Robot | Controller | Model dims used | Obs builder | Gripper |
|---|---|---|---|---|---|
| `pi05_droid` / `pi0_droid` | Panda (fixed) | `JOINT_POSITION` delta (kp=50, ±0.3 rad) | 8: `[Δjoint(7), grip(1)]` | `make_droid_obs` (`{exterior_image_1_left, wrist_image_left, joint_position, gripper_position}`) | DROID `[0,1]` → binarize at 0.5 |
| `pi05_libero` | Panda (fixed) | `OSC_POSE` delta | 7: `[Δeef_pos(3), Δeef_rot(3), grip(1)]` | `make_libero_obs` (state = `[eef_pos(3), eef_axisangle(3), gripper_qpos(2)]`) | passthrough (`±1`) |
| `pi05_robocasa365` | **PandaOmron** (mobile base) | `OSC_POSE` arm + `JOINT_POSITION` torso + `JOINT_VELOCITY` base (HYBRID_MOBILE_BASE, 12-D) | 12 (layout B): `[eef_pos(3), eef_rot(3), grip(1), base_motion(4), control_mode(1)]` where `base_motion = [base_x, base_y, base_yaw, torso_z]` | `make_robocasa_obs` (16-D state in **eef→base→gripper** order, ext = `robot0_agentview_left` w/ fallback to `agentview`, `_resize_with_pad` letterbox to 224) | passthrough |

**Important**: prior versions of `pi05_droid` / `pi0_droid` used
`JOINT_VELOCITY` — that was incorrect. DROID training defaults to
`action_dict.joint_position` (see `src/openpi/training/droid_rlds_dataset.py:34`),
so the 7 arm dims are *delta joint positions* in radians, not velocities.

**`pi05_robocasa365` is now wired end-to-end** (as of 2026-04-29). No more
DROID-obs hack:

- Registered as a real `TrainConfig` in `src/openpi/training/config.py`.
- `src/openpi/policies/robocasa_policy.py:RobocasaInputs/RobocasaOutputs`
  mirrors upstream `robocasa-benchmark/openpi`'s schema (single 16-D
  `observation/state` in eef→base→gripper order, `observation/image` +
  `observation/wrist_image`, action `[:, :12]`).
- `run_pi0_policy_server.sh` boots with `CONFIG=pi05_robocasa365`.
- Sim-side `make_robocasa_obs` builds the state from
  `robot0_base_to_eef_pos/quat`, `robot0_base_pos/quat`,
  `robot0_gripper_qpos` (no manual coordinate-frame math — robosuite's
  mobile_robot exposes the `base_to_eef_*` keys directly).

**Slot-remap subtlety on the sim side.** Model emits layout B; raw
robosuite PandaOmron's composite controller is `[arm 6, torso 1, base 3,
right_gripper 1, cmode 1]` because `robosuite/robots/robot.py:957-958`
**appends** `right_gripper` AFTER the body_parts dict has been flattened.
So `env[10]` is the gripper slot, *not* `env[6]`. The sim does:
```
env[0:6]  ← model[0:6]    # arm
env[6]    ← model[10]     # torso        ← base_motion[3]
env[7:10] ← model[7:10]   # base x/y/yaw ← base_motion[0:3]
env[10]   ← model[6]      # right_gripper ← gripper
env[11]   ← model[11]     # control_mode
```
Forgetting this swap drives the *torso* with the model's gripper signal
(σ ≈ 2, bimodal at ±1.8) and produces the "torso shoots up/down" failure
mode we hit before the fix. See `docs/0428-robocasa-actionspace.md` for
the original layout-B reverse-engineering.

**`--block_base` flag** (also wired in `eval_perturb.py`): zero torso +
base, force `control_mode = -1` (arm-only). Useful to isolate manipulator
behavior from base/torso noise.

**Outstanding (filed at
[robocasa-benchmark/openpi#3](https://github.com/robocasa-benchmark/openpi/issues/3))**:
- `robot0_agentview_left` doesn't register without `gym.make("robocasa/...")`
  — falling back to `agentview` (world-fixed) means the camera is OOD vs
  training distribution.
- Native robosuite tasks (Lift / PickPlaceSingle / Door / Stack /
  PickPlaceCan / NutAssemblySquare) aren't in the robocasa365 multitask
  training set — expect low baseline success.
- Upstream comment "rotate 180 degrees to match train preprocessing" in
  `examples/robocasa/main.py` isn't backed by code; rotation may or may
  not be needed.

```bash
# pi05-libero example (server side already running with CONFIG=pi05_libero):
python viz_sim/run_pi0_policy_sim.py --config pi05_libero --prompt "pick up the alphabet soup"
```

### pi0.5-DROID — attention overlay

![pi0.5-DROID attention overlay](pi05_droid_attn_overlay.png)

Single cv2 canvas: sim frontview (left) + `ext`/`wrist` raw tiles and
`ext attn`/`wrist attn` heatmap overlays (right), with the prompt and current
`[layer N/L, head=...]` selection in the bottom strip. The two trackbars below
the strip control layer and head-aggregation mode live.

Server captures **per-layer / per-head text→image attention** in addition to
actions. Reduction (`serve_policy_attn.py`):

- Trim each layer's `(1, H, seq, seq)` to `[TEXT_START_IDX:N_real, :512]`.
- Mean over the *real* text tokens (mask-aware via the policy's input
  transform). Heads are kept separate.
- Stack across all layers → `(L, H, 512)` float32, returned as
  `result["text_to_img_attn"]`. Meta in `text_to_img_meta`.

Sim renders attention live with two cv2 trackbars on the `openpi sim` window:

- **layer** — 0..L-1 (default 7).
- **head: 0=avg 1=min 2=max** — head-aggregation mode.

Selection takes effect on the next rendered frame; no restart needed. The
attention itself only refreshes every `--horizon` (=8) sim steps when a new
action chunk is queried, but you can scrub layers/heads against the most
recent capture in real time. The current selection is shown in the prompt
strip.

```bash
# Terminal 1 — server (openpi venv)
bash viz_sim/run_pi0_policy_server.sh

# Terminal 2 — sim (robocasa_sim conda)
conda activate robocasa_sim
python viz_sim/run_pi0_policy_sim.py --task Lift --prompt "pick up the red cube"
```

Action layout: `[joint_velocity × 7, gripper × 1]` — 8-dim, identical to
the DROID checkpoint output. Robosuite controller is `JOINT_VELOCITY` so
no remap is needed for the qvel dims. Obs is the flat DROID dict
(`observation/exterior_image_1_left`, `observation/wrist_image_left`,
`observation/joint_position`, `observation/gripper_position`, `prompt`).

**Gripper convention.** pi0/pi0.5-DROID outputs `gripper ∈ [0, 1]`
(0=open, 1=close); robosuite's `GRIP` controller expects `[-1, 1]`. The
DROID branch in `run_pi0_policy_sim.py` binarizes at 0.5 and remaps. LIBERO
and robocasa already emit `[-1, 1]` and pass through unchanged.

**Action diagnostic.** Every sim run records `model_action` per step and
saves `(.npz, .png)` to `--diag_dir` (default
`viz_sim/results/action_diag/`) on any exit path (clean end / `q` / SIGINT).
The PNG shows time series + histogram per dim with ±1 clip lines and a
training-distribution mean±σ band overlaid from `norm_stats.json`. Use
`--no_diag` to skip; standalone re-plot via
`.venv/bin/python viz_sim/diagnose_actions.py path.npz --config <name> --norm_stats <path>`.

## GR00T-N1.7-DROID

ZMQ + msgpack to the NVIDIA inference server. We don't install the `gr00t`
package in the sim env; `gr00t_client.py` is a vendored mini-client that
mirrors `gr00t/policy/server_client.py:PolicyClient` (REQ/REP socket,
numpy via `np.save`/`np.load`).

```bash
# Terminal 1 — server (Isaac-GR00T uv venv). Defaults to nvidia/GR00T-N1.7-DROID
# with TAG=OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT.
CUDA_VISIBLE_DEVICES=0 bash viz_sim/run_gr00t_server.sh

# Terminal 2 — sim (robocasa_sim conda)
conda activate robocasa_sim

# Panda + DROID-style task (model's training distribution)
python viz_sim/run_policy_sim_gr00t.py --robot Panda \
    --task Lift --prompt "pick up the red cube"

# PandaOmron + robocasa mobile-manipulator task (base/torso pinned)
python viz_sim/run_policy_sim_gr00t.py --robot PandaOmron \
    --task PnPCounterToCab --prompt "pick up the can and place it in the cabinet"
```

### Action space — the "RELATIVE" gotcha

> **Despite the embodiment tag name `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`,
> `action/joint_position` is in ABSOLUTE joint angles (radians).** Verified
> against the model's bundled `statistics.json`:
> `action/joint_position/mean ≈ Panda home pose`, `min/max ≈ Panda joint
> limits`. The "RELATIVE" prefix only describes the EEF representation. The
> separate `relative_action/joint_position` key in the stats file (range
> ~±0.35 rad) confirms what real deltas look like — the runtime
> `joint_position` output is not in that space.

Earlier versions of this script anchored deltas at chunk start
(`current_qpos + Δ`) on the assumption that the README's "relative joint
positions (7D)" wording matched the runtime tensor. It does not. The
client now sends `target_qpos = action[:7]` directly.

### Robosuite controller — `input_type` matters

The `JointPositionController` defaults to **`input_type="delta"`** (see
`third_party/robocasa/robosuite/.../parts/generic/joint_pos.py:107`),
which interprets the action as a delta added to current qpos every tick.
Sending GR00T's absolute targets in delta mode caused the runaway
"position fed in as velocity" symptom (continuous arm rotation, no
stable pose, joint-rate saturation).

Our `_droid_arm_cfg()` now sets:

```python
{
    "type": "JOINT_POSITION",
    "input_type": "absolute",      # ← key fix
    "impedance_mode": "fixed",     # required by absolute mode
    "input_max": 3.14, "input_min": -3.14,
    "output_max": 3.14, "output_min": -3.14,
    "kp": 50, "damping_ratio": 1,
    "gripper": {"type": "GRIP"},
}
```

In absolute mode `set_goal()` does `self.goal_qpos = action` directly
(`joint_pos.py:227-228`) — no `scale_action`, no current+Δ accumulation.
The PD law `τ = Kp·(action − qpos) + Kd·(−qvel)` then servoes to the
target.

### Sim configurations

| `--robot` | Robot | Controller | Drives | Notes |
|---|---|---|---|---|
| `Panda` | Panda (fixed) | `JOINT_POSITION` absolute (kp=50, ±π rad identity) | 7 arm joints + gripper | DROID training distribution |
| `PandaOmron` | PandaOmron (mobile) | same on arm; default composite for torso/base | 7 arm joints + gripper only | torso/base/cmode pinned to 0/-1 (GR00T has no base head) |

Action chunk:
- `joint_position` (T, 7) — **absolute joint targets in radians**, sent
  directly to the JOINT_POSITION controller.
- `gripper_position` (T, 1) — absolute in [0, 1], 1=closed; binarized at
  0.5 → GRIP `±1`.
- `eef_9d` — ignored at runtime (we drive the arm via joints). State-side
  use applies the DROID rotation correction
  (`R @ DROID_EEF_ROTATION_CORRECT`, top 2 rows = 6D rep).

### Other notes

- **`control_freq=15`** matches `DROID_CONTROL_FREQUENCY` in upstream
  `examples/DROID/main_gr00t.py`. Earlier 20 Hz over-consumed each chunk
  by 33%.
- Images resized to **180×320** with `resize_with_pad` (not 224×224 like
  pi0.5). Obs is nested:
  `{video.{exterior_image_1_left, wrist_image_left}, state.{eef_9d, gripper_position, joint_position}, language.annotation.language.language_instruction}`.
- Camera key falls back: `robot0_agentview_left → agentview` if the task
  doesn't register the robot-mounted camera.
- The script queries `policy.get_modality_config()` at startup and adapts
  `video.delta_indices` (N1.7-DROID uses `[0]`, registry default was
  `[-15, 0]`). Frame buffer is sized accordingly.
- **Action diagnostic** mirrors the pi0.5 path: every run records
  `model_action` per step and saves `.npz + .png` to `--diag_dir` on any
  exit path. Use `--no_diag` to skip.
- **OOD caveat (same as `pi05_robocasa365`)**: GR00T-DROID was trained
  on real DROID Panda data. On native robosuite tasks the cameras and
  scene are out of distribution; on robocasa tasks without
  `gym.make("robocasa/...")` the training-distribution camera
  `robot0_agentview_left` may not register. Expect plausible arm motion,
  low task-success baseline.
- **No live attention overlay yet** — the GR00T server doesn't expose
  attention; the canvas shows a "no attn (gr00t)" placeholder.

## Files

- `serve_policy_attn.py` — pi0.5 websocket server that wraps the policy with
  `AttnCapturingPolicy`. Returns `(L, H, 512)` text→img attention per infer.
- `run_pi0_policy_server.sh` — launcher for the above (openpi `.venv`).
  Defaults to `CONFIG=pi05_robocasa365`; override per checkpoint.
- `run_pi0_policy_sim.py` — pi0.5 sim client with live layer/head trackbars,
  per-config obs builders (`make_droid_obs` / `make_libero_obs` /
  `make_robocasa_obs`), and `--block_base` / `--diag_dir` / `--no_diag` flags.
- `diagnose_actions.py` — standalone module: turns a recorded `(T, D)` action
  array into per-dim time-series + histogram diagnostic plots. Auto-invoked
  by `run_pi0_policy_sim.py` on exit; usable standalone on any saved npz.
- `eval_perturb.py` — headless eval client for the layer-7 KV-perturbation
  experiment. Same `make_robocasa_obs` + 12-D action remap as the live sim;
  supports `--block_base`.
- `perturb_orch.sh` — orchestrator: 5 tasks × 7 conditions × N episodes,
  one server (no restarts), then renders videos + report. `BLOCK_BASE=1`
  env var propagates `--block_base` to every cell.
- `install_pi0_client_in_sim_env.sh` — installs `openpi-client` into `robocasa_sim`.
- `run_gr00t_server.sh` — GR00T server launcher (Isaac-GR00T uv venv).
- `run_policy_sim_gr00t.py` — GR00T sim client.
- `gr00t_client.py` — vendored ZMQ/msgpack `PolicyClient`.
- `install_gr00t_client_in_sim_env.sh` — installs ZMQ/msgpack/scipy/Pillow.
- `setup_robocasa_env.sh` — first-time conda env setup for the sim side.
- `test_viewer.py` — minimal canvas/camera smoke test.
- `eval_runner.py` — headless eval client. Writes `ep_NNN.npz` per episode and
  `_summary.json` per (model, task) under `/mnt/sda/edward/projects/robocasa_365/`.
  At the end of each run it calls `render_example_video.render_for_task(task)` so
  the comparison animation is updated automatically.
- `build_combined_viz.py` — aggregates `ep_NNN.npz` across all (model, task) into
  `results/combined_summary.json`, per-task PNGs (`combined_{task}.png`), and
  `combined.html`. Runs at the end of orchestration scripts.
- `render_example_video.py` — builds 5 animated WebPs, one per task, with 3
  ckpt columns (ext + wrist + attention overlay). Writes `results/video/{task}.webp`.
  Called automatically by `eval_runner.py`; can also be invoked standalone:
  `uv run python viz_sim/render_example_video.py [--task Lift]`.
- `eval_robocasa365_gym.py` — headless eval that builds the env via
  `gym.make("robocasa/<TASK>", split=..., camera_widths=224, camera_heights=224)`
  (real kitchen scenes + `robot0_agentview_left` camera) and sends the
  wrapper's 5-key dict action. Per-model obs/action adapters for
  `pi05_robocasa365` / `pi05_libero` / `pi05_droid`. Output goes to
  `/mnt/sda/edward/projects/robocasa_365_eval_gym/{model}/{task}/ep_NNN.npz`.
- `robocasa365_gym_orch.sh` — orch for the above. 3 ckpts × 5 default
  tasks × `EPISODES` episodes, brings the server up/down per ckpt so each
  config's input transform loads correctly. Env vars: `EPISODES`,
  `SEED_BASE`, `MAX_STEPS`, `SPLIT` (pretrain/target/test), `GPU`, `TASKS`,
  `MODELS`.
- `test_robocasa365_e2e.py` — single-episode smoke test for the
  gym-wrapper path. Writes `attn_grid.png` + `rollout.webp` + `summary.json`
  to `results/test_robocasa365_e2e/{task}/`. Use to verify the plumbing
  before launching a full sweep.
- `render_eval_gym.py` — reads npzs under `robocasa_365_eval_gym/` and
  writes animated GIFs (default; `--format webp` for compact files) in
  two modes: `successes/` (per-episode ext+wrist with attention overlay)
  and `compare/` (3-model side-by-side per task).

## Real-kitchen gym-wrapper eval (`eval_robocasa365_gym.py`)

The original `eval_robocasa365.py` builds the env with raw
`robosuite.make(task, robots="PandaOmron", ...)`, which is fine for native
robosuite tasks (Lift / Stack / etc.) but **never registers the robocasa
kitchen scenes or the `robot0_agentview_left` camera mount**. The
gym-wrapper port closes that gap:

- **Env**: `gym.make("robocasa/<TASK>", split="pretrain", seed=...,
  camera_widths=224, camera_heights=224)` — registers all 396 robocasa
  kitchen envs via `third_party/robocasa/robocasa/wrappers/gym_wrapper.py`
  and exposes the training-distribution `robot0_agentview_left` camera.
- **Obs**: `obs["video.robot0_agentview_left"]` (already RGB-flipped),
  `obs["state.*"]` dict (eef-rel + base + gripper), and
  `obs["annotation.human.task_description"]` (per-episode templated
  language). DROID / LIBERO state vectors that need raw keys
  (`robot0_joint_pos`, `robot0_eef_pos`) come from
  `env.unwrapped._get_observations(force_update=False)`.
- **Action**: send the wrapper's 5-key dict
  (`action.{end_effector_position, end_effector_rotation, gripper_close,
  base_motion, control_mode}`). The wrapper's `step()` rebuilds env_action
  from the dict via `cc.part_controllers` ordering + appended
  `right_gripper`, so the layout-B → env-slot remap from the raw path is
  no longer needed.

End-to-end smoke test (1 episode, attention diagram + animation):

```bash
bash viz_sim/run_pi0_policy_server.sh CONFIG=pi05_robocasa365 GPU=2 &
# wait for "websockets.server:server listening"
/home/edward/miniconda3/envs/robocasa_sim/bin/python \
    viz_sim/test_robocasa365_e2e.py --task PickPlaceCounterToCabinet
# → results/test_robocasa365_e2e/PickPlaceCounterToCabinet/{attn_grid.png,rollout.webp,summary.json}
```

Full sweep (3 ckpts × 5 tasks × N episodes, brings server up/down per
ckpt so each config's input transform loads correctly):

```bash
bash viz_sim/robocasa365_gym_orch.sh                       # defaults: 20 eps, GPU=2, split=pretrain
EPISODES=5 TASKS="CloseDrawer" bash viz_sim/robocasa365_gym_orch.sh
SPLIT=target bash viz_sim/robocasa365_gym_orch.sh          # held-out scenes/objects
```

Output: `/mnt/sda/edward/projects/robocasa_365_eval_gym/{model}/{task}/ep_NNN.npz`
plus `_summary.json`. Each npz: `success`, `final_reward`, `max_reward`,
`steps`, `prompt`, `eef_traj`, `gripper_traj`, `reward_traj`,
`action_taken (T, 12)`, `pred_chunks (n_chunks, 15, 12)`, `attn_steps`,
`attn_stacks (k, 18, 8, 512)`, `frame_steps`, `ext_frames`, `wrist_frames`.

### Sweep results (2026-05-10, split=pretrain, 20 eps/cell)

3 models × 5 tasks × 20 episodes:

| Model | PickPlaceCounterToStove | PickPlaceCounterToSink | OpenDrawer | CloseDrawer | TurnOnSinkFaucet |
|---|---:|---:|---:|---:|---:|
| **pi05_robocasa365** | 3/20 (`r̄_max` 0.15) | 9/20 (0.45) | 1/20 (0.05) | **18/20 (0.90)** | 1/20 (0.05) |
| pi05_libero          | 0/20 (0.00) | 0/20 (0.00) | 0/20 (0.00) | 0/20 (0.00) | 0/20 (0.00) |
| pi05_droid           | 0/20 (0.00) | 0/20 (0.00) | 0/20 (0.00) | 0/20 (0.00) | 0/20 (0.00) |

Notes:
- `pi05_robocasa365` is doing real work — CloseDrawer 90 %, Sink pick-and-place
  45 %. Stove / OpenDrawer / Faucet are still hard (5–16 %).
- `pi05_libero` / `pi05_droid` 0 % is expected: LIBERO's 7-D OSC_POSE+grip
  has no robocasa base/torso/cmode head, and DROID's 7-D joint velocities
  are routed into an OSC_POSE controller slot (incompatible). They're
  recorded as a baseline / for the attention captures, not as a
  realistic comparison.

### Visualization (`render_eval_gym.py`)

Reads the eval npzs and writes animated GIFs (phone-friendly default;
`--format webp` for compact files):

```bash
.venv/bin/python viz_sim/render_eval_gym.py                   # both modes (default)
.venv/bin/python viz_sim/render_eval_gym.py --mode successes  # per-success only
.venv/bin/python viz_sim/render_eval_gym.py --mode compare    # 3-model per-task only
.venv/bin/python viz_sim/render_eval_gym.py --task CloseDrawer
```

Outputs under `results/video_eval_gym/`:
- `successes/{model}__{task}__epNNN.gif` — one row of ext + wrist with
  layer-7 head-mean attention overlay (header: model / task / ep /
  r_max / prompt).
- `compare/{task}.gif` — 3 columns (one per ckpt) × 2 rows (ext / wrist)
  with attention overlay; per-column header shows whether the chosen
  episode succeeded. Picker prefers a successful episode per cell; falls
  back to ep_000 when none exists.

## Eval data layout & post-eval auto-render

Per-episode rollouts live at `/mnt/sda/edward/projects/robocasa_365/{model}/{task}/ep_NNN.npz`.
Each npz contains `success`, `final_reward`, `steps`, `prompt`, plus subsampled
`attn_stacks` (40, 18, 8, 512), `ext_frames`, `wrist_frames`. After every
`eval_runner.py` invocation the `_summary.json` is written and then
`render_example_video.render_for_task(args.task)` rebuilds
`results/video/{task}.webp` (3 ckpt columns × ext/wrist rows). Missing ckpts
render as a blank panel, so the file is useful even mid-sweep.

## TODOs / known gaps

- ~~**robocasa365 OOD on native robosuite tasks.** Wiring
  `gym.make("robocasa/...")` is the proper fix.~~ **Done** (2026-05-10).
  See `eval_robocasa365_gym.py` + `robocasa365_gym_orch.sh` above — real
  kitchen scenes + `robot0_agentview_left` camera now flow through.
  Original `eval_robocasa365.py` (raw `robosuite.make`) is kept for native
  robosuite-task baselines. Upstream issue:
  [robocasa-benchmark/openpi#3](https://github.com/robocasa-benchmark/openpi/issues/3).
- **Image rotation question.** Upstream `examples/robocasa/main.py` has a
  comment `IMPORTANT: rotate 180 degrees to match train preprocessing`
  but the code below it doesn't actually rotate. Unclear if rotation is
  done elsewhere (env wrapper, dataloader) or the comment is stale. Same
  upstream issue tracks this.
- **No live attention overlay for GR00T** — the GR00T server doesn't
  expose attention; pi0.5-side plumbing only.
- **No LIBERO action diagnostic config** in `diagnose_actions.py` —
  labels list exists but training-distribution overlay needs a known
  `norm_stats.json` path.

## When you edit anything here

Update this README in the same change. Add/rename the file in the table or
file list, and update the runtime notes if obs/action layout, default
ports, controller settings, or the attention payload shape change.
