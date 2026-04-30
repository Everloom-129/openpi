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
# Terminal 1 — server (Isaac-GR00T uv venv). Defaults to nvidia/GR00T-N1.7-DROID.
bash viz_sim/run_gr00t_server.sh

# Terminal 2 — sim (robocasa_sim conda)
conda activate robocasa_sim
python viz_sim/run_policy_sim_gr00t.py --task Lift --prompt "pick up the red cube"
```

Notes:

- Robosuite controller is `JOINT_POSITION` with `input_max=output_max=3.14`,
  `input_min=output_min=-3.14` for identity passthrough.
- Images are resized to **180×320** with `resize_with_pad` (vs. 224×224 for
  pi0.5). Obs is nested:
  `{video.{exterior_image_1_left, wrist_image_left}, state.{eef_9d, gripper_position, joint_position}, language.annotation.language.language_instruction}`.
- The script queries `policy.get_modality_config()` at startup and adapts
  `video.delta_indices` (N1.7-DROID actually uses `[0]`, registry default
  was `[-15, 0]`). Frame buffer is sized accordingly.
- Action chunk: `joint_position` (T, 7) is **relative**, applied as
  `target_qpos = current_qpos + Δ` anchored at chunk start. `gripper_position`
  (T, 1) is absolute in [0, 1] and binarized to GRIP `±1`.
- `eef_9d` uses the DROID rotation correction
  (`R @ DROID_EEF_ROTATION_CORRECT`, then top 2 rows = 6D rep). World-frame
  match to DROID is approximate.
- No live attention overlay yet (the GR00T server doesn't expose it); the
  attention plumbing is pi0.5-specific for now.

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

## Eval data layout & post-eval auto-render

Per-episode rollouts live at `/mnt/sda/edward/projects/robocasa_365/{model}/{task}/ep_NNN.npz`.
Each npz contains `success`, `final_reward`, `steps`, `prompt`, plus subsampled
`attn_stacks` (40, 18, 8, 512), `ext_frames`, `wrist_frames`. After every
`eval_runner.py` invocation the `_summary.json` is written and then
`render_example_video.render_for_task(args.task)` rebuilds
`results/video/{task}.webp` (3 ckpt columns × ext/wrist rows). Missing ckpts
render as a blank panel, so the file is useful even mid-sweep.

## TODOs / known gaps

- **robocasa365 OOD on native robosuite tasks.** The training distribution
  is robocasa kitchen scenes + the PandaOmron-mounted `robot0_agentview_left`
  camera, neither of which exists when we call `robosuite.make(env_name=...)`
  directly. We fall back to `agentview` (world-fixed) and accept the OOD —
  baseline success on Lift / PickPlaceSingle / Door / etc. is expected to
  be near zero. Wiring `gym.make("robocasa/...")` is the proper fix
  (tracked at [robocasa-benchmark/openpi#3](https://github.com/robocasa-benchmark/openpi/issues/3)).
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
