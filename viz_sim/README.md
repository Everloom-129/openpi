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

Note: `run_pi0_policy_sim.py` is still wired to the DROID obs/action layout
(JOINT_VELOCITY, 8-dim action, DROID-flat obs dict). Non-DROID configs serve
fine, but matching sim-side adapters are TODO.

### pi0.5-DROID — attention overlay

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
no remap is needed. Obs is the flat DROID dict (`observation/exterior_image_1_left`,
`observation/wrist_image_left`, `observation/joint_position`,
`observation/gripper_position`, `prompt`).

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
- `run_pi0_policy_sim.py` — pi0.5 sim client with live layer/head trackbars.
- `install_pi0_client_in_sim_env.sh` — installs `openpi-client` into `robocasa_sim`.
- `run_gr00t_server.sh` — GR00T server launcher (Isaac-GR00T uv venv).
- `run_policy_sim_gr00t.py` — GR00T sim client.
- `gr00t_client.py` — vendored ZMQ/msgpack `PolicyClient`.
- `install_gr00t_client_in_sim_env.sh` — installs ZMQ/msgpack/scipy/Pillow.
- `setup_robocasa_env.sh` — first-time conda env setup for the sim side.
- `test_viewer.py` — minimal canvas/camera smoke test.

## When you edit anything here

Update this README in the same change. Add/rename the file in the table or
file list, and update the runtime notes if obs/action layout, default
ports, controller settings, or the attention payload shape change.
