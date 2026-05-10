# DeLock — End-to-end Implementation in openpi

Paper: https://suninghuang19.github.io/delock_page/ (`paper.md` in this dir).

DeLock has two **independent** mechanisms that combine to defeat low-data
post-training "lock-in":

1. **Visual-encoder weight-drift regularization** during fine-tuning:
   `L_total = L_BC + λ · ‖θ_v − θ_v_pre‖²`
2. **Contrastive Prompt Guidance (CPG)** at inference. At every flow step `t`:
   `v_CPG = v(o, τ⁻, t) + w · ( v(o, τ⁺, t) − v(o, τ⁻, t) )`
   where `τ⁻` is the *trained* prompt (carries the post-training bias) and
   `τ⁺` is the novel test-time instruction.

The paper's mechanistic analysis (Fig 4a — text→image cross-attention shift)
is exactly the kind of analysis the `viz/` tools in this repo already do for
π₀.₅, so we don't add visualization scaffolding here.

## Where each piece lands in openpi

This is the critical orientation point. openpi has **two parallel model
implementations**:

| stack            | files                                          | what it does                       |
| ---------------- | ---------------------------------------------- | ---------------------------------- |
| **JAX / Flax**   | `src/openpi/models/`, `scripts/train.py`       | training (LoRA wired here)         |
| **PyTorch**      | `src/openpi/models_pytorch/`, `viz_sim/`       | inference / sim / serving / viz    |

LoRA is implemented in `src/openpi/models/gemma.py` via `lora.Einsum`.
The PyTorch `gemma_pytorch.py` does **not** know about `lora_configs` at all
— it only reads `width`, `depth`, `num_heads`, etc. So `paligemma_variant=
"gemma_2b_lora"` on the PyTorch path silently drops LoRA and ends up as
full-precision full-FT.

That means:
- The **vis-encoder regularizer** belongs in the JAX training loop
  (`scripts/train.py:loss_fn`), so it composes cleanly with the existing
  LoRA-trainable filter.
- **CPG** belongs in the PyTorch inference path (`pi0_pytorch.py`), since
  that's what the policy server / sim use.

This is exactly the JAX-train / PyTorch-infer split openpi already uses.

## Implementation plan

### 1. Training: visual-encoder weight-drift regularizer (JAX)
File: `scripts/train.py`.

- Snapshot the *initial* visual-encoder parameter subtree right after
  `init_train_state(...)` returns (use the `nnx.PathRegex` machinery already
  in this file, e.g. `.*PaliGemma_0/img/.*`, to pick out the SigLIP tower).
  Store it `replicated_sharding`d and frozen — it is the `θ_v_pre` reference.
- In `loss_fn`, after computing `chunked_loss`, walk over the same subtree of
  *current* params and add `λ * Σ (p − p_pre)²`. Return both `loss_bc` and
  `loss_reg` for logging.
- Add `vis_reg_lambda: float = 0.0` to `TrainConfig` (default disabled →
  zero behavior change to existing configs).
- Register a new `TrainConfig`: `pi05_droid_delock` cloned from
  `pi05_droid_finetune` but with:
  - `paligemma_variant="gemma_2b_lora"`, `action_expert_variant="gemma_300m_lora"`
    → matches paper's r=16/32 attn+ffn LoRA + `num_kv_heads=1`, `head_dim=256`.
  - `freeze_filter = Pi0Config(...).get_freeze_filter()`
  - `vis_reg_lambda = 1e-4` as a starting value (paper does not specify λ
    numerically — see Open Question O1).
  - `ema_decay = None` (paper disables EMA in Appendix B).

### 2. Inference: Contrastive Prompt Guidance (PyTorch)
File: `src/openpi/models_pytorch/pi0_pytorch.py`.

- New method `sample_actions_cpg(self, device, obs_pos, obs_neg, *, w, num_steps=10)`
  that mirrors `sample_actions` but:
  - Encodes the prefix **twice** (once per prompt — same images, different
    text tokens) → two KV caches `kv_pos`, `kv_neg`.
  - In the denoising loop, runs `denoise_step(...)` twice (one per cache,
    ≈ `2× suffix` cost — prefix is cached, only suffix is re-forwarded each
    step), composes `v_cpg = v_neg + w · (v_pos − v_neg)`, and Eulers
    `x_t += dt · v_cpg`.
  - Asserts the two observations agree on images/state, only `lang_tokens`
    differ — guards against image-preprocessing drift between calls.
  - `w = 1.0` recovers vanilla `v_pos` sampling; `w = 0.0` reverts to the
    trained-prompt-only behavior. Paper uses `w > 1` (extrapolation).

### 3. Policy-server / sim wiring (PyTorch)
- `viz_sim/serve_policy_attn.py`: when the obs payload contains
  `prompt_neg` (str) and `cpg_w` (float), tokenize the negative prompt
  through the same tokenizer used for the positive prompt, build a
  `Observation` clone with the alternate prompt, and route to
  `sample_actions_cpg`. Otherwise use the existing `sample_actions` path.
- `viz_sim/run_pi0_policy_sim.py`: add `--prompt_neg`, `--cpg_w`. Inject
  into the obs dict on every websocket call. Default both to None → no
  behavior change.

### 4. Eval (out of scope here)
- The paper evaluates on **LIBERO** (sim) + **DROID** (real) with paired
  (trained, novel) prompts and 80–100 demos per task. None of that is wired
  in this repo.
- To actually run a DeLock training+eval cycle the user needs:
  1. A demo set in openpi format (≥80 episodes, narrow instruction coverage)
     for one of the existing TrainConfigs (e.g. DROID format works through
     `pi05_droid` data pipeline).
  2. A paired (trained, novel) prompt — e.g. paper's Block-Stacking [C]
     "stack blue block on green block" / "stack green block on blue block".
  3. λ + w sweeps. Paper does not publish numeric λ; w is paper-tunable.
  This PR delivers the *mechanism*; the eval suite is a follow-up.

## Assumptions & open questions

- **A1 (LoRA scope)**: paper's r=16 (Gemma_2b) / r=32 (Gemma_300m) attn+ffn
  LoRA is exactly what `gemma_2b_lora` / `gemma_300m_lora` already provide
  in `src/openpi/models/gemma.py:88-108`. ✅ verified — matches paper.
- **A2 (vis encoder identity)**: "visual encoder θ_v" = the SigLIP vision
  tower under `paligemma.img` (JAX) / `paligemma.vision_tower` (PyTorch).
  This subtree is not LoRA-adapted in either variant — it is full-rank
  trainable, which is what makes the regularizer non-trivial.
- **A3 (CPG runtime cost)**: prefix is KV-cached per prompt, so cost is
  `2× suffix forward` per denoising step + `1× extra prefix forward` once
  per inference. With pi0.5 / 10 denoising steps the overhead is small.
- **A4 (action shape)**: paper uses horizon 10. We keep whatever
  `action_horizon` the chosen openpi config defines; the paper's horizon
  is a training-config knob, not a CPG/regularization knob.
- **A5 (negative prompt at training)**: only used at inference. Training is
  unchanged except for the regularizer.
- **A6 (image preprocessing for CPG)**: identical for both forwards — same
  observation, only the text token sequence differs. We assert this in code.
- **O1 (λ value)**: paper does not publish a numeric λ. Starting at
  `1e-4` (a common L2-on-weights starting point); to be tuned empirically.
- **O2 (negative-prompt selection)**: paper uses *the trained prompt* as
  the negative. For multi-prompt training sets, the natural choice is "the
  trained prompt closest to τ⁺ in lexical / embedding distance", but this
  PR exposes `prompt_neg` as a free-form user-supplied string and leaves
  selection to the caller.

## Status

- [x] §1: training-side regularization + `pi05_droid_delock` TrainConfig (JAX)
  - `TrainConfig.vis_reg_lambda` + `vis_reg_path_regex` in
    `src/openpi/training/config.py`
  - Snapshot of `θ_v_pre` + extra L2 term in `scripts/train.py:loss_fn`
  - `pi05_droid_delock` registered (LoRA r=16/32 + λ=1e-4 + EMA off)
- [x] §2: `sample_actions_cpg` in `src/openpi/models_pytorch/pi0_pytorch.py`
  - Plus `Policy.infer_cpg` in `src/openpi/policies/policy.py` (PyTorch only)
- [x] §3: server + sim wiring
  - `viz_sim/serve_policy_attn.py` routes `obs["prompt_neg"]` + `obs["cpg_w"]`
    to `infer_cpg`; backwards compatible (default = vanilla `infer`).
  - `viz_sim/run_pi0_policy_sim.py` exposes `--prompt_neg` + `--cpg_w` flags;
    they are mutually required (set both or neither).
- [ ] §4: eval — out of scope here

## How to run

**Train (JAX, A100/H100)** — pick the config that matches your benchmark.

Paper-faithful path on **LIBERO** (recommended, runs end-to-end on JAX):
```
uv run scripts/compute_norm_stats.py --config-name pi05_libero_delock
uv run scripts/train.py pi05_libero_delock --exp_name=delock_libero_run0 --overwrite
```

The paper builds 4 LIBERO lock-in probes (Mug-on-Plate [C], Block-Stacking
[C], Open-Microwave [S], Mug-on-Plate [S]) by **subsetting** the
`physical-intelligence/libero` dataset to a single concept/spatial variant
per task with ~100 demos. To replicate, pre-process the LeRobot dataset
into a narrow split before launching training. The current config trains
on the full LIBERO mixture; that's still meaningful for benchmarking but
won't reproduce the paper's lock-in regime exactly.

DROID variant — replace the LeRobot DROID `repo_id` with your demo dataset:
```
uv run scripts/train.py pi05_droid_delock --exp_name=delock_droid_run0
```

**Inference with CPG — JAX (LIBERO eval)**:
```python
from openpi.training import config as _config
from openpi.policies import policy_config

cfg = _config.get_config("pi05_libero_delock")
policy = policy_config.create_trained_policy(cfg, "checkpoints/pi05_libero_delock/.../9999")

# CPG (Algorithm 2 in the paper)
result = policy.infer_cpg_jax(
    obs_pos={**obs, "prompt": "open upper microwave"},   # τ⁺ novel
    obs_neg={**obs, "prompt": "open lower microwave"},   # τ⁻ trained
    cpg_w=1.5,
)
```
This works because both `Pi0` (JAX) and `Policy` now have
`sample_actions_cpg` / `infer_cpg_jax`. The serve_policy.py path can also
be wired to dispatch on `prompt_neg + cpg_w` if you run remote eval.

**Inference with CPG (PyTorch policy server + sim)**:
```
# 1) start server with the DeLock-trained checkpoint
CONFIG=pi05_droid CKPT=checkpoints/.../delock_block_stacking_run0/<step> \
    bash viz_sim/run_pi0_policy_server.sh

# 2) run the sim with both prompts
.venv/bin/python viz_sim/run_pi0_policy_sim.py \
    --task Lift --config pi05_droid \
    --prompt     "stack green block on blue block"   `# τ⁺ (novel)`  \
    --prompt_neg "stack blue block on green block"   `# τ⁻ (trained)` \
    --cpg_w 1.5
```
`--cpg_w 1.0` is identical to vanilla sampling on `--prompt`. Sweep `w` in
`{0.5, 1.0, 1.5, 2.0, 3.0}` to find the steerable range.

## Test suite (37 passing + 1 dummy-model skip)

`baseline/delock/tests/` — runs in ~13s on CPU, no GPU / checkpoint required.

```
JAX_PLATFORMS=cpu .venv/bin/python -m pytest baseline/delock/tests/ -v
```

| File                       | What it covers                                           | Tests |
| -------------------------- | -------------------------------------------------------- | ----- |
| `test_vis_reg.py`          | `_vis_drift_l2` math + path regex against pi0.5 tree     | 5     |
| `test_cpg.py`              | CPG combination math + `sample_actions_cpg` invariants  | 6     |
| `test_train_config.py`     | `pi05_droid_delock` matches paper Appendix B + freeze    | 4     |
| `test_serve_routing.py`    | `AttnCapturingPolicy` routes to `infer_cpg` correctly   | 3     |
| `test_export.py`           | result-export end-to-end (png + webp + json)             | 3     |
| `test_run_cpg_sweep.py`    | sweep runner stacks N infer calls into export schema     | 6     |
| `test_cpg.py` (extended)   | + body-math integration + linearity-in-w + pos/neg symmetry | 9 |
| `test_libero_config.py`    | LIBERO DeLock TrainConfig matches Appendix B + freeze    | 5     |
| `test_jax_cpg.py`          | JAX `sample_actions_cpg` runs on dummy pi05 (CPU, ~2 min) | 2 + 1 skip |

Verified invariants worth highlighting:
- `_vis_drift_l2` returns 0 for identical pytrees, equals `Σ(p−p_pre)²` on
  perturbation, promotes bf16→fp32 internally, handles empty trees.
- Vis-reg regex matches **23 SigLIP leaves** under `PaliGemma/img/` and
  **zero LLM leaves** — won't silently regularize the wrong subtree.
- `sample_actions_cpg` calls `denoise_step` exactly `2 × num_steps` times,
  alternating pos/neg caches; image/state mismatch between observations
  trips the assertion.
- Freeze filter under `pi05_droid_delock` puts LLM/action-expert base in the
  frozen set, leaves LoRA adapters AND vis encoder in the trainable set —
  matches paper §3.2 ("regularize, don't freeze the vis encoder").
- Server skips the attention buffer in CPG mode (otherwise τ⁺ and τ⁻ writes
  would mash); routes to `infer_cpg` only when both `prompt_neg` AND `cpg_w`
  are present (single-flag requests fall back, catching client bugs).
- CPG combination is **linear in w** and obeys the **pos/neg ↔ 1−w symmetry**
  (`v_cpg(w; pos, neg) ≡ v_cpg(1−w; neg, pos)`) — these two properties pin
  the formula tighter than endpoint tests at w=0 and w=1 alone could.

## Training-time loss diagnostics

The training loop now logs three loss components separately to wandb so
`vis_reg_lambda` can be tuned diagnostically:

- `loss_bc` — pure behavioural-cloning loss (`mean(chunked_loss)`).
- `loss_reg_raw` — `‖θ_v − θ_v_pre‖²`, *pre-λ* — directly tracks how far
  the visual encoder has drifted in raw param-space units. Should grow
  monotonically (slowly) over training.
- `loss_reg_scaled` — `λ · ‖θ_v − θ_v_pre‖²`, the actual term added to
  the loss. Compare against `loss_bc` directly:
  - `loss_reg_scaled ≪ loss_bc` → λ too small, regularizer is doing nothing.
  - `loss_reg_scaled ≫ loss_bc` → λ too large, model can't learn the task.
  - In the same OOM as `loss_bc` → λ is in the right ballpark.

This is the diagnostic the paper does not provide — they don't publish a
numeric λ, so this gives a way to decide whether the chosen λ is doing
useful work without running an end-to-end ablation.

The pieces have **not** been run end-to-end (no demos / no real checkpoint
in this PR's scope). Two outstanding pre-flight checks for the user before
the first real training run:

1. **CPG correctness invariant** — `cpg_w=1.0` ⇒ identical to
   `sample_actions(obs_pos)`; `cpg_w=0.0` ⇒ identical to `sample_actions(obs_neg)`
   (modulo numeric noise). The math is `v_neg + w*(v_pos − v_neg)` so this
   holds by construction; worth confirming with a 5-line `torch.allclose`
   on the first checkpoint that loads.
2. **Action-horizon mismatch** — `pi05_droid_delock` uses `action_horizon=10`
   (paper) while `pi05_droid_finetune` uses 16. Anything downstream that
   assumes 16-step chunks (sim `OPEN_LOOP_HORIZON`, render scripts) needs
   to handle 10 instead. The sim defaults to `OPEN_LOOP_HORIZON=8` so this
   is fine for the standard rollout path.

## Result export (img / video)

`baseline/delock/export_cpg_results.py` — pure numpy + matplotlib + PIL,
no openpi imports. Takes a saved CPG run npz (sweep over guidance scales
`w`) and produces:

- `cpg_lines.png` — per-action-dim line plot of the final action chunk,
  one curve per `w`, optional GT overlay.
- `cpg_denoising.webp` — animated 2-row figure: top row is the action
  chunk vs chunk step at each denoising step; bottom row is
  `‖a_w(t) − a_{w=1}(t)‖` over time (contrastive distance from vanilla
  τ⁺ sampling). Useful for "when does each `w` start to diverge?"
- `cpg_summary.json` — per-w mean/std + an `argmax_w_per_dim` field
  showing which guidance setting moves each action dim the most relative
  to vanilla.

Expected npz schema:
```python
{"trajectory": (n_w, n_steps, action_horizon, action_dim),
 "w_values":   (n_w,),
 "prompt_pos": str, "prompt_neg": str,
 "task": str, "ckpt": str,    # optional
 "gt_action": (action_horizon, action_dim),  # optional}
```

CLI:
```
.venv/bin/python baseline/delock/export_cpg_results.py path/to/cpg_run.npz \
    --out-dir baseline/delock/results/run_42 --fps 6
```

### Producing the npz: the CPG sweep runner

`baseline/delock/run_cpg_sweep.py` calls the websocket policy server N times
(once per `w` in the sweep) and stacks the per-denoising-step trajectories
returned in `result["action_trajectory"]` into the schema above. The server
returns `action_trajectory` automatically because `serve_policy_attn.py` now
arms the action-trajectory buffer on every infer call (`(num_steps, H, D)`
float32, small + cheap).

Recipe:
```
# 1) Save one observation from a sim run (any single env step you want
#    to sweep CPG at). Inside run_pi0_policy_sim.py, dump policy_obs:
#       import pickle; pickle.dump(policy_obs, open("obs.pkl", "wb"))
# 2) Boot the policy server:
CONFIG=pi05_droid CKPT=...delock_ckpt... bash viz_sim/run_pi0_policy_server.sh
# 3) Run the sweep:
.venv/bin/python baseline/delock/run_cpg_sweep.py \
    --obs-pickle obs.pkl \
    --prompt-pos "stack green block on blue block" \
    --prompt-neg "stack blue block on green block" \
    --w-values 0.0 0.5 1.0 1.5 2.0 \
    --out baseline/delock/results/run_42/cpg_run.npz
# 4) Render artifacts:
.venv/bin/python baseline/delock/export_cpg_results.py \
    baseline/delock/results/run_42/cpg_run.npz \
    --out-dir baseline/delock/results/run_42
```

A demo using a synthetic trajectory lives in
`baseline/delock/results/demo_synthetic/` (generated by
`tests/test_export.py::test_export_smoke_synthetic`) — you can open the
`.webp` in a browser to see the format without needing a real run.

> Note: in the demo summary, `argmax_w_per_dim` reports `w=0.0` for every
> dim because the synthetic generator places `w=0` farthest from the
> `w=1` reference by construction. Real runs will produce varied per-dim
> argmax values that actually carry information about which dims are
> steerable.

## Known limitations

- **Attention buffer is disabled when CPG is active.** Both the τ⁺ and τ⁻
  forwards write into the same `gemma_pytorch.attn_buffer`, so the captured
  attention would be a mash of the two. To compare τ⁺ vs τ⁻ attention
  (paper Fig 4a), run **two non-CPG rollouts** (one per prompt) and diff
  client side. The CPG response simply omits `text_to_img_attn`.
- **`examples/convert_jax_model_to_pytorch.py` silently drops LoRA adapters.**
  The converter has zero references to `lora` / `adapter`. When you train
  with `paligemma_variant="gemma_2b_lora"` and convert the resulting JAX ckpt
  to PyTorch, the LoRA `_a`/`_b` matrices are dropped from the safetensors
  output. The "PyTorch DeLock ckpt" then collapses to *base + slightly-drifted
  SigLIP* — i.e. effectively `pi05_base` with no LIBERO knowledge — and you
  get **0 % success** at eval and conclude (incorrectly) that DeLock didn't
  learn anything. **Workaround:** point `--policy.dir` at the **JAX
  checkpoint directory** directly (`scripts/serve_policy.py` auto-detects
  format). LoRA-trained DeLock must be served via the JAX path until the
  converter learns to merge `W + (α/r) · B @ A` into the base before saving.

## What we ran on this hardware (LIBERO benchmark, 1× RTX 3090 24GB)

End-to-end run on 2026-05-03 → 2026-05-04. Documents the actual costs/quirks
that came up vs. the paper's Appendix B recipe.

### Training: `pi05_libero_delock`

| | Paper Appendix B | This run | Why |
|---|---|---|---|
| Base | `pi05_base` | `pi05_base` | matches |
| Steps | 10,000 | 10,000 | matches |
| Batch size | **32** | **16** | bs=32 OOMed on 3090 (peak 24.21 GB rematerialized to 23.81 GB; 3090 has 24 GB total) |
| Effective samples | 320 k | 160 k | bs cut → 2× fewer samples seen |
| LR | 5e-5 cosine | 5e-5 cosine | matches (not LR-scaled for the bs reduction) |
| EMA | off | off | matches |
| LoRA r=16/32 attn+ffn | yes | yes | matches |
| λ (vis-reg) | unspecified | 1e-4 | starting point; observable in wandb under `loss_reg_scaled` |
| Wall time | — | **~15.5 h** | bs=16 → 5.5 s/iter on 3090, GPU 100% util |

Cost from a cold start (no cached deps): pi05_base download ~5 min, LIBERO
LeRobot dataset ~30 min (33 GB), `compute_norm_stats.py` ~30 min, training
15.5 h. Final ckpt at `checkpoints/pi05_libero_delock/delock_run0/9999`
(8.9 GB) plus an intermediate at step 5000.

### Inference (LIBERO Docker eval)

Two pieces had to be patched before the LIBERO Docker eval would run:

1. `scripts/docker/serve_policy.Dockerfile` originally bind-mounts `pyproject.toml`
   verbatim into the build, but `pyproject.toml` declares `sam-2 = { path =
   "third_party/sam2" }` as a path dep. The CUDA-runtime image has no nvcc
   and the bind mount is read-only, so sam2's `setup.py` fails twice (once
   to find the dir, once to write `SAM_2.egg-info`). Since `sam-2` is only
   used by `viz/` tooling and **not** by `serve_policy.py`, the Dockerfile
   was patched to copy `pyproject.toml`/`uv.lock` to `/app/`, `sed`-strip
   the two `sam-2` lines, then `uv sync --no-frozen`.
2. The `openpi_server` Docker image's CUDA + cuBLAS combination throws
   `CUBLAS_STATUS_NOT_SUPPORTED` on a bf16 batched matmul on this 3090.
   Workaround: skip the `openpi_server` container, run `scripts/serve_policy.py`
   **natively in the host venv** (where bf16 works fine — we just trained on
   it for 15 h), and only Dockerize the LIBERO `runtime` (the eval client).
   Compose this with `docker compose ... up --no-deps runtime`. `network_mode:
   host` on both services means the runtime container connects to the host
   server at `localhost:8000` transparently.
3. EGL fails inside the runtime container (`Cannot initialize a EGL device
   display`). Set `MUJOCO_GL=osmesa` for the runtime to fall back to pure
   software rendering — slower per frame but reliable on any GPU+driver combo.

End-to-end recipe that worked:

```bash
# 1) Native server (host venv, bf16 fine):
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/serve_policy.py --env LIBERO policy:checkpoint \
    --policy.config pi05_libero_delock \
    --policy.dir /home/edward/projects/openpi_vis/checkpoints/pi05_libero_delock/delock_run0/9999
    # ^ JAX checkpoint dir, NOT the PyTorch conversion (LoRA would be dropped).

# 2) LIBERO Docker eval client only:
CLIENT_ARGS="--args.task-suite-name libero_spatial --args.num-trials-per-task 50" \
MUJOCO_GL=osmesa \
docker compose -f examples/libero/compose.yml up --no-deps runtime --abort-on-container-exit
```

### Results — `libero_spatial` (50 trials × 10 tasks = 500 trials per ckpt)

| Method | Suite total | Per-task (1..10) | Notes |
|---|---|---|---|
| `pi05_libero` (public ckpt, 30k bs=256) | **97.6 %** (488/500) | 100, 100, 100, 98, 90, 98, 100, 94, 98, 98 | matches paper's 98.8 % within noise |
| `pi05_libero_delock` (this run, 10k bs=16, **JAX-direct serve**) | partial | task 1 first 9 trials → 7/9 (77.8 %) | run was stopped early; eps ~31 s like vanilla, model is responding correctly |
| `pi05_libero_delock` via **PyTorch-converted ckpt** | **0 %** (0/175 across 4 tasks before stop) | 0, 0, 0, 0 | LoRA was dropped at conversion — was effectively evaluating `pi05_base`. Don't do this. |

The 0 % run is preserved here as a cautionary tale, not a result. Read the
LoRA-converter limitation above before drawing any conclusion from a low
DeLock number obtained through the PyTorch conversion path.

### Caveats on the apples-to-oranges in this comparison

The 97.6 % vanilla baseline is the **public** `pi05_libero` ckpt — trained
for **30 000 steps at batch size 256** on full LIBERO, ~7.68 M samples. The
DeLock run here was **10 k steps at bs=16**, ~160 k samples — **48× less
effective compute**. So:

- Vanilla > DeLock at this budget is the expected outcome and not a
  refutation of the DeLock mechanism.
- For a fair comparison at this hardware budget, train a *vanilla*
  `pi05_libero` for 10 k steps × bs=16 from `pi05_base` (another ~15.5 h
  on this 3090) and compare against that. Until that's done, the DeLock
  number reads as "how far does 10 k bs=16 + LoRA + vis-reg get on a
  3090" and not "DeLock vs SFT".
- For a paper-faithful **lock-in** comparison, the dataset must also be
  hand-subsetted to a single concept/spatial variant (paper §C.2). The
  current `physical-intelligence/libero` mixture exercises generalization
  across the whole benchmark, not the narrow probe the paper studies.

## Reference

- Paper: `paper.md` (this dir)
- Project page: https://suninghuang19.github.io/delock_page/
