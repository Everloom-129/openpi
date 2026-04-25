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
```
RESULTS_ROOT/{left,right}/{success,failure}/{date}/{episode}/{frame:05d}/{frame:05d}.h5
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
_gpt.enable_attn_buffer()
try:
    result = policy.infer(example)
    buf = _gpt.get_attn_buffer()          # dict[layer_idx, ndarray(1,8,seq,seq)]
finally:
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

## GPU Requirements

- Inference only: >8 GB (RTX 4090 sufficient)
- Batch pipeline: ~16 GB per worker; pipeline_mp.py limits to 2 workers/GPU
- LoRA fine-tuning: >22.5 GB
- Full fine-tuning: >70 GB (A100/H100)

## Tests

Tests are co-located with source under `src/openpi/`. The `manual` pytest marker gates tests requiring manual setup/hardware. The test suite auto-falls back to CPU if no GPU is detected.
