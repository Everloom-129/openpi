# Attention Pipeline Design

*Date: 2025-03-19*

## Overview

This document describes the refactored attention capture and storage pipeline introduced in `viz/pipeline.py`. The key design goal is to eliminate intermediate NPY files and write compressed HDF5 directly from RAM after each inference call.

**Files involved:**

| File | Role |
|---|---|
| `src/openpi/models_pytorch/gemma_pytorch.py` | Model forward pass; captures attention into RAM buffer |
| `viz/attn_h5_writer.py` | Converts RAM buffer → compressed HDF5 |
| `viz/pipeline.py` | Orchestrates inference, buffer lifecycle, and H5 writes |
| `viz/config/counterfactual.yaml` | Configurable counterfactual prompt variants |

---

## Design Pattern: Global Singleton + Inversion of Control

The model code (`gemma_pytorch.py`) does not decide where to store attention data. The calling pipeline (`pipeline.py`) controls capture behaviour by arming and disarming a module-level buffer before and after each `policy.infer()` call. This is **Inversion of Control**: the model is a passive data producer; the pipeline is the active consumer.

The buffer itself is a **module-level singleton** — a `dict[int, np.ndarray]` that persists for exactly the duration of one inference call.

---

## Data Flow

```
pipeline.py                    gemma_pytorch.py              attn_h5_writer.py
     │                               │                               │
     │  enable_attn_buffer()         │                               │
     │──────────────────────────────►│  _ATTN_BUFFER = {}            │
     │                               │  (armed, ready to receive)    │
     │                               │                               │
     │  policy.infer(example)        │                               │
     │──────────────────────────────►│                               │
     │                               │  forward() running...         │
     │                               │                               │
     │                               │  for i, layer_attn in         │
     │                               │    prefix_output.attentions:  │
     │                               │    _ATTN_BUFFER[i] =          │
     │                               │      tensor.cpu().numpy()     │
     │                               │  (all 18 layers → RAM dict)   │
     │                               │                               │
     │◄──────────────────────────────│  return result                │
     │                               │                               │
     │  buf = get_attn_buffer()      │                               │
     │◄──────────────────────────────│  return _ATTN_BUFFER          │
     │                               │                               │
     │  [finally]                    │                               │
     │  clear_attn_buffer()          │                               │
     │──────────────────────────────►│  _ATTN_BUFFER = None          │
     │                               │  (disarmed, no stale data)    │
     │                               │                               │
     │  write_attn_h5_from_buffer(   │                               │
     │    buf, h5_path, imgs, ...)   │                               │
     │──────────────────────────────────────────────────────────────►│
     │                               │                               │  _write_h5_core():
     │                               │                               │  - detect seq_len
     │                               │                               │  - tokenize instruction
     │                               │                               │  - resize images → 224×224
     │                               │                               │  - for each layer:
     │                               │                               │      write full (8,seq,seq)
     │                               │                               │      write text_to_img slice
     │                               │                               │  → gzip-4 compressed .h5
     │◄──────────────────────────────────────────────────────────────│  return True
```

---

## Buffer Lifecycle (per frame)

```
enable_attn_buffer()   →   policy.infer()   →   get_attn_buffer()
       │                         │                       │
   _ATTN_BUFFER = {}         model fills            snapshot dict
   (armed)                   _ATTN_BUFFER           before clear
                             layer by layer
                                                         │
                                              [finally] clear_attn_buffer()
                                                    _ATTN_BUFFER = None
                                                    (disarmed)
                                                         │
                                              write_attn_h5_from_buffer()
                                                    → .h5 on disk
```

The `finally` guard ensures `clear_attn_buffer()` is called even if `policy.infer()` raises an exception, preventing stale data from leaking into the next frame.

---

## HDF5 Schema

Each `.h5` file stores one inference call (one frame, one prompt):

```
{frame:05d}.h5
├── meta/
│   ├── prefix_len       int32    — TEXT_START_IDX = 768
│   ├── frame_idx        int32
│   ├── seq_len          int32    — full sequence length (image + text + action)
│   ├── n_real_tokens    int32
│   ├── instruction      str
│   ├── token_texts      str[]    — one label per text token
│   └── token_ids        int32[]
├── images/
│   ├── exterior         uint8[224,224,3]   gzip-4
│   └── wrist            uint8[224,224,3]   gzip-4
└── prefix/
    ├── layer_0/
    │   ├── full         float32[8, seq_len, seq_len]   gzip-4
    │   └── text_to_img  float32[8, n_text, 512]        gzip-4
    ├── layer_1/
    │   └── ...
    └── layer_17/
        └── ...
```

**Token layout** (Pi0.5 / DROID):

```
[0:256]        [256:512]     [512:768]    [768:N]        [N:N+8]
ext_camera     wrist_camera  zero_pad     text_tokens    action_tokens
   256              256          256          ~100             8
                                ▲
                          TEXT_START_IDX = 768
```

---

## Output Directory Structure

```
RESULTS_ROOT/
└── {success,failure}/
    └── {date}/
        └── {episode_id}/
            ├── pi05.md              ← completion marker
            ├── 00000/
            │   ├── 00000.h5         ← main inference (original prompt)
            │   ├── 00000_cube.h5    ← counterfactual: object_swap
            │   ├── 00000_pen.h5
            │   └── 00000_empty.h5  ← counterfactual: empty prompt
            ├── 00008/
            │   └── ...
            └── ...
```

Downsampling: inference runs every `OPEN_LOOP_HORIZON = 8` frames (action chunking horizon).

---

## Counterfactual Configuration

Counterfactual prompts are defined in `viz/config/counterfactual.yaml` — no code changes needed to add or remove variants:

```yaml
prompts:
  - key: cube
    prompt: "find the cube and pick it up"
    method: object_swap

  - key: empty
    prompt: ""
    method: empty
```

| Field | Purpose |
|---|---|
| `key` | Filename suffix: `{frame:05d}_{key}.h5` |
| `prompt` | Text sent to the model (empty string = no instruction) |
| `method` | Analysis grouping: `object_swap`, `style`, `negation`, `empty`, `custom` |

---

## Old vs New: Disk I/O Comparison

### Old approach (NPY intermediate files)

```
forward() ──► np.save("attn/0/layer_0.npy")
              np.save("attn/0/layer_1.npy")
              ...  × 18 layers   (disk write × 18)
                        ↓
write_attn_h5(dir) ──► np.load × 18  ──► .h5
                        (disk read × 18)

Total I/O: ~576 MB per inference (seq≈1000, float32, 18 layers)
Leftover files: 18 × .npy in attn/{device_id}/layers_prefix/
```

### New approach (pure RAM)

```
forward() ──► _ATTN_BUFFER[i] = tensor.numpy()  × 18 layers  (RAM only)
                        ↓
write_attn_h5_from_buffer(buf) ──► .h5
                        (one compressed disk write)

Total I/O: only the final gzip-compressed .h5
Leftover files: none
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Module-level global buffer | `forward()` is deep in the call stack; passing a capture target as an argument would require changes across multiple abstraction layers |
| `finally: clear_attn_buffer()` | Prevents stale attention from a failed inference leaking into the next frame's H5 |
| `_ATTN_BUFFER_NOTIFIED` one-time print | Confirms capture is active on first use without flooding logs during batch processing |
| Full matrix saved for every layer | Removes the old `FULL_MATRIX_LAYERS = {1,4,5,7,10}` gate — analysis tooling now has access to all layers |
| YAML-configured counterfactuals | Decouples prompt selection from pipeline logic; researchers can iterate on prompt sets without code changes |
