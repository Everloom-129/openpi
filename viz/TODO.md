# Attention Masking & Visualization — TODO

## Completed

- [x] **Per-layer attention logit masking** — `_mask_attn_percentile()` in `gemma.py`
  - Mode 1: mask top N% of logits (redistribute attention away from dominant patterns)
  - Mode 2: mask bottom N% of logits
  - Configurable via `Pi0Config(attn_logit_mask_layers={7: 1}, attn_logit_mask_percentile=10.0)`
  - Threaded through `nn.scan` with per-layer mode array (`in_axes=0`)
  - Applied prefix-only in `sample_actions()`

- [x] **Unit tests** — `src/openpi/models/attn_mask_test.py` (10 tests)
  - Mode 0/1/2, batch independence, percentile accuracy, JIT compatibility, config integration

- [x] **Dashboard JAX inference** — `viz/dashboard/inference_jax.py`
  - `run_jax_inference()` produces same dict schema as PyTorch `inference.py`
  - All existing views (grid_heatmap, attn_matrix, image_heatmap, action_view) work unchanged

- [x] **JAX attention buffer API** — `gemma.py`
  - `enable_jax_attn_buffer()`, `get_jax_attn_buffer()`, `clear_jax_attn_buffer()`
  - API stubs in place; not wired to scan output (OOM risk with 18-layer stacking)


## In Progress / Next Steps

### Real-Time Attention Visualization During Robot Inference

- [x] **Synchronous attention server** (`scripts/serve_policy_with_attn.py`)
  - WebSocket policy server (port 8000) + HTTP attention viewer (port 8001)
  - Captures prefix + suffix attention via RAM buffer on every `policy.infer()`
  - Renders matplotlib heatmaps (top-5 text→image + action→image) to PNG
  - Synchronous: viz is rendered before action is returned to the robot client
  - Browser auto-refreshes at ~5Hz via `/status.json` polling
  - Configurable `--viz-layer` (0-17) and `--viz-head` (mean/max/0-7)
  - Launch: `bash scripts/run_attn_server.sh [checkpoint_dir] [device]`

**Architecture** (sync, in-process):
```
Robot Client              GPU Server                         Browser
(main.py)     websocket   (serve_policy_with_attn.py)       (http://host:8001)
  obs  ─────────────────→  policy.infer(obs)
                              │
                              ├─ capture attn buffer (RAM)
                              ├─ render matplotlib PNG
                              ├─ update shared AttnState
                              │
  actions ←─────────────── return actions
                                                             poll /status.json
                                                             reload /attn.png
```

**Follow-up tasks**:
- [ ] **Dashboard "Live" mode** (`app.py`)
  - Add "Live (Server)" mode to the sidebar mode selector
  - Poll the attention server's `/status.json` endpoint at ~2Hz
  - Display using existing views (grid_heatmap, image_heatmap, attn_matrix)
  - Show server timing info (infer_ms, frame age)

- [ ] **Client-side forwarding** (optional, for remote servers)
  - If server and dashboard are on different machines, add a lightweight
    websocket relay that forwards the attention data from server to dashboard host

### JAX Attention Capture (Full)

**Problem**: `nn.scan` + `nn.remat` makes it impractical to return full attention
from all 18 layers in the scan output (OOM: ~600MB for seq_len=1000).

**Potential approaches**:
- [ ] Use `jax.experimental.io_callback` gated by a runtime flag — only transfers
      data when buffer is armed, but needs careful handling with remat
- [ ] Use Flax `self.sow('intermediates', ...)` with `mutable=['intermediates']` and
      scan `variable_axes={"intermediates": 0}` — idiomatic but complex setup
- [ ] Unrolled forward: bypass scan for visualization, manually iterate layers
      with sliced params — most flexible but requires deep Flax linen knowledge

**Current workaround**: Use PyTorch checkpoint + `inference.py` for attention
visualization. JAX model used for masking experiments only.

### Ablation Experiment Pipeline

- [ ] **Batch masking sweep script** — run inference with different masking configs
      (vary layer, mode, percentile) and compare action outputs
- [ ] **Action divergence metric** — quantify how much masking changes predicted
      actions vs. baseline (e.g., L2 distance, cosine similarity)
- [ ] **Integration with robot evaluation** — run masked model on real robot,
      record success rates per masking config
- [ ] **Dashboard comparison view** — side-by-side comparison of baseline vs.
      masked attention patterns and predicted actions
