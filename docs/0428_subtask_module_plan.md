# Subtask Prediction Module — Build Plan

**Date**: 2026-04-28
**Branch**: tony/visualize_attention
**Reference**: https://github.com/LisavilaLee/li/openpi_with_subtask (commit `99c738a`)

`claude --resume 114b62be-d709-4b08-b2d4-6bb99066df35`

---

## Context

Upstream (Physical-Intelligence/openpi) `pi₀.₅` produces actions from a single
flat prompt. The reference fork adds a **subtask prediction head** so that the
VLM first autoregressively generates a low-level *subtask* string from the
high-level prompt, then conditions action flow-matching on the completed
prompt. This is jointly trained with a CE language-modeling loss on the
subtask region of the prefix, alongside the standard flow-matching loss.

Why we want it locally:
- Adds an interpretable language-level intermediate that we can visualize
  alongside attention — natural extension of the `viz/` tools on this branch.
- Lets us probe whether the model's "plan" (decoded subtask) aligns with the
  attention pattern we've been studying (action→image / text→image).
- Enables counterfactuals at the *subtask* level, not just the prompt level.

The reference repo's only commit (`99c738a`) is a complete, working
implementation; this plan ports it to our tree, keeping it side-by-side with
the existing Pi0 / Pi0.5 code rather than replacing it.

---

## Architecture (from reference)

Token sequence used for training:
```
[BOS] "Task: <high>. Subtask: " <low> [EOS] [PAD...]
       └────── prefix (bidir) ──┘└── subtask (causal) ──┘
```
- `token_ar_mask`: `0` over the prefix (bidirectional), `1` over the subtask + EOS (causal).
- `token_loss_mask`: `True` only over the subtask + EOS region — CE loss is masked elsewhere.
- Image tokens stay bidirectional; suffix (action) tokens unchanged.

Training loss: `subtask_ce_loss + flow_matching_loss` (per sample, averaged over horizon for flow).
The CE loss is computed by reading hidden states `prefix_out[:, num_image : num_image + num_text - 1]`,
projecting through `PaliGemma.llm(..., method="decode_to_logits")`, and using
next-token-prediction targets (`tokens[:, 1:]`).

Inference is **two-stage**:
1. `generate_subtask(observation)` — eager Python loop with KV cache, greedy
   argmax, stops on EOS or `max_tokens`. Produces `int32[B, gen_len]`.
2. `build_full_observation(observation, subtask_tokens)` — fills the padded
   region of the original `"Task: X. Subtask: <PAD...>"` prompt with the
   generated tokens; the resulting observation is then passed to the standard
   `sample_actions` path.

Subtask is **cached on the policy object** keyed by raw prompt string, so the
expensive generate step runs once per episode (per unique instruction), not
per timestep.

---

## Critical files to modify (local paths)

| Concern | File | Change |
|---|---|---|
| ModelType enum | `src/openpi/models/model.py` | add `PI05_SUBTASK = "pi05_subtask"` |
| Tokenizer | `src/openpi/models/tokenizer.py` | add `tokenize_high_low_prompt`, `tokenize_high_level_prefix`, `detokenize` to `PaligemmaTokenizer` |
| Transforms | `src/openpi/transforms.py` | add `TokenizeSubtaskTraining`, `TokenizeSubtaskInference` dataclasses |
| Model class | `src/openpi/models/pi0.py` | add `_compute_subtask_ce_loss` to `Pi0`; add `Pi05Subtask(Pi0)` with `compute_loss`, `generate_subtask`, `build_full_observation` |
| Config | `src/openpi/models/pi0_config.py` | add `Pi05SubtaskConfig(Pi0Config)` overriding `model_type`, `create`, `inputs_spec` (so spec includes `token_ar_mask` + `token_loss_mask`) |
| Policy | `src/openpi/policies/policy.py` | add subtask cache (`_cached_subtask_prompt/tokens/text`); detect `Pi05Subtask` and run stage-1 + `build_full_observation` before flow inference; emit `generated_subtask` in outputs |
| Training entries | `src/openpi/training/config.py` | wire `TokenizeSubtaskTraining` into the model-specific transform group; register configs `pi05_subtask_libero` (train) and `pi05_subtask_libero_infer` (uses `subtask_inference=True` → `TokenizeSubtaskInference`); add `debug_pi05_subtask` |
| Test | `scripts/test_subtask_generation.py` (new) | minimal load-and-generate driver, ported from reference `test_subtask_generation.py` |

`token_loss_mask` already exists on `Observation` (used by FAST); no schema
change needed there.

---

## TODOs

### Phase 1 — Port the model code (no training yet)
- [ ] **model.py**: add `PI05_SUBTASK` enum value; verify `Observation.from_dict` already picks up `token_loss_mask` and `token_ar_mask` (it does, lines 128–129 locally).
- [ ] **tokenizer.py**: port `tokenize_high_low_prompt`, `tokenize_high_level_prefix`, `detokenize` from reference `src/openpi/models/tokenizer.py:55–155`. Keep the existing `tokenize` (state-discretized) intact.
- [ ] **transforms.py**: port `TokenizeSubtaskTraining` and `TokenizeSubtaskInference` (reference lines 269–325). Default identity-subtask (`high = low = prompt`) — leave a comment marking this as the swap-in point for real annotations.
- [ ] **pi0.py**: 
  - Add `_compute_subtask_ce_loss` helper on `Pi0` (reference lines 287–317).
  - Add `Pi05Subtask(Pi0)` subclass with overridden `compute_loss` (joint CE + flow), plus `generate_subtask` and `build_full_observation` (reference lines 320–524).
  - Verify `PaliGemma.llm(..., method="decode_to_logits")` and `method="embed"` exist in our local `gemma.py`; if the local fork renamed them, adapt accordingly.
- [ ] **pi0_config.py**: add `Pi05SubtaskConfig(Pi0Config)` with `pi05=True`, `model_type → PI05_SUBTASK`, and an `inputs_spec` that adds `token_ar_mask` (int32) + `token_loss_mask` (bool) to the observation spec (reference lines 117–153).

### Phase 2 — Inference path
- [ ] **policy.py**: in `Policy.__init__`, add `_cached_subtask_prompt/tokens/text = None`. In `infer`, after building the observation but before `sample_actions`:
  - Capture `raw_prompt` from inputs before the tokenize transform consumes it.
  - If model has `generate_subtask` and `observation.token_ar_mask is None`:
    - Cache hit → reuse cached tokens.
    - Cache miss → call `model.generate_subtask`, detokenize for logging, store in cache.
  - Call `model.build_full_observation(observation, subtask_tokens)` and use the returned obs for `sample_actions`.
  - Add `generated_subtask` to the `outputs` dict so callers (and our viz) can read it.
- [ ] **scripts/test_subtask_generation.py**: port the reference test as a minimal smoke test (load checkpoint → tokenize prompt → call `generate_subtask` → print).

### Phase 3 — Training wiring (only if/when we want to actually train)
- [ ] **training/config.py**: 
  - Replace the default tokenize step with `TokenizeSubtaskTraining` for `Pi05SubtaskConfig` model type.
  - Add `subtask_inference: bool = False` to the relevant config dataclass; when set, swap in `TokenizeSubtaskInference` instead.
  - Register `pi05_subtask_libero` (train) and `pi05_subtask_libero_infer` (eval) entries; add `debug_pi05_subtask` with `paligemma_variant="dummy"` for fast CPU tests.

### Phase 4 — Visualization integration (the actual reason we want this)
- [ ] After Phase 2 lands, decide what `viz/dashboard/inference.py` should capture: `generated_subtask` text, the per-token logits/entropy for the subtask region, and (optionally) attention during stage-1 generation.
- [ ] Add a "Subtask" panel to `viz/dashboard/views/` that shows the decoded subtask alongside the standard attention views. Out of scope for this plan — open a follow-up doc.

---

## Verification

End-to-end smoke test (no training required, reuses an existing pi05 checkpoint as a starting point — generation will be garbage until fine-tuned, but the tensor shapes and code paths must work):

```bash
# 1. Library imports cleanly
.venv/bin/python -c "from openpi.models.pi0_config import Pi05SubtaskConfig; \
    from openpi.models.pi0 import Pi05Subtask; \
    from openpi.transforms import TokenizeSubtaskTraining, TokenizeSubtaskInference; \
    print('ok')"

# 2. Tokenizer round-trip
.venv/bin/python -c "from openpi.models.tokenizer import PaligemmaTokenizer; \
    t = PaligemmaTokenizer(max_len=200); \
    toks, mask, ar, loss = t.tokenize_high_low_prompt('pick up the cube', 'grasp the cube'); \
    print(toks.shape, mask.sum(), ar.sum(), loss.sum(), repr(t.detokenize(toks)))"

# 3. End-to-end generate (needs a checkpoint that has the subtask-tuned head;
#    with a vanilla pi05 ckpt the call should still run and return tokens,
#    just nonsense ones).
uv run python scripts/test_subtask_generation.py \
    --checkpoint_dir <path-to-pi05-ckpt> \
    --prompt "pick up the black bowl on the stove and place it on the plate"

# 4. Dummy debug train step (CPU, ~seconds)
uv run python scripts/train.py --config debug_pi05_subtask --max_steps 2
```

Acceptance: (1)–(3) print without exceptions; (4) loss is finite, both
`subtask_loss` and `flow_loss` are non-zero in the first step.

---

## Notes / risks

- `decode_to_logits` and `embed` methods on `PaliGemma.llm` — confirm they exist
  in our local `src/openpi/models/gemma.py`. The reference repo is on the same
  upstream commit family, but we should grep before assuming.
- `generate_subtask` runs in **eager Python** (loop length is dynamic).
  Reference notes ~25–50s per call; the policy-level cache keyed on prompt
  is what makes this acceptable in closed-loop sim.
- Identity subtask (`high = low = prompt`) is a placeholder — to actually train
  meaningful subtask prediction we need real annotations. Keep the swap point
  obvious in `TokenizeSubtaskTraining`.
- Token layout for visualization: when the subtask path is active, the text
  region of the prefix now contains `"Task: ... Subtask: ..."` instead of the
  pi0.5 `"Task: ..., State: ...; Action: "` template. Our `viz/dashboard`
  token-label generation (`inference.py`, `attn_h5_writer.py`) will need a
  branch on `model_type == PI05_SUBTASK`. This is the same shape of bug
  already documented in `CLAUDE.md` for π₀ vs π₀.₅ — follow the same fix
  pattern when we get to Phase 4.
