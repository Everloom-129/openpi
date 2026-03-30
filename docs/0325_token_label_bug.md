# Bug: Online inference token labels used raw (un-normalized) state

**Date:** 2025-03-25
**File:** `viz/dashboard/inference.py` → `run_inference()`
**Status:** Fixed

---

## Summary

The token labels shown on the online inference dashboard were wrong for π₀.₅ checkpoints. Every state number in labels like `"State: 128 64 ..."` was incorrect because the code discretized raw proprioception values instead of quantile-normalized values.

---

## Root cause

`run_inference` needed to reconstruct token label strings (displayed in the UI) from the input example *after* `policy.infer()` had already run. The old implementation did this manually:

```python
# OLD CODE (buggy)
joint_pos = example.get("observation/joint_position", np.zeros(7))
gripper_pos = np.atleast_1d(example.get("observation/gripper_position", np.zeros(1)))
state = np.concatenate([np.atleast_1d(joint_pos), gripper_pos])
disc = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
state_str = " ".join(map(str, disc))
full_prompt = f"Task: {instruction}, State: {state_str};\nAction: "
ids = tokenizer._tokenizer.encode(full_prompt, add_bos=True)
```

This skipped the normalization step the model actually uses. The model's `_input_transform` pipeline runs:

```
DroidInputs  →  Normalize(use_quantiles=True)  →  TokenizePrompt(discrete_state_input=True)
```

The `Normalize` step applies **quantile normalization**:

```python
normalized = (raw_state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
```

DROID joint angles are in radians and routinely fall outside `[-1, 1]` (e.g., -2.02, +2.0). The pi05 `q01`/`q99` values map these to the full `[0, 255]` discretization range. Without normalization:
- Values above `+1` all collapse to bin 255
- Values below `-1` all collapse to bin -1 (invalid index)

### Concrete example

| Joint | Raw value | Old bin | q01 / q99 | Normalized | New bin |
|-------|-----------|---------|-----------|------------|---------|
| 5 (elbow) | 2.0 rad | **255** | 1.172 / 3.467 | −0.28 | **~92** |
| 3 (shoulder) | −1.5 rad | **−1** (invalid) | −2.773 / −0.454 | +0.10 | **~140** |

The attention maps were computed correctly (the model ran with the right tokens), but the **labels** on the heatmap x-axis were completely wrong for every state token.

---

## Three bugs in the old code

| # | Bug | Impact |
|---|-----|--------|
| 1 | **No normalization** — raw joint positions discretized directly | State token labels entirely wrong; bins 0 or 255 instead of spread across 0–255 |
| 2 | **No prompt cleaning** — used `instruction` as-is; model uses `.strip().replace("_"," ")` | Token splits differ if prompt contains `_` or `\n` |
| 3 | **No model-family dispatch** — `is_pi05` check was fragile; π₀ uses z-score, π₀.₅ uses quantile | Would produce wrong labels for π₀ checkpoints too |

---

## Fix

Replace the manual reconstruction with a single call to the policy's own transform pipeline:

```python
# NEW CODE (correct)
inputs_copy = {**example}                           # shallow copy: pop("prompt") safe
transformed = policy._input_transform(inputs_copy)  # runs Normalize + TokenizePrompt
token_ids = np.asarray(transformed["tokenized_prompt"])
token_mask = np.asarray(transformed["tokenized_prompt_mask"])
n_real = int(token_mask.sum())
real_ids = token_ids[:n_real].tolist()
tokenizer = PaligemmaTokenizer()
token_texts = [tokenizer._tokenizer.id_to_piece(i) for i in real_ids]
```

The shallow copy (`{**example}`) protects `example["prompt"]` from being consumed by `TokenizePrompt`'s `data.pop("prompt")`. Both model families (π₀ and π₀.₅) work correctly without branching — the transform pipeline already encodes the right normalization and tokenization strategy for the loaded checkpoint.

---

## Tests

`viz/dashboard/test/test_inference_tokens.py` covers:

- **`TestNormalizationMath`** — numerical proof that raw and quantile-normalized discretization differ for typical DROID joint angles
- **`TestRunInferenceTokenLabels`** — unit tests with a fake policy verifying:
  - Labels come from `_input_transform`, not manual reconstruction
  - `example["prompt"]` is not mutated (shallow-copy protection)
  - Padding tokens (mask=False) are excluded from labels
  - `n_real_tokens` matches the mask sum
  - Fallback to generic `tok_i` labels on transform failure
  - `text_to_img` attention slice shape is correct
- **`TestNormalizeTokenizePipeline`** *(manual)* — real transform chain with GCS tokenizer download

```bash
uv run pytest viz/dashboard/test/test_inference_tokens.py -v -m "not manual"
# 13 passed
```
