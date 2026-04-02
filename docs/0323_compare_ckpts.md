# Checkpoint Comparison View — Implementation Notes

`viz/dashboard/views/ckpt_compare.py` + wiring in `app.py` (Compare Online mode)
`viz/dashboard/inference.py` — the inference runner that feeds both models

---

## What this view does

Side-by-side comparison of two checkpoints (typically π₀ vs π₀.₅, or two fine-tune stages)
on the **same input frame**. Both models run forward passes independently; the view shares
controls (token selector, layer, camera, aggregation) across both columns.

Four tabs: Grid Heatmap | Image Heatmap | Attention Matrix | Entropy Curves.

---

## Data flow

```
sidebar: "▶ Run Both" clicked
    → app.py: _load_ep(ep_dir, frame, camera, instruction)
          DROID format → pipeline.load_example()
          duck format  → attn_map.load_duck_example()
    → inference.run_inference(policy_a, example)  → cmp_data_a (dict)
    → inference.run_inference(policy_b, example)  → cmp_data_b (dict)
    → st.session_state["cmp_data_a/b"]
    → ckpt_compare.render(data_a, data_b, ...)
```

---

## The slice dict schema (output of `run_inference`)

```python
{
  "meta": {
      "prefix_len":    768,          # TEXT_START_IDX constant
      "seq_len":       int,          # full sequence length from attention buffer
      "n_real_tokens": int,          # min(n_text, len(token_texts))
      "instruction":   str,
      "token_texts":   list[str],    # truncated to n_real_tokens
  },
  "images": {
      "exterior": np.ndarray (224,224,3) uint8,
      "wrist":    np.ndarray (224,224,3) uint8,
  },
  "prefix": {
      "layer_0": {
          "text_to_img": np.ndarray (8, n_text, 512),  # t2i
          "full":        np.ndarray (8, seq, seq),
      },
      ...  # one entry per captured layer (all 18)
  }
}
```

### t2i tensor explained

`t2i` shape: `(n_heads=8, n_text_actual, 512)`

Sliced from the full attention matrix in `inference.py`:
```python
t2i = attn[:, TEXT_START_IDX : TEXT_START_IDX + n_text_actual, :TOTAL_IMAGE_TOKENS]
#            ↑ queries = text token positions                    ↑ keys = image positions
```

Each element `t2i[head, tok, img_pos]` is the attention weight from text token `tok`
to image patch `img_pos`. Image patches are arranged:
- `[0:256]`   exterior camera (16×16 patches)
- `[256:512]` wrist camera (16×16 patches)

---

## Token layout assumed (from loader.py / CLAUDE.md)

```
[0:256]        [256:512]      [512:768]       [768:768+n_text]   [768+n_text : +8]
ext patches    wrist patches  zero padding    text tokens        action tokens
    256             256            256            ~3–100               8
```

`TEXT_START_IDX = 768` is hard-coded as the same for both π₀ and π₀.₅.

### π₀.₅ text format
```
"Task: {instruction}, State: {s0} ... {s7};\nAction: "
```
~30–100 tokens (instruction + 8 discretized joint-state numbers).
Tokenized via `PaligemmaTokenizer.tokenize(prompt, state=state_array)`.

### π₀ text format (current implementation in `inference.py`)
```python
cleaned = instruction.replace("_", " ").replace("\n", " ")
ids = tokenizer._tokenizer.encode(cleaned, add_bos=True) + tokenizer._tokenizer.encode("\n")
```
Typically 3–20 tokens (instruction only, no state).
Per CLAUDE.md: "state is a continuous suffix token, not in text".

---

## Cross-model token mismatch — the crash & fix

When comparing π₀.₅ (model A, ~30 tokens) vs π₀ (model B, ~3 tokens):

- The token selector is built **from model A's labels only** (`labels_a if labels_a else labels_b`)
- Default selection is `token_labels[min(3, len-1)]` → index 3
- `_build_grid_figure` was indexing `t2i[:, tok_idx, :]` with no bounds check
- π₀'s `t2i.shape[1] == 3` → `IndexError: index 3 is out of bounds for axis 1 with size 3`

**Fix applied**: clamp `tok_idx` to `t2i.shape[1] - 1` inside `_build_grid_figure`
(mirrors the existing clamp in `_build_heatmap_row` and `_entropy_curve`).

---

## Known issues & TODOs

### TODO-1: Verify `is_pi05` detection logic
**File**: `inference.py:97`
```python
is_pi05 = bool(getattr(getattr(policy, "_model", None), "pi05", True))
```
The default fallback is `True` (assumes π₀.₅ if the attribute is missing).
If a π₀ checkpoint doesn't expose `policy._model.pi05`, it will be tokenized
with the π₀.₅ format (including state tokens), producing wrong token labels.
**Verify**: What attribute path does the actual π₀ policy expose? Check against
official `openpi` source for `pi0_droid` policy class.

### TODO-2: Verify pi0 tokenization — BOS + "\n" concatenation
**File**: `inference.py:116`
```python
ids = tokenizer._tokenizer.encode(cleaned, add_bos=True) + tokenizer._tokenizer.encode("\n")
```
Concatenating two separate `encode()` calls may not produce the same tokens as
encoding `cleaned + "\n"` in one call. Tokenizer boundary effects (e.g., a space
before `\n`, subword merges across the join) could shift token identities.
Also: does π₀ include BOS in its prefix? Check training data pipeline in official code.

### TODO-3: Verify TEXT_START_IDX = 768 for pi0
`TEXT_START_IDX = 768` is stated as the same for both models (256 ext + 256 wrist + 256 padding).
**Verify**: Does π₀ use the same 256-token zero-padding block at [512:768]?
Or does the zero-padding differ in length? If the actual text starts at a different
offset, all `t2i` slices for π₀ will be indexing the wrong rows of the attention matrix.

### TODO-4: n_text computed from seq_len may include non-text positions
**File**: `inference.py:94`
```python
n_text = seq_len - TEXT_START_IDX
```
`seq_len` is the full sequence length from the buffer. For π₀, if the continuous
state token is appended into the same forward pass sequence (after the instruction tokens),
`n_text` will be `n_instruction_tokens + n_state_tokens`, not just instruction tokens.
`n_text_actual = min(n_text, len(token_texts))` then trims to `len(token_texts)` (the
encoded instruction length), silently dropping the tail — but the trimming is correct
only if the state tokens always appear *after* the instruction in the sequence.
**Verify**: What is the actual π₀ sequence layout in `gemma_pytorch.py`? How many
positions does the continuous state occupy?

### TODO-5: Token selector is shared — silent clamping hides mismatched viewing
When model A has 30 tokens and model B has 3, selecting e.g. token 15 (valid for A)
will clamp to token 2 for model B. The UI gives no indication that the two columns
are visualizing different (misaligned) tokens. The caption still reads "token 'foo'"
for both, but model B is showing something else.
**Consider**: Show per-model token text labels below each column's figure, or warn
when clamping occurs.

### TODO-6: Attention matrix tab draws wrong boundaries for variable-length text
`_render_attn_matrix` draws vertical/horizontal lines at `[256, 512, 768]`.
The action token boundary at `768 + n_text` is not drawn, so the action region
is not visually separated from text. Also: for π₀ with 3 text tokens, the
action region starts at 771, which looks almost identical to 768 on the plot.

### TODO-7: `available_layers` passed as union, but each model may lack some layers
`app.py:600`: `_cmp_all_layers = sorted(set(_layers_a) | set(_layers_b))`
Layers present in A but absent in B (and vice versa) are included. The grid figure
shows "N/A" for missing layers, which is correct. But the single-layer slider
(used for Heatmap and Matrix tabs) can be set to a layer that only one model has —
the other column will silently show nothing. Consider using the *intersection* for
the single-layer slider, or at least note which model is missing the layer.

---

## Running the tests

```bash
PYTHONPATH=. uv run pytest viz/dashboard/views/test_ckpt_compare.py -v
```

Tests cover all pure helpers in `ckpt_compare.py` and reproduce the exact crash
scenario (TODO-1 in the cross-model section above) to confirm the fix holds.
