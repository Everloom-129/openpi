# Pi0.5 VQA Implementation Plan

## 1. Goal
Enable the Pi0.5 model (specifically the PaliGemma VLM part) to generate autoregressive text responses (VQA) given an image and a text prompt.

## 2. Current Status & Issues
We have created a script `viz/h2_2_vqa.py` and added `generate` methods to the model classes.
However, we are encountering two main issues:
1.  **Dtype Mismatch**: `RuntimeError: invalid dtype for bias - should match query's dtype`. This occurs in the attention mechanism when passing `attention_mask` alongside `inputs_embeds`.

## 3. Analysis of GitHub Issues (References #679, #701)
*Note: Inferred from context and common HF/PaliGemma behaviors.*

The core problem lies in how Hugging Face's `generate()` handles `inputs_embeds` and `attention_mask` when using `bfloat16` models.
- When `inputs_embeds` are passed, HF might not automatically cast the `attention_mask` to the correct dtype (bfloat16) for the underlying `scaled_dot_product_attention` if the mask is provided as a boolean/integer padding mask.
- Alternatively, if we pass a pre-computed float mask (additive bias), it MUST be `bfloat16`.

## 4. Implementation Plan

### Step 1: Fix Dtype Mismatch in `generate`
We need to ensure the `attention_mask` passed to `paligemma.generate` is in the format and dtype that the model expects.

**Strategy A: Pass Boolean/Long Padding Mask**
Standard HF `generate` expects `attention_mask` to be a tensor of 1s (keep) and 0s (masked).
- Ensure `prefix_pad_masks` is converted to `torch.long` or `torch.bool` before passing to `generate`.
- Do NOT pass a float mask unless we are manually handling the 4D expansion (which HF `generate` usually does internally).

**Strategy B: Manual 4D Mask (Fallback)**
If Strategy A fails, we might need to construct the 4D additive attention mask (0.0 for keep, -inf for masked) in `bfloat16` and pass it. However, `generate` might overwrite or misinterpret this if not passed as `attention_mask` with correct flags (or it might not support 4D masks in `generate` easily).


### 5. Suggestions from GitHub Issues and PI05 author:

- [ ] https://github.com/open-pi/openpi/issues/679
- [ ] https://github.com/open-pi/openpi/issues/701


Hi Karl! I would like to ask for your help on pi05 subtask prediction ability
As mentioned  issue#679, issue#647 etc， I tried to implement an AutoRegression head for PI05, but I don't know where should I adapt the model arch.
Could I know when will PI release this 'let model talk' ability? As per my owngoing reasoning project at Penn, I hope to play with the text prediction head. Open-source this part will be very exciting!
If PI decide to only open-source action decoding for now, I wonder if the model ckpt has this ability thus I can post-train one with my own AR head & text loss
Karl
  4:35 PM
I don't think we have a clear timeline for releasing AR decoding capabilities for pi05 -- the model is trained with it tho, so it shoudl be pretty easy to finetune it with your own prompts + data for a few steps to get it to produce text output
4:35
would need to implement AR decoding for pi05 though (no new head required) -- you can take inspiration from pi0-FAST model for how to implement it
Karl
  4:47 PM
sounds cool!  you can check the pi-FAST implementation so that may be easier

**Recommended Action**:
Modify `src/openpi/models_pytorch/pi0_pytorch.py`:
- Cast `prefix_pad_masks` to `torch.long` (or `int64`).
- Ensure it is on the correct device.

### Step 2: Verify Tokenizer & Output
- Ensure the output tokens are decoded correctly using the SentencePiece tokenizer.
- Verify the prompt format (e.g., does it need `\n` at the end? Yes, typically).

## 5. Verification
- Run `uv run viz/h2_2_vqa.py`.
- Check if "duck" or "bowl" appears in the output.

