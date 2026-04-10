from typing import Literal

import pytest
import numpy as np
import os
import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

# ── In-RAM attention capture ───────────────────────────────────────────────────
# Usage (pipeline.py):
#   _gpt.enable_attn_buffer()         # arm prefix capture
#   _gpt.enable_suffix_attn_buffer(capture_steps=1)  # arm suffix capture (first N NFE steps)
#   result = policy.infer(example)
#   buf        = _gpt.get_attn_buffer()
#   suffix_buf = _gpt.get_suffix_attn_buffer()
#   _gpt.clear_attn_buffer()          # always in finally block
#   _gpt.clear_suffix_attn_buffer()   # always in finally block
#   write_attn_h5_from_buffer(buf, suffix_attn_buffer=suffix_buf, ...)
#
# Prefix buffer: captures PaliGemma (prefix-only forward, Case 1).
# Suffix buffer: captures gemma_expert action-token forward (Case 2).
#   Shape per layer: (1, n_heads, 8_action_steps, prefix_seq_len+8)
#   The first 512 columns are image-patch positions (ext 0:256, wrist 256:512).

_ATTN_BUFFER: dict[int, np.ndarray] | None = None
_ATTN_BUFFER_NOTIFIED: bool = False

_SUFFIX_ATTN_BUFFER: dict[int, np.ndarray] | None = None
_SUFFIX_ATTN_BUFFER_NOTIFIED: bool = False
_SUFFIX_ATTN_BUFFER_CALL_COUNT: int = 0   # NFE steps seen so far this infer()
_SUFFIX_ATTN_BUFFER_CAPTURE_STEPS: int = 1  # how many early steps to average


def enable_attn_buffer() -> None:
    """Arm the in-RAM prefix attention buffer. Call once before each policy.infer()."""
    global _ATTN_BUFFER, _ATTN_BUFFER_NOTIFIED
    _ATTN_BUFFER = {}
    if not _ATTN_BUFFER_NOTIFIED:
        print("[attn] Prefix attention capture enabled — prefix attention will be held in RAM.")
        _ATTN_BUFFER_NOTIFIED = True


def get_attn_buffer() -> dict[int, np.ndarray] | None:
    """Return the prefix buffer dict {layer_idx: ndarray} or None if not armed."""
    return _ATTN_BUFFER


def clear_attn_buffer() -> None:
    """Disarm and discard the prefix buffer. Call in a finally block after infer()."""
    global _ATTN_BUFFER
    _ATTN_BUFFER = None


def enable_suffix_attn_buffer(capture_steps: int = 1) -> None:
    """Arm the in-RAM suffix (action-token) attention buffer.

    Args:
        capture_steps: Number of early denoising NFE steps to capture and average.
            Subsequent steps are ignored.  Default 1 captures only the first step.
            Set to a large number (e.g. 999) to capture and average all steps.
    """
    global _SUFFIX_ATTN_BUFFER, _SUFFIX_ATTN_BUFFER_NOTIFIED
    global _SUFFIX_ATTN_BUFFER_CALL_COUNT, _SUFFIX_ATTN_BUFFER_CAPTURE_STEPS
    _SUFFIX_ATTN_BUFFER = {}
    _SUFFIX_ATTN_BUFFER_CALL_COUNT = 0
    _SUFFIX_ATTN_BUFFER_CAPTURE_STEPS = capture_steps
    if not _SUFFIX_ATTN_BUFFER_NOTIFIED:
        print("[attn] Suffix attention capture enabled — action-token attention will be held in RAM.")
        _SUFFIX_ATTN_BUFFER_NOTIFIED = True


def get_suffix_attn_buffer() -> dict[int, np.ndarray] | None:
    """Return the suffix buffer dict {layer_idx: ndarray} or None if not armed."""
    return _SUFFIX_ATTN_BUFFER


def get_suffix_attn_buffer_step_count() -> int:
    """Return how many NFE steps were captured in the last infer() call."""
    return _SUFFIX_ATTN_BUFFER_CALL_COUNT


def clear_suffix_attn_buffer() -> None:
    """Disarm and discard the suffix buffer. Call in a finally block after infer()."""
    global _SUFFIX_ATTN_BUFFER, _SUFFIX_ATTN_BUFFER_CALL_COUNT
    _SUFFIX_ATTN_BUFFER = None
    _SUFFIX_ATTN_BUFFER_CALL_COUNT = 0


# ── Per-step suffix buffer (stores every NFE step separately) ─────────────────
# Use this when you need to analyze how attention evolves over denoising.
# Each entry in the list is one NFE step: {layer_idx: ndarray(1, n_heads, 8, k)}.

_SUFFIX_ATTN_STEPS_BUFFER: list[dict[int, np.ndarray]] | None = None


def enable_suffix_attn_steps_buffer() -> None:
    """Arm per-step suffix attention capture.

    Every NFE call appends a fresh {layer_idx: ndarray} snapshot to the list.
    Use get_suffix_attn_steps_buffer() after infer() to retrieve all steps.
    Can be used together with enable_suffix_attn_buffer().
    """
    global _SUFFIX_ATTN_STEPS_BUFFER
    _SUFFIX_ATTN_STEPS_BUFFER = []


def get_suffix_attn_steps_buffer() -> list[dict[int, np.ndarray]] | None:
    """Return list of per-step attention dicts, or None if not armed.

    Index 0 = first denoising step (most noise), index -1 = last step (clean action).
    Each dict: {layer_idx: ndarray(1, n_heads, 8_action_steps, seq_len)}.
    """
    return _SUFFIX_ATTN_STEPS_BUFFER


def clear_suffix_attn_steps_buffer() -> None:
    """Disarm and discard the per-step buffer. Call in a finally block after infer()."""
    global _SUFFIX_ATTN_STEPS_BUFFER
    _SUFFIX_ATTN_STEPS_BUFFER = None


# ── Action trajectory buffer (stores x_t after each Euler step) ───────────────
# Each entry is float32(action_horizon, action_dim) — the denoised action chunk
# after that NFE step.  Index 0 = after 1st step (least denoised),
# index -1 = after last step (== pred_action / final clean action).

_ACTION_TRAJ_BUFFER: list[np.ndarray] | None = None


def enable_action_traj_buffer() -> None:
    """Arm the action trajectory buffer. Call once before each policy.infer()."""
    global _ACTION_TRAJ_BUFFER
    _ACTION_TRAJ_BUFFER = []


def append_action_traj(x_t: np.ndarray) -> None:
    """Append x_t (after one Euler step) to the buffer. Called from pi0_pytorch."""
    if _ACTION_TRAJ_BUFFER is not None:
        _ACTION_TRAJ_BUFFER.append(x_t)


def get_action_traj_buffer() -> list[np.ndarray] | None:
    """Return list of per-step action arrays, or None if not armed.

    Index 0 = after 1st denoising step (mostly noise),
    index -1 = final clean action (== pred_action).
    Each entry: float32(action_horizon, action_dim).
    """
    return _ACTION_TRAJ_BUFFER


def clear_action_traj_buffer() -> None:
    """Disarm and discard the action trajectory buffer. Call in a finally block."""
    global _ACTION_TRAJ_BUFFER
    _ACTION_TRAJ_BUFFER = None


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        global _SUFFIX_ATTN_BUFFER_CALL_COUNT
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
                output_attentions=True,
            )

            # --- Capture attention into RAM buffer ---
            if not self.training and _ATTN_BUFFER is not None:
                if prefix_output.attentions is not None:
                    for i, layer_attn in enumerate(prefix_output.attentions):
                        _ATTN_BUFFER[i] = layer_attn.detach().cpu().to(torch.float32).numpy()
            # ----------------------------------------

            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
                output_attentions=True,
            )

            # --- Capture suffix attention into RAM buffer ---
            # Only accumulate the first _SUFFIX_ATTN_BUFFER_CAPTURE_STEPS NFE steps.
            # After that the buffer holds their per-layer average and further steps
            # are ignored.  Accumulation: running sum → divide on the final step.
            if not self.training and _SUFFIX_ATTN_BUFFER is not None:
                if suffix_output.attentions is not None:
                    if _SUFFIX_ATTN_BUFFER_CALL_COUNT < _SUFFIX_ATTN_BUFFER_CAPTURE_STEPS:
                        for i, layer_attn in enumerate(suffix_output.attentions):
                            arr = layer_attn.detach().cpu().to(torch.float32).numpy()
                            if i in _SUFFIX_ATTN_BUFFER:
                                _SUFFIX_ATTN_BUFFER[i] = _SUFFIX_ATTN_BUFFER[i] + arr
                            else:
                                _SUFFIX_ATTN_BUFFER[i] = arr
                    _SUFFIX_ATTN_BUFFER_CALL_COUNT += 1
                    # Normalize to an average on the last captured step
                    if _SUFFIX_ATTN_BUFFER_CALL_COUNT == _SUFFIX_ATTN_BUFFER_CAPTURE_STEPS:
                        n = _SUFFIX_ATTN_BUFFER_CAPTURE_STEPS
                        for i in _SUFFIX_ATTN_BUFFER:
                            _SUFFIX_ATTN_BUFFER[i] = _SUFFIX_ATTN_BUFFER[i] / n
            # ------------------------------------------------

            # --- Per-step capture (appends every NFE step to a list) ---------
            if not self.training and _SUFFIX_ATTN_STEPS_BUFFER is not None:
                if suffix_output.attentions is not None:
                    step_snap: dict[int, np.ndarray] = {}
                    for i, layer_attn in enumerate(suffix_output.attentions):
                        step_snap[i] = layer_attn.detach().cpu().to(torch.float32).numpy()
                    _SUFFIX_ATTN_STEPS_BUFFER.append(step_snap)
            # -----------------------------------------------------------------

            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, attn_weights = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )

                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
