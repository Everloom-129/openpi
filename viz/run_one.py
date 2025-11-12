import os
import math
import json
import numpy as np
import torch

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.models.model import IMAGE_KEYS


def capture_pi_activations(
    config_name: str,
    checkpoint_dir: str,
    output_path: str,
    sample: dict | None = None,
    device: str | None = None,
):
    """
    Loads a π0/π0.5 checkpoint, freezes weights, runs one forward with hooks, and saves:
      - A_vis[l]: per-layer vision attentions (SigLIP)
      - A_txt[l]: per-layer language attentions (PaliGemma text model; prefix-only)
      - A_xattn[l]: text↔vision slice from language attentions (if present)
      - T_vis: vision token embeddings arranged in patch grid (per camera)
      - T_txt: text input token embeddings (per token)

    output_path: file (.pt) to save a dict containing these tensors (as CPU numpy).
    """
    # 1) Load policy (auto-detects PyTorch if checkpoint has model.safetensors)
    cfg = _config.get_config(config_name)
    pol = policy_config.create_trained_policy(cfg, checkpoint_dir)

    if not getattr(pol, "_is_pytorch_model", False):
        raise RuntimeError("This helper currently supports the PyTorch pipeline only.")

    # 2) Freeze
    model = pol._model  # openpi.models_pytorch.pi0_pytorch.PI0Pytorch
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Choose device
    use_device = device or (
        pol._pytorch_device if hasattr(pol, "_pytorch_device") else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(use_device)

    # 3) Enable attentions/hidden states on VLM components
    #    - Vision (SigLIP)
    #    - Language (PaliGemma text/backbone)
    #    - Expert Gemma (suffix; optional, not required for A_xattn)
    vlm = model.paligemma_with_expert
    vlm.paligemma.vision_tower.config.output_attentions = True
    vlm.paligemma.vision_tower.config.output_hidden_states = True

    vlm.paligemma.language_model.config.output_attentions = True
    vlm.paligemma.language_model.config.output_hidden_states = True
    vlm.paligemma.language_model.config._attn_implementation = "eager"  # ensure eager attn

    vlm.gemma_expert.model.config.output_attentions = True
    vlm.gemma_expert.model.config.output_hidden_states = True
    vlm.gemma_expert.model.config._attn_implementation = "eager"

    # 4) Buffers populated by hooks
    buffers = {
        "vision": [],  # list of per-camera BaseModelOutputWithPooling (attentions, hidden_states)
        "language": None,  # PaliGemma language forward output (attentions, hidden_states) for prefix
        "expert": None,  # optional: expert outputs (not used for A_xattn)
    }

    # Vision hook: called once per camera image when embed_image() runs
    def _vision_hook(_module, _inp, out):
        # out is transformers BaseModelOutputWithPooling
        # Store a lightweight copy (attentions + hidden_states only)
        buffers["vision"].append(
            {
                "attentions": tuple(a.detach().cpu() for a in (out.attentions or ())),
                "hidden_states": tuple(h.detach().cpu() for h in (out.hidden_states or ())),
            }
        )

    # Language hook: called during prefix-only language pass in sample_actions()
    def _language_hook(_module, _inp, out):
        # out is transformers CausalLM output (ModelOutput with attentions/hidden_states)
        if buffers["language"] is None:
            buffers["language"] = {
                "attentions": tuple(a.detach().cpu() for a in (out.attentions or ())),
                "hidden_states": tuple(h.detach().cpu() for h in (out.hidden_states or ())),
                "last_hidden_state": out.last_hidden_state.detach().cpu()
                if hasattr(out, "last_hidden_state")
                else None,
            }

    # Expert hook (optional; not required for requested outputs)
    def _expert_hook(_module, _inp, out):
        if buffers["expert"] is None and hasattr(out, "attentions"):
            buffers["expert"] = {
                "attentions": tuple(a.detach().cpu() for a in (out.attentions or ())),
                "hidden_states": tuple(h.detach().cpu() for h in (out.hidden_states or ())),
            }

    # Register hooks
    vhook = vlm.paligemma.vision_tower.register_forward_hook(_vision_hook)
    lhook = vlm.paligemma.language_model.register_forward_hook(_language_hook)
    ehook = vlm.gemma_expert.model.register_forward_hook(_expert_hook)

    # 5) Build a single-sample input if not provided
    # For DROID policies, the input transform expects raw observation keys
    # like "observation/exterior_image_1_left", not the canonical image/state
    # dict. Provide a minimal compatible example.
    if sample is None:
        H, W = 224, 224
        base = np.zeros((H, W, 3), dtype=np.uint8)
        wrist = np.zeros((H, W, 3), dtype=np.uint8)
        sample = {
            "observation/exterior_image_1_left": base,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": np.zeros((7,), dtype=np.float32),
            "observation/gripper_position": np.zeros((1,), dtype=np.float32),
            "prompt": "pick up the fork",
        }

    # 6) One inference call (runs vision + prefix language; enough for requested captures)
    _ = pol.infer(sample)

    # 7) Remove hooks
    vhook.remove()
    lhook.remove()
    ehook.remove()

    # 8) Post-process into requested artifacts
    out = {}
    # A_vis[l] per vision layer, and T_vis per camera in patch grid
    # Each vision call yields:
    # - hidden_states: tuple(layer_states...), where hidden_states[0] == input embeddings to encoder
    # - attentions: tuple(layer_attn...), each [B, heads, N, N]
    # We map captures to IMAGE_KEYS order (embed_prefix iterates in this canonical order)
    cam_keys_in_order = IMAGE_KEYS  # ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    A_vis_all_cams = {}
    T_vis_all_cams = {}
    n_img_tokens_per_cam = []

    for idx, cam_key in enumerate(cam_keys_in_order):
        if idx >= len(buffers["vision"]):
            continue  # camera might have been skipped entirely
        vis = buffers["vision"][idx]
        # attentions per layer
        A_vis_all_cams[cam_key] = [a.numpy() for a in vis["attentions"]]

        # token embeddings at encoder input (after patch+pos embed)
        if vis["hidden_states"] and len(vis["hidden_states"]) > 0:
            t0 = vis["hidden_states"][0].numpy()  # [B, N, D]
            B, N, D = t0.shape
            # infer patch grid
            side = int(math.isqrt(N))
            if side * side != N:
                # fall back to flat if not square
                T_vis_all_cams[cam_key] = t0  # [B, N, D]
            else:
                T_vis_all_cams[cam_key] = t0.reshape(B, side, side, D)  # [B, H_p, W_p, D]
            n_img_tokens_per_cam.append(N)
        else:
            n_img_tokens_per_cam.append(0)

    out["A_vis"] = A_vis_all_cams
    out["T_vis"] = T_vis_all_cams

    # A_txt[l] per text layer, T_txt per token, and A_xattn[l] (text↔vision slice)
    if buffers["language"] is None:
        # If nothing captured (unlikely), create empties
        out["A_txt"] = []
        out["A_xattn"] = []
        out["T_txt"] = None
    else:
        lang = buffers["language"]
        A_txt = [a.numpy() for a in (lang["attentions"] or ())]
        out["A_txt"] = A_txt

        # Compute text/vision split on prefix sequence
        n_img = int(sum(n_img_tokens_per_cam)) if n_img_tokens_per_cam else 0
        # attention tensors have shape [B, H, Lq, Lk]
        A_xattn = []
        if A_txt:
            L = A_txt[0].shape[-1]
            if L >= n_img:
                # slice: queries = text positions [n_img:, :], keys = vision positions [:, :n_img]
                for a in A_txt:
                    A_xattn.append(a[:, :, n_img:, :n_img])
        out["A_xattn"] = A_xattn

        # T_txt: input token embeddings (language) at positions [n_img:]
        T_txt = None
        if lang["hidden_states"] and len(lang["hidden_states"]) > 0:
            x0 = lang["hidden_states"][0].numpy()  # [B, L, D], inputs_embeds
            T_txt = x0[:, n_img:, :]
        out["T_txt"] = T_txt

    # 9) Save to disk (numpy on CPU)
    # Convert nested dicts/lists of ndarrays
    def to_serializable(x):
        if isinstance(x, dict):
            return {k: to_serializable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_serializable(v) for v in x]
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return x

    serializable = to_serializable(out)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(serializable, output_path)
    return serializable


if __name__ == "__main__":
    result = capture_pi_activations(
        config_name="pi05_droid",
        checkpoint_dir="/home/tony/projects/openpi/checkpoints/viz/pi05_droid_pytorch",  # or local converted PyTorch path
        output_path="pi05_buffers.pt",
        sample=None,  # use default dummy sample
        device="cuda:0",
    )
    print("Saved:", "checkpoints/pi05_buffers.pt")
