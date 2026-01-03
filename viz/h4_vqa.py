import os
import sys
import torch
import numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models import model as _model

try:
    from viz.attn_map import load_duck_example
except ImportError:
    from attn_map import load_duck_example
import jax


def main():
    # 1. Load Policy
    print("Loading Pi05 model (this may take a while)...")
    config = _config.get_config("pi05_droid")
    checkpoint_dir = "./checkpoints/viz/pi05_droid_pytorch"
    device = "cuda:1"
    # Create policy
    policy = _policy_config.create_trained_policy(config, checkpoint_dir, pytorch_device=device)

    # Verify PyTorch model
    if not policy._is_pytorch_model:
        print("Error: This script requires a PyTorch model.")
        return

    model = policy._model
    tokenizer = PaligemmaTokenizer()

    # 2. Load Example Data
    print("Loading example observation...")
    # Using index 0 from left camera as default context
    example = load_duck_example(camera="left", index=0)

    print("\n" + "=" * 50)
    print("🤖 Pi05 VQA Interface (Automated Test)")
    print(f"Context: Duck and Bowl image loaded.")
    print("=" * 50 + "\n")

    test_prompts = ["Pick up the duck toy", "Pick up the pink bowl", "What color is the bowl?"]

    for user_input in test_prompts:
        print(f"User: {user_input}")

        try:
            # Update prompt in example
            current_example = example.copy()
            current_example["prompt"] = user_input

            # 3. Preprocess inputs using policy's transform chain
            # This handles resizing, normalization, and tokenization (hopefully)
            inputs = jax.tree.map(lambda x: x, current_example)
            inputs = policy._input_transform(inputs)

            # 4. Convert to PyTorch tensors and move to device
            device = policy._pytorch_device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)

            # 5. Wrap in Observation dataclass/struct
            # The model expects an Observation object, not a dict
            observation = _model.Observation.from_dict(inputs)

            # 6. Generate Response
            # We use the generate method added to PI0Pytorch
            # It returns the generated token IDs
            print("Pi05: ...", end=" ")
            output_ids = model.generate(
                observation,
                max_new_tokens=64,
                min_new_tokens=5,  # Force generation to see what happens
                do_sample=False,  # Greedy decoding for debugging
            )

            # 7. Decode Output
            if isinstance(output_ids, torch.Tensor):
                output_ids = output_ids.cpu().numpy().tolist()

            print(f"IDs: {output_ids}")  # Print IDs for debugging

            # Check what ID 1 is
            try:
                print(f"ID 1 is: {tokenizer._tokenizer.id_to_piece(1)}")
                print(f"ID 2 is: {tokenizer._tokenizer.id_to_piece(2)}")
            except:
                pass

            # Use sentencepiece processor to decode
            sp_processor = tokenizer._tokenizer

            # Handle batch output (though we only have batch size 1)
            for seq in output_ids:
                decoded_text = sp_processor.decode(seq)
                # Clean up text
                decoded_text = decoded_text.replace("\n", " ").strip()
                print(f"{decoded_text}")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()

        print("-" * 30)


if __name__ == "__main__":
    main()
