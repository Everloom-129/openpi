import dataclasses
import os
from PIL import Image

import numpy as np

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import matplotlib.pyplot as plt


def load_duck_example(camera: str = "left", index: int = 0):
    """Load example from duck dataset."""

    if camera == "left":
        camera = "varied_camera_1"
    elif camera == "right":
        camera = "varied_camera_2"
    else:
        raise ValueError("camera must be 'left' or 'right'")

    if not 0 <= index <= 90:
        raise ValueError("index must be between 0 and 90")

    data_dir = "data/visualization/duck/frames"

    # Load images
    ext_path = os.path.join(data_dir, camera, f"{index:05d}.jpg")
    hand_path = os.path.join(data_dir, "hand_camera", f"{index:05d}.jpg")

    ext_img = np.array(Image.open(ext_path))
    hand_img = np.array(Image.open(hand_path))

    return {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": hand_img,
        "observation/joint_position": np.random.rand(7),  # Placeholder
        "observation/gripper_position": np.random.rand(1),  # Placeholder
        "prompt": "do something",
    }


def vis_example(example: dict, name: str):
    os.makedirs(f"results/{name}", exist_ok=True)
    fig = plt.figure(figsize=(12, 8))

    plt.suptitle(example["prompt"], fontsize=16)

    plt.subplot(2, 2, 1)
    plt.imshow(example["observation/exterior_image_1_left"])
    plt.title("Exterior Camera")

    plt.subplot(2, 2, 2)
    plt.imshow(example["observation/wrist_image_left"])
    plt.title("Wrist Camera")

    plt.subplot(2, 2, 3)
    plt.bar(range(len(example["observation/joint_position"])), example["observation/joint_position"])
    plt.title("Joint Positions")

    plt.subplot(2, 2, 4)
    plt.bar(["Gripper"], example["observation/gripper_position"])
    plt.title("Gripper Position")

    plt.tight_layout()
    fig_path = f"results/{name}/origin_{example['prompt']}.jpg"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved example to {fig_path}")


if __name__ == "__main__":


    config = _config.get_config("pi05_droid")
    checkpoint_dir = "/home/tony/projects/openpi/checkpoints/viz/pi05_droid_pytorch"

    # Create a trained policy (automatically detects PyTorch format)
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    camera = "left"
    for index in [0, 85, 90]:
        print(f"VLA index {index}")
        example = load_duck_example(camera=camera, index=index)
        vis_example(example, f"duck_{camera}_{index}")
        breakpoint()
        result = policy.infer(example)
        print("Actions shape:", result["actions"].shape)

    del policy
