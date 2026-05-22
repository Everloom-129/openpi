"""Input/output transforms for the pi0.5 robocasa365 checkpoint.

Mirrors the upstream robocasa-benchmark/openpi reference at
`src/openpi/policies/robocasa_policy.py` and `examples/robocasa/main.py` so
that the schema seen by the model is identical to training.

State (16-D, concatenated in this order):
    [eef_pos_rel(3), eef_rot_rel(4 quat), base_pos(3), base_rot(4 quat),
     gripper_qpos(2)]

Action (12-D, layout B from `robocasa/utils/env_utils.py:convert_action`):
    [eef_pos(3), eef_rot_axisangle(3), gripper_close(1), base_motion(4),
     control_mode(1)]
"""
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robocasa_example() -> dict:
    return {
        "observation/state": np.random.rand(16).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"RobocasaInputs supports PI0/PI05 only, got {self.model_type}")

        state = np.asarray(data["observation/state"], dtype=np.float32).reshape(-1)
        if state.shape[0] != 16:
            raise ValueError(
                f"RobocasaInputs expected 16-D state (eef_pos+eef_rot+base_pos+base_rot+gripper), "
                f"got {state.shape[0]}"
            )

        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Upstream returns the full 12-D robocasa action; the client (sim or
        # gym wrapper) re-routes [:7]=arm/gripper, [7:11]=base, [11]=control_mode.
        return {"actions": np.asarray(data["actions"][:, :12])}
