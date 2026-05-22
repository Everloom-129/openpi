from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer_cpg_jax(
        self,
        obs_pos: dict,
        obs_neg: dict,
        *,
        cpg_w: float,
        noise: np.ndarray | None = None,
    ) -> dict:
        """DeLock contrastive prompt guidance inference (JAX path).

        Mirrors infer() but runs the input transform on both prompts (which
        share images / state and only differ in the tokenized prompt) and
        routes to ``model.sample_actions_cpg``. Returns the same output dict
        shape as infer().
        """
        if self._is_pytorch_model:
            raise NotImplementedError("Use infer_cpg() for the PyTorch path.")
        inputs_pos = jax.tree.map(lambda x: x, obs_pos)
        inputs_neg = jax.tree.map(lambda x: x, obs_neg)
        inputs_pos = self._input_transform(inputs_pos)
        inputs_neg = self._input_transform(inputs_neg)
        inputs_pos = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs_pos)
        inputs_neg = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs_neg)

        sample_kwargs: dict[str, Any] = dict(self._sample_kwargs)
        if noise is not None:
            n = jnp.asarray(noise)
            if n.ndim == 2:
                n = n[None, ...]
            sample_kwargs["noise"] = n
        sample_kwargs["cpg_w"] = float(cpg_w)

        observation_pos = _model.Observation.from_dict(inputs_pos)
        observation_neg = _model.Observation.from_dict(inputs_neg)

        self._rng, sub_rng = jax.random.split(self._rng)
        start_time = time.monotonic()
        # Note: not module_jit'd; CPG is opt-in and the dual-prefix path is rare
        # enough that re-tracing each call is acceptable. If you sweep many w,
        # the inner while_loop dominates compile cost is amortized inside it.
        actions = self._model.sample_actions_cpg(sub_rng, observation_pos, observation_neg, **sample_kwargs)
        outputs = {"state": inputs_pos["state"], "actions": actions}
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    def infer_cpg(
        self,
        obs_pos: dict,
        obs_neg: dict,
        *,
        cpg_w: float,
        noise: np.ndarray | None = None,
    ) -> dict:
        """DeLock contrastive prompt guidance inference. PyTorch-only.

        ``obs_pos`` / ``obs_neg`` are two raw observations that share images +
        state, differing only in the ``"prompt"`` field. Both are run through
        the same input transform (which tokenizes the prompt), then handed to
        ``model.sample_actions_cpg`` which dual-forwards the prefix and
        combines the per-step vector fields with weight ``cpg_w``.

        ``cpg_w == 1.0`` is identical to ``infer(obs_pos)``; ``cpg_w == 0.0``
        recovers ``infer(obs_neg)``. Paper uses ``cpg_w > 1`` (extrapolation).
        """
        if not self._is_pytorch_model:
            raise NotImplementedError("CPG path is implemented only for the PyTorch model.")

        inputs_pos = jax.tree.map(lambda x: x, obs_pos)
        inputs_neg = jax.tree.map(lambda x: x, obs_neg)
        inputs_pos = self._input_transform(inputs_pos)
        inputs_neg = self._input_transform(inputs_neg)
        inputs_pos = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs_pos
        )
        inputs_neg = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs_neg
        )

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_t = torch.from_numpy(noise).to(self._pytorch_device)
            if noise_t.ndim == 2:
                noise_t = noise_t[None, ...]
            sample_kwargs["noise"] = noise_t

        observation_pos = _model.Observation.from_dict(inputs_pos)
        observation_neg = _model.Observation.from_dict(inputs_neg)

        start_time = time.monotonic()
        actions = self._model.sample_actions_cpg(
            self._pytorch_device,
            observation_pos,
            observation_neg,
            cpg_w=cpg_w,
            **sample_kwargs,
        )
        outputs = {"state": inputs_pos["state"], "actions": actions}
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
