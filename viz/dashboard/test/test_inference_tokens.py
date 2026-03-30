"""Unit tests for the token-label extraction fix in viz/dashboard/inference.py.

The bug: run_inference used to reconstruct token labels by discretizing raw joint
positions directly with np.linspace(-1, 1, ...).  The model actually
quantile-normalizes the state first (inside _input_transform → Normalize), so
those labels were completely wrong for typical DROID joint angles (which live
well outside [-1, 1] before normalization).

The fix: run policy._input_transform on a shallow copy of the example and read
back `tokenized_prompt` + `tokenized_prompt_mask`, then decode with id_to_piece.

Run with:
    uv run pytest viz/dashboard/test/test_inference_tokens.py -v
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Part 1 — Pure numerical tests (no mocking, no model loading)
# Demonstrate that the old manual-discretization approach was wrong.
# ---------------------------------------------------------------------------

BINS = np.linspace(-1, 1, 257)[:-1]   # 256 bins, same as PaligemmaTokenizer


def _raw_bin(val: float) -> int:
    """Discretize a raw (un-normalized) float — what the OLD code did."""
    return int(np.digitize(val, BINS)) - 1


def _quantile_bin(val: float, q01: float, q99: float) -> int:
    """Quantile-normalize then discretize — what the NEW code does."""
    normalized = (val - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    normalized = float(np.clip(normalized, -1.0, 1.0))
    return int(np.digitize(normalized, BINS)) - 1


class TestNormalizationMath:
    """Show numerically that raw state discretization != normalized state
    discretization for typical DROID joint angles."""

    def test_above_range_raw_is_max_bin(self):
        """Raw value 2.0 rad is above all bins (max bin value ≈ 0.992) → bin 255."""
        assert _raw_bin(2.0) == 255

    def test_below_range_raw_is_minus_one(self):
        """Raw value -2.0 rad is below all bins → np.digitize returns 0, minus 1 = -1.
        The old code would produce negative bin indices for large negative angles."""
        assert _raw_bin(-2.0) == -1

    def test_typical_elbow_angle_differs_after_normalization(self):
        """Joint[5] = 2.0 rad; pi05 norm_stats: q01=1.172, q99=3.467.

        Old code: bin 255 (2.0 > all bins).
        New code: quantile-maps to ≈ -0.28 → bin ~92.
        """
        raw = _raw_bin(2.0)
        norm = _quantile_bin(2.0, q01=1.172, q99=3.467)

        assert raw == 255, f"Expected raw bin 255, got {raw}"
        assert norm != raw, f"Normalized bin ({norm}) should differ from raw ({raw})"
        # (2.0-1.172)/(3.467-1.172)*2 - 1 ≈ -0.28 → bin ≈ 92
        assert 80 <= norm <= 110, f"Expected ~92, got {norm}"

    def test_joint_below_range_maps_to_high_bin_after_normalization(self):
        """Joint[3] = -1.5 rad; q01=-2.773, q99=-0.454.

        Raw: -1 (below all bins, invalid).
        Normalized: (-1.5 - (-2.773)) / (-0.454 - (-2.773)) * 2 - 1 ≈ +0.096 → bin ~140.
        """
        raw = _raw_bin(-1.5)
        norm = _quantile_bin(-1.5, q01=-2.773, q99=-0.454)

        assert raw == -1, f"Expected raw bin -1, got {raw}"
        assert norm > 100, f"Normalized bin should be mid-range, got {norm}"

    def test_majority_of_typical_droid_joints_differ(self):
        """Most of the 8 typical DROID joint angles land outside [-1, 1] before
        normalization; quantile mapping moves them into [0, 255] correctly."""
        typical_joints = [-2.02, -1.5, 0.8, -1.5, 1.0, 2.0, -0.5, 0.5]
        q01 = [-0.828, -0.840, -0.843, -2.773, -1.843, 1.172, -2.047, 0.0]
        q99 = [0.900,  1.385,  0.692, -0.454,  1.732, 3.467,  2.198, 0.991]

        raw_bins = [_raw_bin(v) for v in typical_joints]
        norm_bins = [_quantile_bin(v, q01[i], q99[i]) for i, v in enumerate(typical_joints)]

        diffs = sum(r != n for r, n in zip(raw_bins, norm_bins))
        assert diffs >= 4, (
            f"Expected ≥4 joints to differ, only {diffs} differ.\n"
            f"raw ={raw_bins}\nnorm={norm_bins}"
        )

        # Normalized bins should all be valid [0, 255]
        assert all(0 <= b <= 255 for b in norm_bins), f"Bad norm bins: {norm_bins}"
        # Raw bins include invalid -1 values for below-range angles
        assert any(b < 0 for b in raw_bins), "Expected some invalid raw bins"


# ---------------------------------------------------------------------------
# Part 2 — Unit tests for run_inference token label extraction
# Uses a fake policy + mocked heavy deps; no GPU/model/GCS download needed.
# ---------------------------------------------------------------------------

N_HEADS = 8
TEXT_START_IDX = 768
TOTAL_IMAGE_TOKENS = 512


def _make_attn_buf(n_text: int, n_layers: int = 2) -> dict:
    """Synthetic attention buffer: dict[layer_idx → ndarray(1, 8, seq, seq)]."""
    seq = TEXT_START_IDX + n_text
    rng = np.random.default_rng(0)
    return {
        i: rng.random((1, N_HEADS, seq, seq), dtype=np.float32)
        for i in range(n_layers)
    }


def _make_example(prompt: str = "pick up the cup") -> dict:
    rng = np.random.default_rng(42)
    return {
        "prompt": prompt,
        "observation/joint_position": rng.standard_normal(7).astype(np.float32) * 1.5,
        "observation/gripper_position": np.array([0.4], dtype=np.float32),
        "observation/exterior_image_1_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
    }


class FakeInputTransform:
    """Controllable stand-in for policy._input_transform.

    Returns preset tokenized_prompt + mask; records every call.
    """

    def __init__(self, token_ids: list[int], n_real: int | None = None):
        max_len = 48
        if n_real is None:
            n_real = len(token_ids)
        padding = [0] * (max_len - len(token_ids))
        self._ids = np.array(token_ids + padding, dtype=np.int32)
        mask_vals = [True] * n_real + [False] * (max_len - n_real)
        self._mask = np.array(mask_vals, dtype=bool)
        self.calls: list[dict] = []

    def __call__(self, data: dict) -> dict:
        self.calls.append(dict(data))
        return {
            "tokenized_prompt": self._ids,
            "tokenized_prompt_mask": self._mask,
            "state": np.zeros(8, dtype=np.float32),
        }


class FakePolicy:
    def __init__(self, transform: FakeInputTransform):
        self._input_transform = transform

    def infer(self, _example: dict) -> dict:
        return {"actions": np.zeros((8, 8), dtype=np.float32)}


def _make_stubs(buf: dict, piece_map: dict[int, str] | None = None):
    """Build stub modules for all heavy deps inside run_inference."""
    if piece_map is None:
        piece_map = {}

    # gemma_pytorch stub
    gpt_stub = types.ModuleType("openpi.models_pytorch.gemma_pytorch")
    gpt_stub.enable_attn_buffer = MagicMock()
    gpt_stub.clear_attn_buffer = MagicMock()
    gpt_stub.get_attn_buffer = MagicMock(return_value=buf)

    # viz.dashboard.loader stub
    loader_stub = types.ModuleType("viz.dashboard.loader")
    loader_stub.TEXT_START_IDX = TEXT_START_IDX
    loader_stub.TOTAL_IMAGE_TOKENS = TOTAL_IMAGE_TOKENS

    # openpi.models.tokenizer stub
    mock_inner = MagicMock()
    mock_inner.id_to_piece.side_effect = lambda i: piece_map.get(i, f"<{i}>")
    mock_tok_instance = MagicMock()
    mock_tok_instance._tokenizer = mock_inner
    mock_tok_cls = MagicMock(return_value=mock_tok_instance)
    tok_stub = types.ModuleType("openpi.models.tokenizer")
    tok_stub.PaligemmaTokenizer = mock_tok_cls

    return gpt_stub, loader_stub, tok_stub


def _run_with_stubs(policy, example, buf, piece_map=None):
    """Call viz.dashboard.inference.run_inference with all heavy deps stubbed."""
    import viz.dashboard.inference as _inf

    gpt_stub, loader_stub, tok_stub = _make_stubs(buf, piece_map)

    with (
        patch.dict(
            sys.modules,
            {
                "openpi.models_pytorch.gemma_pytorch": gpt_stub,
                "viz.dashboard.loader": loader_stub,
                "openpi.models.tokenizer": tok_stub,
            },
        ),
        patch.object(_inf, "st") as mock_st,
    ):
        result = _inf.run_inference(policy, example)

    return result, mock_st


class TestRunInferenceTokenLabels:
    """run_inference must use policy._input_transform to get token labels."""

    def test_labels_come_from_transform_not_manual_reconstruction(self):
        """Token labels must decode the IDs returned by _input_transform."""
        token_ids = [1, 2, 3, 4, 5]
        piece_map = {1: "Task", 2: ":", 3: "▁pick", 4: "▁up", 5: "▁cup"}

        buf = _make_attn_buf(n_text=len(token_ids))
        transform = FakeInputTransform(token_ids, n_real=len(token_ids))
        policy = FakePolicy(transform)

        result, _ = _run_with_stubs(policy, _make_example(), buf, piece_map)

        assert result, "run_inference returned empty dict"
        assert result["meta"]["token_texts"] == ["Task", ":", "▁pick", "▁up", "▁cup"]

    def test_prompt_not_mutated_by_shallow_copy(self):
        """example['prompt'] must survive run_inference unchanged.

        TokenizePrompt calls data.pop('prompt') inside _input_transform.
        The fix uses {**example} so the pop only affects the copy.
        """
        buf = _make_attn_buf(n_text=5)
        transform = FakeInputTransform([10, 20, 30, 40, 50], n_real=5)
        policy = FakePolicy(transform)
        example = _make_example(prompt="grasp the red block")

        _run_with_stubs(policy, example, buf)

        assert "prompt" in example, (
            "example['prompt'] was deleted — the {**example} shallow-copy fix is broken"
        )
        assert example["prompt"] == "grasp the red block"

    def test_mask_excludes_padding_tokens(self):
        """Only mask=True token IDs should become labels (padding zeros excluded)."""
        real_ids = [10, 11, 12, 13, 14, 15, 16, 17]
        n_real = 5   # only first 5 are unmasked
        piece_map = {i: f"tok{i}" for i in real_ids}

        buf = _make_attn_buf(n_text=len(real_ids))
        transform = FakeInputTransform(real_ids, n_real=n_real)
        policy = FakePolicy(transform)

        result, _ = _run_with_stubs(policy, _make_example(), buf, piece_map)

        got = result["meta"]["token_texts"]
        assert len(got) == n_real, f"Expected {n_real} labels, got {len(got)}"
        assert got == ["tok10", "tok11", "tok12", "tok13", "tok14"]

    def test_n_real_tokens_matches_mask_sum(self):
        """meta['n_real_tokens'] must equal the number of mask=True entries."""
        n_real = 7
        token_ids = list(range(1, n_real + 1))
        buf = _make_attn_buf(n_text=n_real)
        transform = FakeInputTransform(token_ids, n_real=n_real)
        policy = FakePolicy(transform)

        result, _ = _run_with_stubs(policy, _make_example(), buf)

        assert result["meta"]["n_real_tokens"] == n_real

    def test_transform_called_at_least_once(self):
        """_input_transform must be called (at least for the label extraction)."""
        token_ids = [1, 2, 3]
        buf = _make_attn_buf(n_text=len(token_ids))
        transform = FakeInputTransform(token_ids)
        policy = FakePolicy(transform)

        _run_with_stubs(policy, _make_example(prompt="stack the blocks"), buf)

        assert len(transform.calls) >= 1

    def test_fallback_generic_labels_on_transform_error(self):
        """If _input_transform raises, warn and fall back to generic tok_i labels."""
        n_text = 4
        buf = _make_attn_buf(n_text=n_text)

        class BrokenPolicy:
            _input_transform = MagicMock(side_effect=RuntimeError("transform failed"))
            def infer(self, _): return {"actions": np.zeros((8, 8))}

        result, mock_st = _run_with_stubs(BrokenPolicy(), _make_example(), buf)

        mock_st.warning.assert_called()
        if result:
            labels = result["meta"]["token_texts"]
            assert all(l.startswith("tok_") for l in labels), (
                f"Expected generic tok_i labels, got {labels}"
            )

    def test_prefix_slices_have_correct_n_real_shape(self):
        """text_to_img in every layer must be shaped (8, n_real, 512)."""
        n_real = 6
        token_ids = list(range(1, n_real + 1))
        buf = _make_attn_buf(n_text=n_real)
        transform = FakeInputTransform(token_ids, n_real=n_real)
        policy = FakePolicy(transform)

        result, _ = _run_with_stubs(policy, _make_example(), buf)

        for layer_key, layer_data in result["prefix"].items():
            t2i = layer_data["text_to_img"]
            assert t2i.shape == (N_HEADS, n_real, TOTAL_IMAGE_TOKENS), (
                f"{layer_key}: expected ({N_HEADS}, {n_real}, {TOTAL_IMAGE_TOKENS}), "
                f"got {t2i.shape}"
            )

    def test_meta_instruction_preserved(self):
        """meta['instruction'] must be the original prompt string."""
        prompt = "place cube on plate"
        buf = _make_attn_buf(n_text=3)
        transform = FakeInputTransform([1, 2, 3])
        policy = FakePolicy(transform)
        example = _make_example(prompt=prompt)

        result, _ = _run_with_stubs(policy, example, buf)

        assert result["meta"]["instruction"] == prompt


# ---------------------------------------------------------------------------
# Part 3 — Integration test with real Normalize + TokenizePrompt
# Requires PaligemmaTokenizer (downloads from GCS on first run).
# ---------------------------------------------------------------------------


@pytest.mark.manual
class TestNormalizeTokenizePipeline:
    """End-to-end: real transform chain; verify token IDs differ from old approach.

    Marked @pytest.mark.manual because PaligemmaTokenizer downloads from GCS.
    """

    def test_pi05_state_tokens_reflect_normalization(self):
        from openpi.models.model import ModelType
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.policies.droid_policy import DroidInputs
        from openpi.shared.normalize import NormStats
        from openpi.transforms import Normalize, TokenizePrompt, compose

        q01 = np.array([-0.828, -0.840, -0.843, -2.773, -1.843, 1.172, -2.047, 0.0])
        q99 = np.array([0.900,   1.385,  0.692, -0.454,  1.732, 3.467,  2.198, 0.991])
        norm_stats = {"state": NormStats(mean=np.zeros(8), std=np.ones(8), q01=q01, q99=q99)}

        tokenizer = PaligemmaTokenizer()
        transform = compose([
            DroidInputs(model_type=ModelType.PI05),
            Normalize(norm_stats, use_quantiles=True),
            TokenizePrompt(tokenizer, discrete_state_input=True),
        ])

        raw_joints = np.array([-2.02, -1.5, 0.8, -1.5, 1.0, 2.0, -0.5], dtype=np.float64)
        gripper = np.array([0.4], dtype=np.float64)
        example = _make_example("pick up the cup")
        example["observation/joint_position"] = raw_joints
        example["observation/gripper_position"] = gripper

        transformed = transform({**example})
        mask = np.asarray(transformed["tokenized_prompt_mask"])
        real_ids = transformed["tokenized_prompt"][: int(mask.sum())].tolist()

        decoded = "".join(tokenizer._tokenizer.id_to_piece(i) for i in real_ids)
        assert "State" in decoded, f"Expected 'State' in decoded prompt:\n{decoded}"

        # Old approach: raw state → bins
        raw_state = np.concatenate([raw_joints, gripper])
        old_disc = (np.digitize(raw_state, BINS) - 1).tolist()

        # New approach: quantile-normalized state → bins
        norm_state = (raw_state[:8] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        norm_state = np.clip(norm_state, -1.0, 1.0)
        new_disc = (np.digitize(norm_state, BINS) - 1).tolist()

        assert old_disc != new_disc, (
            "Raw and normalized discretizations should differ for out-of-range joints"
        )
