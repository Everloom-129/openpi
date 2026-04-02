"""Unit tests for ckpt_compare.py pure helpers.

Run with:
    uv run pytest viz/dashboard/views/test_ckpt_compare.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from viz.dashboard.views.ckpt_compare import (
    _attn_to_heatmap,
    _build_grid_figure,
    _build_heatmap_row,
    _get_t2i,
    _get_token_labels,
    _overlay,
    NUM_IMAGE_TOKENS,
)

# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

N_HEADS = 8
TOTAL_IMAGE_TOKENS = 512  # mirrors loader.py


def _make_t2i(n_text: int, seed: int = 0) -> np.ndarray:
    """Return a random normalised text-to-image attention tensor (8, n_text, 512)."""
    rng = np.random.default_rng(seed)
    raw = rng.random((N_HEADS, n_text, TOTAL_IMAGE_TOKENS), dtype=np.float32)
    # softmax-normalise over the 512 image dimension so values sum to 1
    raw /= raw.sum(axis=-1, keepdims=True)
    return raw


def _make_data(n_text: int, token_labels: list[str] | None = None) -> dict:
    """Build a minimal inference slice-dict with `n_text` text tokens across 3 layers."""
    layers = {0: _make_t2i(n_text, 0), 5: _make_t2i(n_text, 5), 10: _make_t2i(n_text, 10)}
    prefix = {f"layer_{l}": {"text_to_img": t, "full": np.zeros((N_HEADS, 768 + n_text, 768 + n_text), dtype=np.float32)} for l, t in layers.items()}

    if token_labels is None:
        token_labels = [f"tok{i}" for i in range(n_text)]

    return {
        "meta": {
            "prefix_len": 768,
            "seq_len": 768 + n_text,
            "n_real_tokens": n_text,
            "instruction": "test instruction",
            "token_texts": token_labels,
        },
        "images": {
            "exterior": np.full((224, 224, 3), 128, dtype=np.uint8),
            "wrist": np.full((224, 224, 3), 64, dtype=np.uint8),
        },
        "prefix": prefix,
    }


def _make_camera_img() -> np.ndarray:
    return np.full((224, 224, 3), 100, dtype=np.uint8)


# ---------------------------------------------------------------------------
# _get_t2i
# ---------------------------------------------------------------------------

class TestGetT2i:
    def test_reads_from_prefix(self):
        data = _make_data(n_text=5)
        t2i = _get_t2i(data, layer=5)
        assert t2i is not None
        assert t2i.shape == (N_HEADS, 5, TOTAL_IMAGE_TOKENS)

    def test_missing_layer_returns_none(self):
        data = _make_data(n_text=5)
        assert _get_t2i(data, layer=99) is None

    def test_callable_loader(self):
        """Supports the _load_t2i callable pattern used by the Results backend."""
        t2i_store = {3: _make_t2i(7)}
        data = {"_load_t2i": lambda layer: t2i_store.get(layer)}
        assert _get_t2i(data, 3) is not None
        assert _get_t2i(data, 99) is None


# ---------------------------------------------------------------------------
# _get_token_labels
# ---------------------------------------------------------------------------

class TestGetTokenLabels:
    def test_basic(self):
        data = _make_data(n_text=4, token_labels=["▁mine", "craft", "▁is", "▁fun"])
        labels = _get_token_labels(data)
        # leading ▁ stripped
        assert labels == ["mine", "craft", "is", "fun"]

    def test_deduplication(self):
        data = _make_data(n_text=3, token_labels=["tok", "tok", "tok"])
        labels = _get_token_labels(data)
        assert labels == ["tok#1", "tok#2", "tok#3"]

    def test_empty_token_replaced_by_index(self):
        data = _make_data(n_text=2, token_labels=["▁", "word"])
        labels = _get_token_labels(data)
        assert labels[0].startswith("[")  # e.g. "[0]"
        assert labels[1] == "word"

    def test_respects_n_real_tokens(self):
        """n_real_tokens truncates the list."""
        data = _make_data(n_text=4, token_labels=["a", "b", "c", "d"])
        data["meta"]["n_real_tokens"] = 2
        labels = _get_token_labels(data)
        assert labels == ["a", "b"]


# ---------------------------------------------------------------------------
# _attn_to_heatmap
# ---------------------------------------------------------------------------

class TestAttnToHeatmap:
    def test_exterior_shape(self):
        attn_512 = np.random.rand(TOTAL_IMAGE_TOKENS).astype(np.float32)
        hmap = _attn_to_heatmap(attn_512, "exterior")
        assert hmap.shape == (112, 112)

    def test_wrist_shape(self):
        attn_512 = np.random.rand(TOTAL_IMAGE_TOKENS).astype(np.float32)
        hmap = _attn_to_heatmap(attn_512, "wrist")
        assert hmap.shape == (112, 112)

    def test_exterior_uses_first_256(self):
        attn_512 = np.zeros(TOTAL_IMAGE_TOKENS, dtype=np.float32)
        attn_512[:NUM_IMAGE_TOKENS] = 1.0      # exterior patch region
        hmap = _attn_to_heatmap(attn_512, "exterior")
        assert hmap.mean() > 0.5

    def test_wrist_uses_second_256(self):
        attn_512 = np.zeros(TOTAL_IMAGE_TOKENS, dtype=np.float32)
        attn_512[NUM_IMAGE_TOKENS:] = 1.0      # wrist patch region
        hmap = _attn_to_heatmap(attn_512, "wrist")
        assert hmap.mean() > 0.5


# ---------------------------------------------------------------------------
# _overlay
# ---------------------------------------------------------------------------

class TestOverlay:
    def test_output_shape_and_dtype(self):
        img = np.full((224, 224, 3), 128, dtype=np.uint8)
        hmap = np.random.rand(112, 112).astype(np.float32)
        out = _overlay(img, hmap)
        assert out.shape == (112, 112, 3)
        assert out.dtype == np.uint8

    def test_uniform_heatmap_blends_cleanly(self):
        img = np.full((224, 224, 3), 200, dtype=np.uint8)
        hmap = np.ones((112, 112), dtype=np.float32)
        out = _overlay(img, hmap)
        # All finite, no NaN
        assert np.all(np.isfinite(out.astype(np.float32)))


# ---------------------------------------------------------------------------
# _build_heatmap_row
# ---------------------------------------------------------------------------

class TestBuildHeatmapRow:
    def test_returns_224_heatmap(self):
        data = _make_data(n_text=10)
        hmap = _build_heatmap_row(data, layer=5, tok_idx=3, head_sel="mean", camera="exterior")
        assert hmap is not None
        assert hmap.shape == (224, 224)

    def test_clamps_tok_idx_in_bounds(self):
        """tok_idx beyond n_text should not raise."""
        data = _make_data(n_text=3)
        hmap = _build_heatmap_row(data, layer=5, tok_idx=99, head_sel="mean", camera="exterior")
        assert hmap is not None

    def test_missing_layer_returns_none(self):
        data = _make_data(n_text=5)
        hmap = _build_heatmap_row(data, layer=99, tok_idx=0, head_sel="mean", camera="exterior")
        assert hmap is None


# ---------------------------------------------------------------------------
# _build_grid_figure  — the site of the original crash
# ---------------------------------------------------------------------------

class TestBuildGridFigure:
    """Cross-model scenario: A is pi0.5 (~30 tokens), B is pi0 (~3 tokens)."""

    def _t2i_dict(self, n_text: int, layers=(0, 5, 10)) -> dict:
        return {l: _make_t2i(n_text, seed=l) for l in layers}

    def test_tok_idx_in_bounds(self):
        """Normal case: tok_idx within both models' n_text."""
        t2i = self._t2i_dict(n_text=10)
        cam = _make_camera_img()
        fig = _build_grid_figure(t2i, tok_idx=2, camera_img=cam,
                                  camera="exterior", layers=[0, 5, 10],
                                  agg_fn=lambda x: x.mean(axis=0))
        assert fig is not None

    def test_tok_idx_equals_n_text_crashes_before_fix(self):
        """tok_idx == n_text is the exact scenario that crashed (index out of bounds)."""
        t2i = self._t2i_dict(n_text=3)   # pi0 with short prompt
        cam = _make_camera_img()
        # Before fix: tok_idx=3 with n_text=3 → IndexError on axis 1 with size 3
        fig = _build_grid_figure(t2i, tok_idx=3, camera_img=cam,
                                  camera="exterior", layers=[0, 5, 10],
                                  agg_fn=lambda x: x.mean(axis=0))
        assert fig is not None

    def test_tok_idx_far_out_of_bounds(self):
        """tok_idx from pi0.5's long token list applied to pi0's 3-token t2i."""
        t2i = self._t2i_dict(n_text=3)
        cam = _make_camera_img()
        fig = _build_grid_figure(t2i, tok_idx=25, camera_img=cam,
                                  camera="exterior", layers=[0, 5, 10],
                                  agg_fn=lambda x: x.max(axis=0))
        assert fig is not None

    def test_none_layer_renders_na_placeholder(self):
        """Layers not in the dict should show 'N/A' without crashing."""
        t2i = self._t2i_dict(n_text=5)
        cam = _make_camera_img()
        fig = _build_grid_figure(t2i, tok_idx=1, camera_img=cam,
                                  camera="wrist", layers=[0, 5, 10, 99],
                                  agg_fn=lambda x: x.mean(axis=0))
        assert fig is not None

    def test_different_agg_functions(self):
        t2i = self._t2i_dict(n_text=8)
        cam = _make_camera_img()
        for agg in [
            lambda x: x.mean(axis=0),
            lambda x: x.max(axis=0),
            lambda x: np.median(x, axis=0),
            lambda x: x.std(axis=0),
        ]:
            fig = _build_grid_figure(t2i, tok_idx=2, camera_img=cam,
                                      camera="exterior", layers=[0, 5],
                                      agg_fn=agg)
            assert fig is not None

    def test_single_layer(self):
        t2i = {5: _make_t2i(n_text=6)}
        cam = _make_camera_img()
        fig = _build_grid_figure(t2i, tok_idx=0, camera_img=cam,
                                  camera="exterior", layers=[5],
                                  agg_fn=lambda x: x.mean(axis=0))
        assert fig is not None


# ---------------------------------------------------------------------------
# Cross-model: simulate the full pi0.5 A vs pi0 B scenario
# ---------------------------------------------------------------------------

class TestCrossModelScenario:
    """Reproduce the exact failure path: A has 30 tokens, B has 3 tokens,
    token selector defaults to index 3 (min(3, 30-1))."""

    def test_default_tok_idx_does_not_crash_short_model(self):
        n_a, n_b = 30, 3
        layers = [0, 5, 10]
        t2i_a = {l: _make_t2i(n_a, l) for l in layers}
        t2i_b = {l: _make_t2i(n_b, l) for l in layers}
        cam = _make_camera_img()

        # Simulate what the dashboard does: default_tok = token_labels[min(3, len-1)]
        labels_a = [f"tok{i}" for i in range(n_a)]
        tok_idx = min(3, len(labels_a) - 1)  # → 3

        agg = lambda x: x.mean(axis=0)  # noqa: E731

        fig_a = _build_grid_figure(t2i_a, tok_idx, cam, "exterior", layers, agg)
        fig_b = _build_grid_figure(t2i_b, tok_idx, cam, "exterior", layers, agg)

        assert fig_a is not None
        assert fig_b is not None  # would have raised IndexError before fix
