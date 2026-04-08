"""Tests for the attention logit masking feature in gemma.py."""

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.gemma import _mask_attn_percentile


class TestMaskAttnPercentile:
    """Unit tests for the _mask_attn_percentile helper."""

    BIG_NEG = -2.3819763e38

    def _make_logits(self, b=1, k=1, g=8, t=10, s=10, seed=0):
        """Create random logits with some positions already masked (big_neg)."""
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal((b, k, g, t, s)).astype(np.float32)
        # Mask some positions (simulate causal mask — upper triangle)
        for ti in range(t):
            logits[:, :, :, ti, ti + 1 :] = self.BIG_NEG
        return jnp.array(logits)

    def test_mode_0_noop(self):
        """Mode 0 should return logits unchanged."""
        logits = self._make_logits()
        result = _mask_attn_percentile(logits, mode=0, pct=10.0, big_neg=self.BIG_NEG)
        np.testing.assert_array_equal(np.array(result), np.array(logits))

    def test_mode_1_masks_top_10_percent(self):
        """Mode 1 should set the top 10% of valid logits to big_neg."""
        logits = self._make_logits(b=1, k=1, g=1, t=20, s=20, seed=42)
        result = _mask_attn_percentile(logits, mode=1, pct=10.0, big_neg=self.BIG_NEG)

        logits_np = np.array(logits)
        result_np = np.array(result)

        # Count valid logits (not already big_neg)
        valid = logits_np > (self.BIG_NEG + 1.0)
        n_valid = valid.sum()

        # Count newly masked positions
        was_valid = valid
        now_masked = result_np <= (self.BIG_NEG + 1.0)
        newly_masked = was_valid & now_masked
        n_newly_masked = newly_masked.sum()

        # Should be approximately 10% of valid logits
        expected = int(n_valid * 0.10)
        assert abs(n_newly_masked - expected) <= 2, (
            f"Expected ~{expected} masked, got {n_newly_masked}"
        )

        # All newly masked should have been high-valued
        if n_newly_masked > 0:
            # The masked values should be >= the 90th percentile
            valid_vals = logits_np[valid]
            thresh = np.percentile(valid_vals, 90.0)
            masked_vals = logits_np[newly_masked]
            assert np.all(masked_vals >= thresh - 1e-5)

    def test_mode_2_masks_bottom_10_percent(self):
        """Mode 2 should set the bottom 10% of valid logits to big_neg."""
        logits = self._make_logits(b=1, k=1, g=1, t=20, s=20, seed=42)
        result = _mask_attn_percentile(logits, mode=2, pct=10.0, big_neg=self.BIG_NEG)

        logits_np = np.array(logits)
        result_np = np.array(result)

        valid = logits_np > (self.BIG_NEG + 1.0)
        n_valid = valid.sum()

        now_masked = result_np <= (self.BIG_NEG + 1.0)
        newly_masked = valid & now_masked
        n_newly_masked = newly_masked.sum()

        expected = int(n_valid * 0.10)
        assert abs(n_newly_masked - expected) <= 2

        if n_newly_masked > 0:
            valid_vals = logits_np[valid]
            thresh = np.percentile(valid_vals, 10.0)
            masked_vals = logits_np[newly_masked]
            assert np.all(masked_vals <= thresh + 1e-5)

    def test_does_not_re_mask_already_masked(self):
        """Already masked positions (big_neg) should not be counted or changed."""
        logits = self._make_logits(b=1, k=1, g=1, t=10, s=10)
        valid_before = np.array(logits) > (self.BIG_NEG + 1.0)

        result = _mask_attn_percentile(logits, mode=1, pct=10.0, big_neg=self.BIG_NEG)
        result_np = np.array(result)

        # Positions that were already big_neg should remain big_neg
        still_masked = result_np[~valid_before]
        assert np.all(still_masked <= self.BIG_NEG + 1.0)

    def test_batch_independent(self):
        """Each batch element should be masked independently."""
        rng = np.random.default_rng(123)
        logits = jnp.array(rng.standard_normal((2, 1, 4, 8, 8)).astype(np.float32))
        result = _mask_attn_percentile(logits, mode=1, pct=20.0, big_neg=self.BIG_NEG)

        # Process each batch element separately
        for b in range(2):
            single = logits[b : b + 1]
            single_result = _mask_attn_percentile(single, mode=1, pct=20.0, big_neg=self.BIG_NEG)
            np.testing.assert_allclose(
                np.array(result[b]),
                np.array(single_result[0]),
                rtol=1e-5,
            )

    def test_custom_percentile(self):
        """Different percentile values should mask different amounts."""
        logits = self._make_logits(b=1, k=1, g=1, t=30, s=30, seed=99)
        valid = np.array(logits) > (self.BIG_NEG + 1.0)
        n_valid = valid.sum()

        for pct in [5.0, 20.0, 50.0]:
            result = _mask_attn_percentile(logits, mode=1, pct=pct, big_neg=self.BIG_NEG)
            result_np = np.array(result)
            newly_masked = valid & (result_np <= self.BIG_NEG + 1.0)
            expected = int(n_valid * pct / 100.0)
            assert abs(newly_masked.sum() - expected) <= 3, (
                f"pct={pct}: expected ~{expected}, got {newly_masked.sum()}"
            )

    def test_mode_3_min_filter_keeps_top_10_percent(self):
        """Mode 3 should mask everything below the top 10%, keeping only the highest logits."""
        logits = self._make_logits(b=1, k=1, g=1, t=20, s=20, seed=42)
        result = _mask_attn_percentile(logits, mode=3, pct=10.0, big_neg=self.BIG_NEG)

        logits_np = np.array(logits)
        result_np = np.array(result)

        valid = logits_np > (self.BIG_NEG + 1.0)
        n_valid = valid.sum()

        # Count surviving (not masked) positions
        still_valid = result_np > (self.BIG_NEG + 1.0)
        n_surviving = still_valid.sum()

        # Should keep approximately 10% of valid logits
        expected_surviving = int(n_valid * 0.10)
        assert abs(n_surviving - expected_surviving) <= 2, (
            f"Expected ~{expected_surviving} surviving, got {n_surviving}"
        )

        # Surviving values should be the highest ones
        if n_surviving > 0:
            valid_vals = logits_np[valid]
            thresh = np.percentile(valid_vals, 90.0)
            surviving_vals = logits_np[still_valid]
            assert np.all(surviving_vals >= thresh - 1e-5)

    def test_mode_3_inverse_of_mode_1(self):
        """Mode 3 (keep top N%) and mode 1 (mask top N%) should be complementary."""
        logits = self._make_logits(b=1, k=1, g=1, t=20, s=20, seed=77)
        mode1 = _mask_attn_percentile(logits, mode=1, pct=10.0, big_neg=self.BIG_NEG)
        mode3 = _mask_attn_percentile(logits, mode=3, pct=10.0, big_neg=self.BIG_NEG)

        logits_np = np.array(logits)
        valid = logits_np > (self.BIG_NEG + 1.0)

        # What mode 1 masks should be what mode 3 keeps (and vice versa)
        mode1_masked = (np.array(mode1) <= self.BIG_NEG + 1.0) & valid
        mode3_kept = np.array(mode3) > (self.BIG_NEG + 1.0)
        np.testing.assert_array_equal(mode1_masked, mode3_kept)

    def test_jit_compatible(self):
        """The function should work under jax.jit."""
        logits = self._make_logits()

        @jax.jit
        def masked(logits):
            return _mask_attn_percentile(logits, mode=1, pct=10.0, big_neg=self.BIG_NEG)

        @jax.jit
        def masked_mode3(logits):
            return _mask_attn_percentile(logits, mode=3, pct=10.0, big_neg=self.BIG_NEG)

        result = masked(logits)
        assert result.shape == logits.shape
        result3 = masked_mode3(logits)
        assert result3.shape == logits.shape


class TestMaskAttnConfig:
    """Test that the masking config integrates with Pi0Config."""

    def test_default_config_no_masking(self):
        from openpi.models.pi0_config import Pi0Config

        config = Pi0Config()
        assert config.attn_logit_mask_layers is None
        assert config.attn_logit_mask_percentile == 10.0

    def test_config_with_masking(self):
        from openpi.models.pi0_config import Pi0Config

        config = Pi0Config(attn_logit_mask_layers={7: 1}, attn_logit_mask_percentile=15.0)
        assert config.attn_logit_mask_layers == {7: 1}
        assert config.attn_logit_mask_percentile == 15.0

    def test_config_multiple_layers(self):
        from openpi.models.pi0_config import Pi0Config

        config = Pi0Config(attn_logit_mask_layers={7: 1, 10: 2, 4: 1})
        assert len(config.attn_logit_mask_layers) == 3
