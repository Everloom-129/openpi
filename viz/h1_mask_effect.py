"""
Hypothesis 1: Attention Causal Fidelity Test

This script tests whether attention maps have causal fidelity by masking regions
and measuring the impact on model outputs.

Hypothesis:
    If we mask (occlude) the high-attention regions identified by the Attention Map,
    the model's output Actions should deviate significantly (i.e., error increases).
    Conversely, masking low-attention regions should cause minimal change in Actions.

Experimental Design (Input Perturbation):
    1. Baseline: Record the original Action A_orig generated from the original image I.

    2. Mask Relevant: Based on the extracted Attention Map, generate a binary mask
       that blacks out or adds Gaussian noise to high-attention regions (e.g., top 10%
       of pixels), then feed to the model to get A_mask_high.

    3. Mask Irrelevant: Randomly mask an equal area of low-attention regions,
       feed to the model to get A_mask_low.

    4. Metric: Calculate the difference in Actions (e.g., MSE or Trajectory Distance).

       Fidelity Score = MSE(A_orig, A_mask_high) - MSE(A_orig, A_mask_low)

Expected Result:
    Fidelity Score should be significantly greater than 0, indicating that
    high-attention regions are causally important for the model's predictions.
"""
