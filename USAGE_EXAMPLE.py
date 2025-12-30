#!/usr/bin/env python3
"""
Example usage of Lambert-Beer loss with predicted parameters and spatial regularization.

This example shows how to:
1. Create a model that predicts underwater parameters (β_1, β_2, B_inf)
2. Set up the Lambert-Beer loss with spatial regularization
3. Use it in training
"""

from mast3r.losses import InfoNCE, MatchingLoss, ConfMatchingLoss
from mast3r.lambert_beer_loss import (
    LambertBeerColorLoss,
    MatchingWithLambertBeerLoss,
    SpatialSmoothnessRegularizer
)
from mast3r.model import AsymmetricMASt3R

# Example 1: Model with underwater parameter prediction
# ======================================================

# Create model that predicts β_1, β_2, B_inf in the head
model = AsymmetricMASt3R(
    predict_underwater_params=True,  # Enable prediction of underwater parameters
    desc_mode=('norm'),
    two_confs=True,
    # ... other model parameters
)

# Example 2: Loss setup with predicted parameters
# ================================================

# Base matching loss
base_criterion = InfoNCE(temperature=0.05, mode='proper')
base_matching_loss = MatchingLoss(
    base_criterion,
    negatives_padding=0,
    blocksize=8192
)

# Lambert-Beer loss using predicted parameters
lb_loss = LambertBeerColorLoss(
    similarity_threshold=0.95,  # Use matches with dot product >= 0.95
    use_predicted_params=True,  # Use parameters predicted by the head
    reduction='mean'
)

# Spatial smoothness regularizer
spatial_reg = SpatialSmoothnessRegularizer(weight=0.01)

# Combined loss
combined_matching_loss = MatchingWithLambertBeerLoss(
    base_matching_loss,
    lb_loss,
    lambert_beer_weight=0.1,  # Weight for color restoration loss
    spatial_reg=spatial_reg,  # Spatial regularization
    spatial_reg_weight=0.01   # Weight for spatial regularization
)

# Example 3: Use in training (same interface as ConfMatchingLoss)
# ================================================================

# Wrap with confidence weighting if needed
final_matching_loss = ConfMatchingLoss(
    combined_matching_loss,
    alpha=10.0,
    confmode='mean'
)

# Use in training
# loss, details = final_matching_loss(gt1, gt2, pred1, pred2)
# details will contain:
#   - 'lambert_beer_loss': color restoration loss
#   - 'spatial_reg_loss': spatial smoothness regularization
#   - 'matching_loss': base matching loss
#   - 'total_loss': combined loss

# Example 4: Fixed parameters (without prediction)
# =================================================

import torch

lb_loss_fixed = LambertBeerColorLoss(
    similarity_threshold=0.95,
    beta_1=0.1,  # Fixed β_1
    beta_2=0.12,  # Fixed β_2 (different from β_1)
    b_inf=torch.tensor([0.5, 0.5, 0.5]),  # Fixed background light
    use_predicted_params=False,  # Use fixed parameters
    reduction='mean'
)

combined_matching_loss_fixed = MatchingWithLambertBeerLoss(
    base_matching_loss,
    lb_loss_fixed,
    lambert_beer_weight=0.1
    # No spatial regularization needed for fixed parameters
)



