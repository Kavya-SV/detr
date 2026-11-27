"""
chokam.py — Token importance scorer for RT-DETR encoder features.

ChokamTokenScorer assigns a per-token importance score (0-1) so that
downstream components can selectively attend to the most relevant tokens,
or refine low-score tokens.
"""

import torch
import torch.nn as nn


class ChokamTokenScorer(nn.Module):
    """
    Chokam Stage-1: Computes an importance score (0-1) for every encoder token.

    Args:
        hidden_dim (int): Feature dimension of encoder tokens. Default: 256.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Encoder tokens of shape (B, N, hidden_dim) or (B, C, H, W).

        Returns:
            scores (torch.Tensor): Per-token importance scores of shape (B, N, 1) or (B, 1, H, W).
        """
        if features.dim() == 4:
            # (B, C, H, W) -> (B, H, W, C)
            features_perm = features.permute(0, 2, 3, 1)
            scores = self.score_net(features_perm) # (B, H, W, 1)
            # Explicit bias offset immune to weight resets
            scores = torch.sigmoid(scores + 2.0)
            return scores.permute(0, 3, 1, 2) # (B, 1, H, W)
            
        scores = self.score_net(features)
        scores = torch.sigmoid(scores + 2.0)
        return scores


class SCEWeatherDetector(nn.Module):
    """
    Lightweight Global Average Pooling based weather detector.
    Takes a list of feature maps (P3, P4, P5), pools them, concatenates, and outputs weather logits.
    """
    def __init__(self, in_channels: int = 256, num_weather_classes: int = 4, num_scales: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * num_scales, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_weather_classes)
        )
        
        # Maps weather context to pruning thresholds (tau_low, tau_high)
        self.tau_net = nn.Sequential(
            nn.Linear(num_weather_classes, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
        )

    def forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: list of (B, C, H, W)
        pooled_feats = []
        for feat in x:
            pooled = self.pool(feat).flatten(1) # (B, C)
            pooled_feats.append(pooled)
        
        x_concat = torch.cat(pooled_feats, dim=1)  # (B, C * num_scales)
        logits = self.fc(x_concat)                 # (B, num_weather_classes)
        
        # Compute thresholds
        taus = self.tau_net(logits) # (B, 2)
        # Explicit bias offsets immune to weight resets: low ~0.3, high ~0.7
        # Optimized thresholds: tau_low ~0.20, tau_high ~0.55
        bias_offsets = torch.tensor([[-1.38, 0.20]], device=taus.device, dtype=taus.dtype)
        taus = torch.sigmoid(taus + bias_offsets)
        
        # Ensure tau_low < tau_high
        tau_low, _ = torch.min(taus, dim=1, keepdim=True)  # (B, 1)
        tau_high, _ = torch.max(taus, dim=1, keepdim=True) # (B, 1)
        
        return logits, tau_low, tau_high


class ChokamRefiner(nn.Module):
    """
    1D MLP to refine low-score features.
    Operates on a condensed 1D tensor (N_medium, C) for TRUE FLOP reduction.
    """
    def __init__(self, channels: int = 256):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C)
        # Residual connection ensures features are enhanced without corrupting original representation
        return x + self.refine(x)
