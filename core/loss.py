import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Weighted L1 Loss (Used in Experiment 2) ---
class WeightedL1Loss(nn.Module):
    def __init__(self, weight_factor=2.0):
        super().__init__()
        self.weight_factor = weight_factor

    def forward(self, pred, target, mask):
        # Apply mask to focus only on active chip area
        mask = mask.unsqueeze(1)
        diff = torch.abs(pred - target) * mask
        
        # Apply higher weight to actual congestion spikes (where target > 0.1)
        weight = torch.where(target > 0.1, self.weight_factor, 1.0)
        weighted_diff = diff * weight
        
        return weighted_diff.sum() / (mask.sum() + 1e-8)

# --- 2. Masked Huber Loss (Used in Experiments 3 & 4) ---
class MaskedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, mask):
        mask = mask.unsqueeze(1)
        # Huber loss is robust to outliers and balances L1/L2
        loss = F.huber_loss(pred, target, reduction='none', delta=self.delta)
        masked_loss = loss * mask
        return masked_loss.sum() / (mask.sum() + 1e-8)

# --- 3. Masked Weighted BCE Loss (Used in Experiments 5 & 6) ---
class MaskedWeightedBCE(nn.Module):
    def __init__(self, h_weight=5.0, v_weight=2.0):
        super().__init__()
        self.h_weight = h_weight
        self.v_weight = v_weight

    def forward(self, logits, target, mask):
        # logits: [B, 2, H, W], target: [B, 2, H, W], mask: [B, H, W]
        mask = mask.unsqueeze(1)
        
        # Binary Cross Entropy with Pos_Weight for imbalanced hotspots
        loss_h = F.binary_cross_entropy_with_logits(
            logits[:, 0:1, :, :], target[:, 0:1, :, :], 
            reduction='none', pos_weight=torch.tensor([self.h_weight]).to(logits.device)
        )
        
        loss_v = F.binary_cross_entropy_with_logits(
            logits[:, 1:2, :, :], target[:, 1:2, :, :], 
            reduction='none', pos_weight=torch.tensor([self.v_weight]).to(logits.device)
        )
        
        total_loss = (loss_h + loss_v) * mask
        return total_loss.sum() / (mask.sum() * 2 + 1e-8)
