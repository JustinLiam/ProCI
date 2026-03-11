import torch, torch.nn as nn
from typing import Sequence
import torch.nn.functional as F

class ProtoMemory(nn.Module):
    def __init__(self, classes: Sequence[str], D: int):
        super().__init__()
        self.classes = list(classes)
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        K = len(self.classes)
        self.register_buffer('protos', torch.zeros(K, D))
        self.register_buffer('counts', torch.zeros(K, dtype=torch.long))

    @torch.no_grad()
    def update_batch(self, Z: torch.Tensor, y: torch.Tensor, m=0.9):

        pos_counts = y.sum(dim=0)
        mask = pos_counts > 0
        if not mask.any():
            return
        Z_sum = (Z * y.unsqueeze(-1)).sum(dim=0)
        z_bar_unnormalized = Z_sum[mask] / pos_counts[mask].unsqueeze(-1)
        z_bar = F.normalize(z_bar_unnormalized, p=2, dim=-1)
        current_z_bar_K = torch.zeros_like(self.protos)
        current_z_bar_K[mask] = z_bar
        first_update_mask = (self.counts == 0) & mask
        ema_mask = (self.counts > 0) & mask

        if first_update_mask.any():
            self.protos[first_update_mask] = current_z_bar_K[first_update_mask]

        if ema_mask.any():
            updated_protos = m * self.protos[ema_mask] + (1 - m) * current_z_bar_K[ema_mask]
            self.protos[ema_mask] = F.normalize(updated_protos, p=2, dim=-1)

        self.counts[mask] += pos_counts[mask].long()


    def get_extra_state(self):
        return {'classes': self.classes}

    def set_extra_state(self, state):
        self.classes = list(state['classes'])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}