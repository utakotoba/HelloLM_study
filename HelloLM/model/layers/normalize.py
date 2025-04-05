import torch
import torch.nn as nn


class NormalizeLayer(nn.Module):
    def __init__(self, dim_embed):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(dim_embed))
        self.shift = nn.Parameter(torch.zeros(dim_embed))

    def forward(self, payload):
        normalized = torch.layer_norm(payload, payload.size()[1:], eps=self.eps)
        return self.scale * normalized + self.shift
