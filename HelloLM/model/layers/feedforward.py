import torch.nn as nn
from HelloLM.model.layers.activation import GELU


class FeedForward(nn.Module):
    def __init__(self, dim_embed: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_embed, 4 * dim_embed),
            GELU(),
            nn.Linear(4 * dim_embed, dim_embed),
        )

    def forward(self, payload):
        return self.layers(payload)
