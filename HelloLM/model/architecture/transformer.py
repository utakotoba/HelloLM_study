import torch.nn as nn
from HelloLM.model.layers.attention import AttentionLayer
from HelloLM.model.layers.normalize import NormalizeLayer
from HelloLM.model.layers.feedforward import FeedForward


class Transformer(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        context_length: int,
        heads_count: int,
        dropout_rate: float,
        qkv_bias: bool,
    ):
        super().__init__()
        self.attention_layer = AttentionLayer(
            dim_in=dim_embed,
            dim_out=dim_embed,
            context_length=context_length,
            heads_count=heads_count,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias,
        )
        self.feedforward = FeedForward(dim_embed)

        self.normalize_layers = nn.ModuleList(
            [NormalizeLayer(dim_embed), NormalizeLayer(dim_embed)]
        )
        self.drop_shortcut = nn.Dropout(dropout_rate)

    def forward(self, payload):
        for i, layer in enumerate([self.attention_layer, self.feedforward]):
            shortcut = payload
            payload = self.normalize_layers[i](payload)
            payload = layer(payload)
            payload = self.drop_shortcut(payload)
            payload = payload + shortcut

        return payload
