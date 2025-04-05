import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        heads_count: int,
        context_length: int,
        dropout_rate: float,
        qkv_bias=False,
    ):
        super().__init__()

        assert dim_out % heads_count == 0, "dim_out must be divisible by heads_count"

        self.dim_out = dim_out
        self.head_count = heads_count
        self.dim_head = dim_out // heads_count

        # Adjustable weight layers
        self.qkv_matrix = nn.Linear(dim_in, 3 * dim_out, bias=qkv_bias)
        self.output_project = nn.Linear(dim_out, dim_out)

        # Layer Mask
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, payload: torch.Tensor):
        batch_count, tokens_count, _ = payload.shape

        # Compute query, key, and value in a single matrix multiplication
        qkv = self.qkv_matrix(payload)
        queries, keys, values = torch.chunk(qkv, 3, dim=-1)

        # Reshape and transpose for multi-head attention
        queries = queries.view(batch_count, tokens_count, self.head_count, self.dim_head).transpose(1, 2)
        keys = keys.view(batch_count, tokens_count, self.head_count, self.dim_head).transpose(1, 2)
        values = values.view(batch_count, tokens_count, self.head_count, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = (queries @ keys.transpose(-2, -1)) / (self.dim_head ** 0.5)
        attention_scores.masked_fill_(self.mask[:tokens_count, :tokens_count] == 1, float('-inf'))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch_count, tokens_count, self.dim_out)

        return self.output_project(context_vector)
