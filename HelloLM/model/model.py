import torch
import torch.nn as nn
from HelloLM.config import ModelConfig
from HelloLM.model.architecture.transformer import Transformer
from HelloLM.model.layers.normalize import NormalizeLayer
from torch.utils.checkpoint import checkpoint_sequential


class HelloModel(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            model_config["vocab_size"], model_config["dim_embed"]
        )
        self.positional_embedding = nn.Embedding(
            model_config["context_length"], model_config["dim_embed"]
        )
        self.dropout = nn.Dropout(model_config["dropout_rate"])

        self.transformers = nn.Sequential(
            *[
                Transformer(
                    dim_embed=model_config["dim_embed"],
                    context_length=model_config["context_length"],
                    heads_count=model_config["heads_count"],
                    dropout_rate=model_config["dropout_rate"],
                    qkv_bias=model_config["qkv_bias"],
                )
                for _ in range(model_config["layers_count"])
            ]
        )

        self.normalize = NormalizeLayer(model_config["dim_embed"])
        self.output = nn.Linear(
            model_config["dim_embed"], model_config["vocab_size"], bias=False
        )

    def forward(self, payload: torch.Tensor):
        _, sequence_length = payload.shape
        payload = self.token_embedding(payload) + self.positional_embedding(
            torch.arange(sequence_length, device=payload.device)
        )
        payload = self.dropout(payload)
        
        if self.training:
            chunk_size = 2
            num_segments = len(self.transformers) // chunk_size
            payload = checkpoint_sequential(self.transformers, num_segments, payload, use_reentrant=False)
        else:
            payload = self.transformers(payload)
            
        payload = self.normalize(payload)
        payload = self.output(payload)
        return payload

def simple_generate(model, index, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        index_cond = index[:, -context_size:]

        with torch.no_grad():
            logits = model(index_cond)

        logits = logits[:, -1, :]

        probas = torch.softmax(logits, dim=-1)

        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        index = torch.cat((index, idx_next), dim=1)

    return index
