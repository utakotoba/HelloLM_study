import torch
import torch.nn as nn


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, payload):
        return (
            0.5
            * payload
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2 / torch.pi))
                    * (payload + 0.044715 * torch.pow(payload, 3))
                )
            )
        )
