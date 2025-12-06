import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_block import MambaBlock

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        rms_norm = x / rms * self.weight

        return rms_norm

class Mamba(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, n_layers, expand=2):
        super().__init__()

        # Config parameters
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.expand = expand
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Layers
        self.layers = nn.ModuleList(
            [MambaBlock(self.d_model, self.d_state, self.expand) for i in range(n_layers)]
            )
        self.norms = nn.ModuleList(
            [RMSNorm(self.d_model) for i in range(n_layers)]
        )

        # Final normalization
        self.final_norm = RMSNorm(self.d_model)

        # LM head
        self.lm_head = nn.Linear(
            self.d_model, 
            self.vocab_size, 
            bias=False
            )
        self.lm_head.weight = self.embedding.weight         # Weight tying             

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))                          # Pre-norm for more stability
        
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits