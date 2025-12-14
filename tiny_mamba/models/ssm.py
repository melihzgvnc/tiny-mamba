import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        # A Parameter
        # delta, B, C projection layers
        super().__init__()
        
        self.d_state = d_state
        # self.A = nn.Parameter(torch.randn(d_model, d_state))
        self.A = nn.Parameter(-torch.abs(torch.randn(d_model, d_state)))

        self.delta_proj = nn.Linear(d_model, d_model)        
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)


    def discretize(self, delta, B):
        # Covert continuos to discrete
        A_bar = torch.exp(delta.unsqueeze(-1) * self.A)
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)

        return A_bar, B_bar


    def selective_scan(self, x, delta, B, C):
        # The recurrence loop
        A_bar, B_bar = self.discretize(delta, B)
        
        # Sequential fallback
        batch = x.shape[0]
        d_model = x.shape[2]
        h = torch.zeros(batch, d_model, self.d_state, device=x.device)

        outputs = []
        for t in range(x.shape[1]):
            h = A_bar[:,t] * h + B_bar[:,t] * x[:,t].unsqueeze(-1)
            y = (h * C[:,t].unsqueeze(1)).sum(-1)
            outputs.append(y)

        return torch.stack(outputs, dim=1) 


    def forward(self, x):
        # Tie it all together
        delta = self.delta_proj(x)
        delta = F.softplus(delta) # Apply softplus to delta
        B = self.B_proj(x)
        C = self.C_proj(x)

        output = self.selective_scan(x, delta, B, C)

        return output