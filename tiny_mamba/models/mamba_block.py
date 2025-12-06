import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssm import SelectiveSSM

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, kernel_size=4):
        super().__init__()

        self.d_inner = d_model * expand
        
        self.in_proj = nn.Linear(d_model, self.d_inner)
        self.in_proj_gate = nn.Linear(d_model, self.d_inner)
        self.conv1d = nn.Conv1d(
            self.d_inner, 
            self.d_inner, 
            kernel_size, 
            padding=kernel_size-1, 
            groups=self.d_inner
            )
        self.ssm = SelectiveSSM(self.d_inner, d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        seq_len = x.shape[1]                        # Needed for trimming later
        
        xz = self.in_proj(x)
        x_gate = self.in_proj_gate(x)
        xz = xz.transpose(1, 2)                     # Conv expects (batch, d_inner, seq_len)
        x_conv = self.conv1d(xz)
        x_conv = x_conv[:, :, :seq_len]             # Trim the padding conv added
        x_conv = x_conv.transpose(1, 2)
        activated_conv_output = F.silu(x_conv)
        ssm_out = self.ssm(activated_conv_output)
        activated_gate = F.silu(x_gate)
        gated = ssm_out * activated_gate            # Gating 
        output = self.out_proj(gated)

        return output