from tiny_mamba.models.mamba import Mamba
import torch

model = Mamba(vocab_size=1000, d_model=256, d_state=16, n_layers=6)
input_ids = torch.randint(0, 1000, (2, 128))
print(f"\nInput shape: {input_ids.shape}")
logits = model(input_ids)
print(f"\nLogits shape: {logits.shape}")