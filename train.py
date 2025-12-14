import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tiny_mamba.models.mamba import Mamba

config = {
    "vocab_size":1000,
    "d_model": 256,
    "d_state": 16,
    "n_layers": 4,
    "batch_size": 8,
    "seq_len": 64,
    "lr": 3e-4,
    "epochs": 10,
}

# Device agnostic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATASET OPS ===================
class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        """
        data: 1D tensor of token IDs
        seq_len: length of each training equence
        """
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        # Number of complete sequences we can extract
        # Need seq_len + 1 tokens for each sample 
        return (len(self.data) -1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.data[start:end]

        input_ids = chunk[:-1]
        target_ids = chunk[1:]

        return input_ids, target_ids


# Training Function
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for input_ids, targets in dataloader:

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    data = torch.randint(0, config["vocab_size"], (50000,))

    dataset = TextDataset(data, config["seq_len"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = Mamba(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        d_state=config["d_state"],
        n_layers=config["n_layers"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {loss:.4f}")


if __name__ == "__main__":
    main()