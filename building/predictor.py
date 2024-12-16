import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class BuildingDS(Dataset):
    def __init__(self, contexts: np.ndarray, actions: np.ndarray, outcomes: np.ndarray):
        self.context = torch.tensor(contexts).float()
        self.actions = torch.tensor(actions).float()
        self.outcomes = torch.tensor(outcomes).float()

    def merge(self, contexts: np.ndarray, actions: np.ndarray, outcomes: np.ndarray, keep_pct: float):
        """
        Merges new data into the dataset, keeping a percentage of the old data.
        """
        n = len(self.context)
        keep_idxs = np.random.choice(n, int(n * keep_pct), replace=False)

        self.context = torch.concatenate([self.context[keep_idxs], torch.tensor(contexts).float()], dim=0)
        self.actions = torch.concatenate([self.actions[keep_idxs], torch.tensor(actions).float()], dim=0)
        self.outcomes = torch.concatenate([self.outcomes[keep_idxs], torch.tensor(outcomes).float()], dim=0)

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        return self.context[idx], self.actions[idx], self.outcomes[idx]


class BuildingPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(16, 64),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


def train_model(ds: Dataset, epochs: int=100, batch_size: int=16, device: str="cpu") -> BuildingPredictor:
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = BuildingPredictor()
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.MSELoss()

    with tqdm(range(epochs), leave=False, desc=f"Training on {len(ds)} samples") as pbar:
        for _ in pbar:
            loss_val = 0
            for context, action, outcome in dl:
                context, action, outcome = context.to(device), action.to(device), outcome.to(device)
                optimizer.zero_grad()
                pred = model(torch.concatenate([context, action], dim=1))
                loss = loss_fn(pred, outcome.unsqueeze(1))
                loss_val += loss.item() * len(context)
                loss.backward()
                optimizer.step()
            pbar.set_postfix({"loss": loss_val / len(ds)})

    model.eval()

    return model