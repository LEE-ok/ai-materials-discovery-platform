"""
data_pipeline/mlp_torch.py - PyTorch MLP 모델
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from api.config import RANDOM_STATE


class MLPModel(nn.Module):
    def __init__(self, input_shape: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)


class TorchTrainer:
    def __init__(self, input_shape, epochs=500, batch_size=64, lr=1e-3, patience=30):
        torch.manual_seed(RANDOM_STATE)
        self.model        = MLPModel(input_shape)
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.patience     = patience
        self.train_losses = []
        self.val_losses   = []

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        X_tr = torch.FloatTensor(X_train)
        y_tr = torch.FloatTensor(y_train).unsqueeze(1)
        X_vl = torch.FloatTensor(X_val)
        y_vl = torch.FloatTensor(y_val).unsqueeze(1)

        loader    = DataLoader(TensorDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.L1Loss()
        best_val, best_state, patience_cnt = float('inf'), None, 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(X_train)

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_vl), y_vl).item()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val:
                best_val     = val_loss
                best_state   = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.FloatTensor(X)).numpy().ravel()
