"""
data_pipeline/bnn_torch.py - Bayesian Neural Network (MC Dropout)
예측값 + 불확실성(신뢰구간) 동시 출력
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from api.config import RANDOM_STATE


class BNNModel(nn.Module):
    """MC Dropout 기반 Bayesian Neural Network"""

    def __init__(self, input_shape: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.fc(x)


class BNNTrainer:
    def __init__(self, input_shape, epochs=500, batch_size=64,
                 lr=1e-3, patience=30, dropout_rate=0.1, n_samples=100):
        torch.manual_seed(RANDOM_STATE)
        self.model        = BNNModel(input_shape, dropout_rate)
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.patience     = patience
        self.n_samples    = n_samples   # MC Dropout 샘플 수
        self.train_losses = []
        self.val_losses   = []

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        X_tr = torch.FloatTensor(X_train)
        y_tr = torch.FloatTensor(y_train).unsqueeze(1)
        X_vl = torch.FloatTensor(X_val)
        y_vl = torch.FloatTensor(y_val).unsqueeze(1)

        loader    = DataLoader(TensorDataset(X_tr, y_tr),
                               batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.L1Loss()
        best_val, best_state, patience_cnt = float('inf'), None, 0

        for epoch in range(self.epochs):
            # 학습 시 Dropout 활성화
            self.model.train()
            train_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(X_train)

            # 검증 시 Dropout 비활성화
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

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple:
        """
        MC Dropout으로 여러 번 예측 → 평균(예측값) + 표준편차(불확실성) 반환
        Returns:
            mean: 예측 평균값
            std:  예측 표준편차 (불확실성)
            lower: 95% 신뢰구간 하한
            upper: 95% 신뢰구간 상한
        """
        X_tensor = torch.FloatTensor(X)

        # MC Dropout: 학습 모드로 전환해서 Dropout 활성화
        self.model.train()
        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(X_tensor).numpy().ravel()
                preds.append(pred)

        preds = np.array(preds)  # (n_samples, n_data)
        mean  = preds.mean(axis=0)
        std   = preds.std(axis=0)
        lower = mean - 1.96 * std  # 95% 신뢰구간 하한
        upper = mean + 1.96 * std  # 95% 신뢰구간 상한

        return mean, std, lower, upper

    def predict(self, X: np.ndarray) -> np.ndarray:
        """일반 예측 (평균값만 반환)"""
        mean, _, _, _ = self.predict_with_uncertainty(X)
        return mean