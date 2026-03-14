"""
mlp_torch.py - PyTorch 기반 MLP 모델
Train loss + Val loss 커브 지원
ceramic notebook의 Model/Trainer 구조 참고
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import FONT_FAMILY, OUTPUT_DIR, RANDOM_STATE


# ── 모델 구조 ─────────────────────────────

class MLPModel(nn.Module):
    """
    ceramic의 Model과 동일한 구조
    Linear(입력 → 128) → ReLU → Linear(128 → 64) → ReLU → Linear(64 → 1)
    """
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


# ── PyTorch Trainer ───────────────────────

class TorchTrainer:
    """
    ceramic의 Trainer와 동일한 인터페이스
    fit() → plot_history()
    Train loss + Val loss 기록
    """

    def __init__(self,
                 input_shape: int,
                 epochs:      int   = 500,
                 batch_size:  int   = 64,
                 lr:          float = 1e-3,
                 patience:    int   = 30):
        torch.manual_seed(RANDOM_STATE)
        self.model      = MLPModel(input_shape)
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.patience   = patience
        self.train_losses = []
        self.val_losses   = []

    def fit(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_val:   np.ndarray, y_val:   np.ndarray) -> None:
        """
        ceramic trainer.fit()과 동일한 역할
        early stopping + best model 저장
        """
        # numpy → tensor
        X_tr = torch.FloatTensor(X_train)
        y_tr = torch.FloatTensor(y_train).unsqueeze(1)
        X_vl = torch.FloatTensor(X_val)
        y_vl = torch.FloatTensor(y_val).unsqueeze(1)

        loader = DataLoader(TensorDataset(X_tr, y_tr),
                            batch_size=self.batch_size, shuffle=True)

        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion  = nn.L1Loss()  # MAE (ceramic과 동일)
        best_val   = float('inf')
        best_state = None
        patience_cnt = 0

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(X_train)

            # Val
            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_vl), y_vl).item()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        # Best model 복원
        if best_state:
            self.model.load_state_dict(best_state)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.FloatTensor(X)).numpy().ravel()

    def plot_history(self, target_name: str = '', save_path: str = None) -> None:
        """
        ceramic trainer.plot_history()와 동일한 스타일
        Train loss + Val loss 같이 출력
        """
        import matplotlib.pyplot as plt
        plt.rcParams['font.family']        = FONT_FAMILY
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.train_losses, color='#1f77b4', linewidth=1.0,
                alpha=0.7, label='Train loss')
        ax.plot(self.val_losses,   color='#ff7f0e', linewidth=2.0,
                label='Val loss')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss (MAE)', fontsize=11)
        ax.set_title(f'Loss  {target_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
