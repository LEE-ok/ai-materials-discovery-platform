"""
preprocessor.py - 스케일링 및 전처리
ceramic notebook의 Preprocessor 역할
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os

from config import OUTPUT_DIR, RANDOM_STATE


class Preprocessor:
    """
    ceramic notebook의 Preprocessor와 동일한 인터페이스
    X, y 각각 StandardScaler 적용
    """

    def __init__(self, data_manager):
        self.dm        = data_manager
        self.imputer   = SimpleImputer(strategy='median')
        self.scaler_x  = StandardScaler()
        self.scaler_ps = StandardScaler()
        self.scaler_uts= StandardScaler()

        # 스케일된 데이터
        self.X_train_s   = None
        self.X_val_s     = None
        self.X_test_s    = None
        self.y_ps_train_s  = None
        self.y_uts_train_s = None

    def standard_scaling(self,
                         load_x_scaler:   str = None,
                         load_ps_scaler:  str = None,
                         load_uts_scaler: str = None) -> None:
        """
        X와 y 모두 StandardScaling
        load_*_scaler 경로를 넘기면 저장된 scaler 로드 (예측 시 사용)
        """
        dm = self.dm

        # ── X 처리 ──────────────────────────
        X_train_imp = self.imputer.fit_transform(dm.X_train)
        X_val_imp   = self.imputer.transform(dm.X_val)   if dm.X_val  is not None else None
        X_test_imp  = self.imputer.transform(dm.X_test)  if dm.X_test is not None else None

        if load_x_scaler:
            self.scaler_x = joblib.load(load_x_scaler)
        else:
            self.scaler_x.fit(X_train_imp)

        self.X_train_s = self.scaler_x.transform(X_train_imp)
        self.X_val_s   = self.scaler_x.transform(X_val_imp)  if X_val_imp  is not None else None
        self.X_val_s   = self.scaler_x.transform(X_val_imp)  if X_val_imp  is not None else None
        self.X_test_s  = self.scaler_x.transform(X_test_imp) if X_test_imp is not None else None

        # ── y 처리 (PS) ──────────────────────
        if load_ps_scaler:
            self.scaler_ps = joblib.load(load_ps_scaler)
        else:
            self.scaler_ps.fit(dm.y_ps_train.reshape(-1, 1))
        self.y_ps_train_s = self.scaler_ps.transform(dm.y_ps_train.reshape(-1, 1)).ravel()

        # ── y 처리 (UTS) ─────────────────────
        if load_uts_scaler:
            self.scaler_uts = joblib.load(load_uts_scaler)
        else:
            self.scaler_uts.fit(dm.y_uts_train.reshape(-1, 1))
        self.y_uts_train_s = self.scaler_uts.transform(dm.y_uts_train.reshape(-1, 1)).ravel()

        print("스케일링 완료 (X, Proof Stress, UTS)")

    def save_scalers(self) -> None:
        """스케일러 저장"""
        os.makedirs(f'{OUTPUT_DIR}/scalers', exist_ok=True)
        joblib.dump(self.imputer,    f'{OUTPUT_DIR}/scalers/imputer.pkl')
        joblib.dump(self.scaler_x,   f'{OUTPUT_DIR}/scalers/scaler_x.pkl')
        joblib.dump(self.scaler_ps,  f'{OUTPUT_DIR}/scalers/scaler_ps.pkl')
        joblib.dump(self.scaler_uts, f'{OUTPUT_DIR}/scalers/scaler_uts.pkl')
        print("스케일러 저장 완료 → outputs/scalers/")
