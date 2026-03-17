"""
data_pipeline/parser.py - 데이터 로드 및 전처리
친구 구조의 parser.py와 동일한 역할
"""

import joblib
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from api.config import (
    ALL_FEATURES, HEADER_ROW, NUMERIC_COLS,
    OUTPUT_DIR, RANDOM_STATE, SHEET_NAME,
    TARGET_PS, TARGET_UTS, TARGETS,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
)


class DataManager:
    """데이터 로드 및 Train/Val/Test 분할"""

    def __init__(self):
        self.df          = None
        self.df_model    = None
        self.X_train = self.X_val = self.X_test = None
        self.y_ps_train  = self.y_ps_val  = self.y_ps_test  = None
        self.y_uts_train = self.y_uts_val = self.y_uts_test = None

    def load(self, path: str) -> None:
        df = pd.read_excel(path, sheet_name=SHEET_NAME, header=HEADER_ROW)
        df.replace('Na', np.nan, inplace=True)
        for col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Type of melting'] = pd.to_numeric(df['Type of melting'], errors='coerce').fillna(0)
        self.df = df

        df_model = df[ALL_FEATURES + TARGETS].copy()
        df_model['Solution_treatment_temperature'] = df_model['Solution_treatment_temperature'].fillna(
            df_model['Solution_treatment_temperature'].median()
        )
        df_model = df_model.dropna(subset=TARGETS)
        self.df_model = df_model

        X     = df_model[ALL_FEATURES].values
        y_ps  = df_model[TARGET_PS].values
        y_uts = df_model[TARGET_UTS].values

        val_test = VAL_SIZE + TEST_SIZE
        X_tr, X_tmp, yps_tr, yps_tmp, yuts_tr, yuts_tmp = train_test_split(
            X, y_ps, y_uts, test_size=val_test, random_state=RANDOM_STATE)

        test_r = TEST_SIZE / val_test
        X_val, X_test, yps_val, yps_test, yuts_val, yuts_test = train_test_split(
            X_tmp, yps_tmp, yuts_tmp, test_size=test_r, random_state=RANDOM_STATE)

        self.X_train, self.X_val, self.X_test = X_tr, X_val, X_test
        self.y_ps_train,  self.y_ps_val,  self.y_ps_test  = yps_tr,  yps_val,  yps_test
        self.y_uts_train, self.y_uts_val, self.y_uts_test = yuts_tr, yuts_val, yuts_test

    def summary(self) -> str:
        df = self.df_model
        return (
            f"전체: {len(df):,}행  |  Train: {len(self.X_train):,}  |  "
            f"Val: {len(self.X_val):,}  |  Test: {len(self.X_test):,}\n"
            f"Proof Stress  평균: {df[TARGET_PS].mean():.1f} / 최소: {df[TARGET_PS].min():.1f} / 최대: {df[TARGET_PS].max():.1f} MPa\n"
            f"UTS           평균: {df[TARGET_UTS].mean():.1f} / 최소: {df[TARGET_UTS].min():.1f} / 최대: {df[TARGET_UTS].max():.1f} MPa"
        )


class Preprocessor:
    """StandardScaling 전처리"""

    def __init__(self, data_manager):
        self.dm         = data_manager
        self.imputer    = SimpleImputer(strategy='median')
        self.scaler_x   = StandardScaler()
        self.scaler_ps  = StandardScaler()
        self.scaler_uts = StandardScaler()
        self.X_train_s = self.X_val_s = self.X_test_s = None
        self.y_ps_train_s = self.y_uts_train_s = None

    def fit_transform(self) -> None:
        dm   = self.dm
        X_tr  = self.imputer.fit_transform(dm.X_train)
        X_val = self.imputer.transform(dm.X_val)
        X_te  = self.imputer.transform(dm.X_test)

        self.scaler_x.fit(X_tr)
        self.X_train_s = self.scaler_x.transform(X_tr)
        self.X_val_s   = self.scaler_x.transform(X_val)
        self.X_test_s  = self.scaler_x.transform(X_te)

        self.scaler_ps.fit(dm.y_ps_train.reshape(-1, 1))
        self.y_ps_train_s = self.scaler_ps.transform(dm.y_ps_train.reshape(-1, 1)).ravel()

        self.scaler_uts.fit(dm.y_uts_train.reshape(-1, 1))
        self.y_uts_train_s = self.scaler_uts.transform(dm.y_uts_train.reshape(-1, 1)).ravel()

    def save(self) -> None:
        os.makedirs(f'{OUTPUT_DIR}/scalers', exist_ok=True)
        joblib.dump(self.imputer,    f'{OUTPUT_DIR}/scalers/imputer.pkl')
        joblib.dump(self.scaler_x,   f'{OUTPUT_DIR}/scalers/scaler_x.pkl')
        joblib.dump(self.scaler_ps,  f'{OUTPUT_DIR}/scalers/scaler_ps.pkl')
        joblib.dump(self.scaler_uts, f'{OUTPUT_DIR}/scalers/scaler_uts.pkl')
