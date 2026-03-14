"""
data_loader.py - 데이터 로드 및 요약
ceramic notebook의 DataManager 역할
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    ALL_FEATURES, DATA_PATH, HEADER_ROW, NUMERIC_COLS,
    RANDOM_STATE, SHEET_NAME, TARGET_PS, TARGET_UTS, TARGETS,
    TRAIN_SIZE, VAL_SIZE, TEST_SIZE,
)


class DataManager:
    """
    데이터 로드, 분할, 요약을 담당
    ceramic notebook의 DataManager와 동일한 역할
    """

    def __init__(self):
        self.df        = None
        self.df_model  = None
        self.X_train   = None
        self.X_val     = None
        self.X_test    = None
        self.y_ps_train  = None
        self.y_ps_val    = None
        self.y_ps_test   = None
        self.y_uts_train = None
        self.y_uts_val   = None
        self.y_uts_test  = None

    def load(self, path: str = DATA_PATH) -> None:
        """엑셀 로드 + 전처리 + train/val/test 분할"""
        # 원본 로드
        df = pd.read_excel(path, sheet_name=SHEET_NAME, header=HEADER_ROW)
        df.replace('Na', np.nan, inplace=True)
        for col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Type of melting'] = pd.to_numeric(df['Type of melting'], errors='coerce').fillna(0)
        self.df = df

        # 모델용 데이터프레임
        df_model = df[ALL_FEATURES + TARGETS].copy()
        df_model['Solution_treatment_temperature'] = df_model['Solution_treatment_temperature'].fillna(
            df_model['Solution_treatment_temperature'].median()
        )
        df_model = df_model.dropna(subset=TARGETS)
        self.df_model = df_model

        # train / val+test 분할
        X = df_model[ALL_FEATURES].values
        y_ps  = df_model[TARGET_PS].values
        y_uts = df_model[TARGET_UTS].values

        val_test_size = VAL_SIZE + TEST_SIZE
        X_train, X_tmp, y_ps_train, y_ps_tmp, y_uts_train, y_uts_tmp = train_test_split(
            X, y_ps, y_uts, test_size=val_test_size, random_state=RANDOM_STATE
        )

        # val / test 분할
        test_ratio = TEST_SIZE / val_test_size
        X_val, X_test, y_ps_val, y_ps_test, y_uts_val, y_uts_test = train_test_split(
            X_tmp, y_ps_tmp, y_uts_tmp, test_size=test_ratio, random_state=RANDOM_STATE
        )

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_ps_train,  self.y_ps_val,  self.y_ps_test  = y_ps_train,  y_ps_val,  y_ps_test
        self.y_uts_train, self.y_uts_val, self.y_uts_test = y_uts_train, y_uts_val, y_uts_test

    def summary(self) -> None:
        """ceramic DataManager.summary()와 동일한 스타일로 출력"""
        df = self.df_model
        print("=" * 60)
        print("  오스테나이트계 철강 고온 강도 예측 - 데이터 요약")
        print("=" * 60)
        print(f"  전체 데이터  : {len(df):,} 행 × {len(ALL_FEATURES)} 피처")
        print(f"  학습 (Train) : {len(self.X_train):,} 행  ({TRAIN_SIZE*100:.0f}%)")
        print(f"  검증 (Val)   : {len(self.X_val):,} 행  ({VAL_SIZE*100:.0f}%)")
        print(f"  테스트(Test) : {len(self.X_test):,} 행  ({TEST_SIZE*100:.0f}%)")
        print(f"  목표변수     : {TARGET_PS}")
        print(f"               {TARGET_UTS}")
        print("-" * 60)
        print(f"  [Proof Stress] 평균: {df[TARGET_PS].mean():.1f} / "
              f"최소: {df[TARGET_PS].min():.1f} / 최대: {df[TARGET_PS].max():.1f} MPa")
        print(f"  [UTS]          평균: {df[TARGET_UTS].mean():.1f} / "
              f"최소: {df[TARGET_UTS].min():.1f} / 최대: {df[TARGET_UTS].max():.1f} MPa")
        print("=" * 60)
