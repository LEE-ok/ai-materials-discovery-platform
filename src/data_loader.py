"""
data_loader.py - 데이터 로드 및 전처리
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    ALL_FEATURES, COMPOSITION_FEATURES, DATA_PATH, HEADER_ROW,
    NUMERIC_COLS, RANDOM_STATE, SHEET_NAME, TARGET_PS, TARGET_UTS,
    TEST_SIZE,
)


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """엑셀 원본 로드 + 기본 타입 정리"""
    df = pd.read_excel(path, sheet_name=SHEET_NAME, header=HEADER_ROW)
    df.replace('Na', np.nan, inplace=True)

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Type of melting'] = pd.to_numeric(df['Type of melting'], errors='coerce').fillna(0)
    return df


def preprocess(df: pd.DataFrame):
    """
    모델 입력용 X, y 생성

    Returns
    -------
    X_train, X_test, X_train_s, X_test_s,
    y_ps_train, y_ps_test, y_uts_train, y_uts_test,
    scaler, imputer, df_model
    """
    df_model = df[ALL_FEATURES + [TARGET_PS, TARGET_UTS]].copy()
    df_model['Solution_treatment_temperature'] = df_model['Solution_treatment_temperature'].fillna(
        df_model['Solution_treatment_temperature'].median()
    )
    df_model = df_model.dropna(subset=[TARGET_PS, TARGET_UTS])
    print(f"최종 모델링 데이터: {df_model.shape[0]}행 x {df_model.shape[1]}열")

    X_raw = df_model[ALL_FEATURES].values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X_raw)

    y_ps  = df_model[TARGET_PS].values
    y_uts = df_model[TARGET_UTS].values

    X_train, X_test, y_ps_train, y_ps_test, y_uts_train, y_uts_test = train_test_split(
        X, y_ps, y_uts, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return (
        X_train, X_test, X_train_s, X_test_s,
        y_ps_train, y_ps_test, y_uts_train, y_uts_test,
        scaler, imputer, df_model,
    )
