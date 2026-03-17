"""
models/ai_model.py - 모델 학습 및 예측
친구 구조의 ai_model.py와 동일한 역할
"""

import joblib
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from api.config import (
    ALL_FEATURES, COLORS, FEATURE_DISPLAY_NAMES,
    FIXED_PARAMS, OUTPUT_DIR, RANDOM_STATE,
    TARGET_PS, TARGET_UTS, TORCH_PARAMS, BNN_PARAMS,
)
from data_pipeline.mlp_torch import TorchTrainer
from data_pipeline.bnn_torch import BNNTrainer


def _build_sklearn(name: str):
    p = dict(FIXED_PARAMS[name])
    if name == 'Random Forest':
        return RandomForestRegressor(**p, random_state=RANDOM_STATE, n_jobs=-1)
    return HistGradientBoostingRegressor(**p, random_state=RANDOM_STATE)


def _eval(y_true, y_pred) -> dict:
    return dict(
        mae  = mean_absolute_error(y_true, y_pred),
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2   = r2_score(y_true, y_pred),
    )


class Trainer:
    """RF + GBM + MLP + BNN 학습"""

    def __init__(self, preprocessor):
        self.pp             = preprocessor
        self.dm             = preprocessor.dm
        self.trained_models = {}
        self.torch_trainers = {}
        self.results        = {}

    def fit(self, progress_callback=None) -> None:
        pp = self.pp
        dm = self.dm
        total = 4

        # ── RF / GBM (스케일링 없이 원본 데이터 사용) ──
        for i, name in enumerate(FIXED_PARAMS):
            if progress_callback:
                progress_callback(i, total, f"[{name}] 학습 중...")

            m_ps  = _build_sklearn(name)
            m_uts = _build_sklearn(name)
            m_ps.fit(dm.X_train,  dm.y_ps_train)
            m_uts.fit(dm.X_train, dm.y_uts_train)

            pred_ps  = m_ps.predict(dm.X_test)
            pred_uts = m_uts.predict(dm.X_test)
            ps_m     = _eval(dm.y_ps_test,  pred_ps)
            uts_m    = _eval(dm.y_uts_test, pred_uts)

            self.results[name] = {
                **{f'ps_{k}':  v for k, v in ps_m.items()},
                **{f'uts_{k}': v for k, v in uts_m.items()},
                'pred_ps': pred_ps, 'pred_uts': pred_uts,
            }
            self.trained_models[name] = {'ps': m_ps, 'uts': m_uts}

        # ── Neural Network (스케일된 데이터 사용) ────
        if progress_callback:
            progress_callback(2, total, "[Neural Network] PyTorch 학습 중...")

        input_shape = pp.X_train_s.shape[1]
        tt_ps  = TorchTrainer(input_shape, **TORCH_PARAMS)
        tt_uts = TorchTrainer(input_shape, **TORCH_PARAMS)

        tt_ps.fit(pp.X_train_s,  dm.y_ps_train,  pp.X_val_s, dm.y_ps_val)
        tt_uts.fit(pp.X_train_s, dm.y_uts_train, pp.X_val_s, dm.y_uts_val)

        pred_ps  = tt_ps.predict(pp.X_test_s)
        pred_uts = tt_uts.predict(pp.X_test_s)
        ps_m     = _eval(dm.y_ps_test,  pred_ps)
        uts_m    = _eval(dm.y_uts_test, pred_uts)

        self.results['Neural Network'] = {
            **{f'ps_{k}':  v for k, v in ps_m.items()},
            **{f'uts_{k}': v for k, v in uts_m.items()},
            'pred_ps': pred_ps, 'pred_uts': pred_uts,
        }
        self.trained_models['Neural Network'] = {'ps': tt_ps, 'uts': tt_uts}
        self.torch_trainers = {'ps': tt_ps, 'uts': tt_uts}

        # ── BNN (MC Dropout) ──────────────
        if progress_callback:
            progress_callback(3, total, "[BNN] Bayesian Neural Network 학습 중...")

        bnn_ps  = BNNTrainer(input_shape, **BNN_PARAMS)
        bnn_uts = BNNTrainer(input_shape, **BNN_PARAMS)

        bnn_ps.fit(pp.X_train_s,  dm.y_ps_train,  pp.X_val_s, dm.y_ps_val)
        bnn_uts.fit(pp.X_train_s, dm.y_uts_train, pp.X_val_s, dm.y_uts_val)

        pred_ps  = bnn_ps.predict(pp.X_test_s)
        pred_uts = bnn_uts.predict(pp.X_test_s)
        ps_m     = _eval(dm.y_ps_test,  pred_ps)
        uts_m    = _eval(dm.y_uts_test, pred_uts)

        self.results['BNN'] = {
            **{f'ps_{k}':  v for k, v in ps_m.items()},
            **{f'uts_{k}': v for k, v in uts_m.items()},
            'pred_ps': pred_ps, 'pred_uts': pred_uts,
        }
        self.trained_models['BNN'] = {'ps': bnn_ps, 'uts': bnn_uts}
        self.bnn_trainers = {'ps': bnn_ps, 'uts': bnn_uts}

        if progress_callback:
            progress_callback(total, total, "✅ 학습 완료!")

    def best_model_name(self) -> str:
        return max(self.results, key=lambda x: self.results[x]['ps_r2'])

    def save_models(self) -> None:
        os.makedirs(f'{OUTPUT_DIR}/models', exist_ok=True)
        best = self.best_model_name()
        joblib.dump(self.trained_models[best]['ps'],  f'{OUTPUT_DIR}/models/best_model_ps.pkl')
        joblib.dump(self.trained_models[best]['uts'], f'{OUTPUT_DIR}/models/best_model_uts.pkl')


class Predictor:
    """예측 및 분석"""

    def __init__(self, trainer, preprocessor):
        self.trainer = trainer
        self.pp      = preprocessor
        self.dm      = preprocessor.dm

    def predict(self, inverse_scaling: bool = True) -> tuple:
        best       = self.trainer.best_model_name()
        m_ps       = self.trainer.trained_models[best]['ps']
        m_uts      = self.trainer.trained_models[best]['uts']
        use_scaled = (best == 'Neural Network')

        Xte      = self.pp.X_test_s if use_scaled else self.dm.X_test
        pred_ps  = m_ps.predict(Xte)
        pred_uts = m_uts.predict(Xte)

        # Neural Network만 역스케일링 필요
        # RF/GBM은 원본 데이터로 학습했으므로 역스케일링 불필요
        if inverse_scaling and use_scaled:
            pred_ps  = self.pp.scaler_ps.inverse_transform(pred_ps.reshape(-1,1)).ravel()
            pred_uts = self.pp.scaler_uts.inverse_transform(pred_uts.reshape(-1,1)).ravel()

        result_df = pd.DataFrame({
            '실제_PS':  self.dm.y_ps_test,
            '예측_PS':  pred_ps,
            '오차_PS':  self.dm.y_ps_test  - pred_ps,
            '실제_UTS': self.dm.y_uts_test,
            '예측_UTS': pred_uts,
            '오차_UTS': self.dm.y_uts_test - pred_uts,
        })
        result_df.to_csv(f'{OUTPUT_DIR}/predictions.csv', index=False, encoding='utf-8-sig')
        return result_df, best

    def feature_importance(self) -> tuple:
        if 'Random Forest' not in self.trainer.trained_models:
            return None, None
        rf_ps  = self.trainer.trained_models['Random Forest']['ps']
        rf_uts = self.trainer.trained_models['Random Forest']['uts']
        return rf_ps.feature_importances_, rf_uts.feature_importances_

    def predict_uncertainty(self) -> tuple:
        """BNN으로 예측값 + 95% 신뢰구간 반환"""
        if 'BNN' not in self.trainer.trained_models:
            return None, None, None, None, None, None

        bnn_ps  = self.trainer.trained_models['BNN']['ps']
        bnn_uts = self.trainer.trained_models['BNN']['uts']

        mean_ps,  std_ps,  lo_ps,  hi_ps  = bnn_ps.predict_with_uncertainty(self.pp.X_test_s)
        mean_uts, std_uts, lo_uts, hi_uts = bnn_uts.predict_with_uncertainty(self.pp.X_test_s)

        return (mean_ps, std_ps, lo_ps, hi_ps,
                mean_uts, std_uts, lo_uts, hi_uts)

    def temp_error(self) -> dict:
        best       = self.trainer.best_model_name()
        res        = self.trainer.results[best]
        test_temps = pd.DataFrame(self.dm.X_test, columns=ALL_FEATURES)['Temperature (K)'].values

        result = {}
        for target, pred, actual in [
            ('ps',  res['pred_ps'],  self.dm.y_ps_test),
            ('uts', res['pred_uts'], self.dm.y_uts_test),
        ]:
            result[target] = {
                t: mean_absolute_error(actual[test_temps==t], pred[test_temps==t])
                for t in sorted(np.unique(test_temps))
                if (test_temps==t).sum() >= 3
            }
        return result