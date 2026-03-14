"""
trainer.py - 최적 파라미터로 모델 학습 및 평가
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

from config import RANDOM_STATE


# ── 모델 생성 ─────────────────────────────

def build_model(name: str, params: dict):
    """파라미터 딕셔너리 → 모델 객체"""
    p = dict(params)

    if name == 'Random Forest':
        return RandomForestRegressor(**p, random_state=RANDOM_STATE, n_jobs=-1)

    elif name == 'Gradient Boosting':
        return HistGradientBoostingRegressor(**p, random_state=RANDOM_STATE)

    else:  # Neural Network: n_layers / n_units_lX → hidden_layer_sizes 재구성
        n_layers = p.pop('n_layers')
        layers   = tuple(p.pop(f'n_units_l{i}') for i in range(n_layers))
        return MLPRegressor(
            hidden_layer_sizes=layers,
            max_iter=300,
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            **p,
        )


# ── 학습 + 평가 ───────────────────────────

def _eval(y_true, y_pred) -> dict:
    return dict(
        mae  = mean_absolute_error(y_true, y_pred),
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2   = r2_score(y_true, y_pred),
    )


def train_and_evaluate(
    best_params_all: dict,
    X_train, X_test, X_train_s, X_test_s,
    y_ps_train, y_ps_test, y_uts_train, y_uts_test,
) -> tuple[dict, dict]:
    """
    Returns
    -------
    results       : { model_name: metrics + predictions }
    trained_models: { model_name: { 'ps': model, 'uts': model } }
    """
    results        = {}
    trained_models = {}

    print("\n=== 최적 모델로 최종 학습 & 평가 ===")
    for name, param_dict in best_params_all.items():
        use_scaled = (name == 'Neural Network')
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s  if use_scaled else X_test

        m_ps  = build_model(name, param_dict['ps'])
        m_uts = build_model(name, param_dict['uts'])

        m_ps.fit(Xtr,  y_ps_train)
        m_uts.fit(Xtr, y_uts_train)

        pred_ps  = m_ps.predict(Xte)
        pred_uts = m_uts.predict(Xte)

        ps_metrics  = _eval(y_ps_test,  pred_ps)
        uts_metrics = _eval(y_uts_test, pred_uts)

        results[name] = {
            'ps_mae':   ps_metrics['mae'],  'ps_rmse':   ps_metrics['rmse'],  'ps_r2':   ps_metrics['r2'],
            'uts_mae':  uts_metrics['mae'], 'uts_rmse':  uts_metrics['rmse'], 'uts_r2':  uts_metrics['r2'],
            'pred_ps':  pred_ps,
            'pred_uts': pred_uts,
        }
        trained_models[name] = {'ps': m_ps, 'uts': m_uts}

        print(f"\n[{name}]")
        print(f"  Proof Stress → MAE: {ps_metrics['mae']:.2f} MPa, "
              f"RMSE: {ps_metrics['rmse']:.2f} MPa, R²: {ps_metrics['r2']:.4f}")
        print(f"  UTS          → MAE: {uts_metrics['mae']:.2f} MPa, "
              f"RMSE: {uts_metrics['rmse']:.2f} MPa, R²: {uts_metrics['r2']:.4f}")

    return results, trained_models
