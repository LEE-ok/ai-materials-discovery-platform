"""
optimizer.py - Optuna 하이퍼파라미터 최적화
"""

import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

from config import CV_FOLDS, N_TRIALS_MAP, RANDOM_STATE

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Objective 정의 ────────────────────────

def _rf_objective(X_tr, y_tr):
    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int('n_estimators', 100, 500),
            max_depth        = trial.suggest_int('max_depth', 5, 30),
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10),
            max_features     = trial.suggest_float('max_features', 0.3, 1.0),
        )
        model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        return _cv_mse(model, X_tr, y_tr)
    return objective


def _gbm_objective(X_tr, y_tr):
    def objective(trial):
        params = dict(
            max_iter            = trial.suggest_int('max_iter', 100, 500),
            max_depth           = trial.suggest_int('max_depth', 3, 10),
            learning_rate       = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            min_samples_leaf    = trial.suggest_int('min_samples_leaf', 5, 50),
            l2_regularization   = trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
        )
        model = HistGradientBoostingRegressor(**params, random_state=RANDOM_STATE)
        return _cv_mse(model, X_tr, y_tr)
    return objective


def _mlp_objective(X_tr, y_tr):
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 4)
        layers   = tuple(trial.suggest_int(f'n_units_l{i}', 32, 256) for i in range(n_layers))
        params = dict(
            hidden_layer_sizes  = layers,
            activation          = trial.suggest_categorical('activation', ['relu', 'tanh']),
            alpha               = trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            learning_rate_init  = trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
            max_iter            = 300,
            early_stopping      = True,
            validation_fraction = 0.1,
        )
        model = MLPRegressor(**params, random_state=RANDOM_STATE)
        return _cv_mse(model, X_tr, y_tr)
    return objective


def _cv_mse(model, X, y) -> float:
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, val_idx in kf.split(X):
        model.fit(X[tr_idx], y[tr_idx])
        scores.append(mean_squared_error(y[val_idx], model.predict(X[val_idx])))
    return float(np.mean(scores))


# ── 공개 API ──────────────────────────────

OBJECTIVE_MAP = {
    'Random Forest':     _rf_objective,
    'Gradient Boosting': _gbm_objective,
    'Neural Network':    _mlp_objective,
}


def optimize(name: str, X_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    """지정 모델 Optuna 최적화 → best_params 반환"""
    n_trials = N_TRIALS_MAP[name]
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(OBJECTIVE_MAP[name](X_tr, y_tr), n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_all(X_train, X_train_s, y_ps_train, y_uts_train) -> dict:
    """
    RF / GBM / MLP 각각 PS·UTS 최적화
    { model_name: { 'ps': params, 'uts': params } } 반환
    """
    X_map = {
        'Random Forest':     X_train,
        'Gradient Boosting': X_train,
        'Neural Network':    X_train_s,
    }

    best_params_all = {}
    for name in OBJECTIVE_MAP:
        n_trials = N_TRIALS_MAP[name]
        print(f"\n[{name}] Proof Stress 최적화 중... (trials={n_trials})")
        ps_params = optimize(name, X_map[name], y_ps_train)

        print(f"[{name}] UTS 최적화 중... (trials={n_trials})")
        uts_params = optimize(name, X_map[name], y_uts_train)

        best_params_all[name] = {'ps': ps_params, 'uts': uts_params}
        print(f"  ✅ PS  최적 파라미터: {ps_params}")
        print(f"  ✅ UTS 최적 파라미터: {uts_params}")

    return best_params_all