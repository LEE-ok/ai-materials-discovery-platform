"""
trainer.py - 모델 학습 및 평가
RF + GBM: sklearn 고정 파라미터
Neural Network: PyTorch (Train/Val loss 커브 지원)
ceramic notebook의 Trainer 역할
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import COLORS, FONT_FAMILY, OUTPUT_DIR, RANDOM_STATE
from mlp_torch import TorchTrainer


# ── 고정 파라미터 (RF / GBM) ──────────────

FIXED_PARAMS = {
    'Random Forest': dict(
        n_estimators     = 200,
        max_depth        = 15,
        min_samples_leaf = 2,
        max_features     = 0.8,
    ),
    'Gradient Boosting': dict(
        max_iter          = 200,
        max_depth         = 5,
        learning_rate     = 0.05,
        min_samples_leaf  = 10,
        l2_regularization = 0.1,
    ),
}

# PyTorch MLP 파라미터
TORCH_PARAMS = dict(
    epochs     = 500,
    batch_size = 64,
    lr         = 1e-3,
    patience   = 30,
)

X_SCALED_MAP = {
    'Random Forest':     False,
    'Gradient Boosting': False,
    'Neural Network':    True,
}


def _build_sklearn(name: str):
    p = dict(FIXED_PARAMS[name])
    if name == 'Random Forest':
        return RandomForestRegressor(**p, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        return HistGradientBoostingRegressor(**p, random_state=RANDOM_STATE)


def _eval(y_true, y_pred) -> dict:
    return dict(
        mae  = mean_absolute_error(y_true, y_pred),
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2   = r2_score(y_true, y_pred),
    )


class Trainer:
    """
    ceramic notebook의 Trainer와 동일한 인터페이스
    fit() → plot_history() → evaluate()
    """

    def __init__(self, preprocessor):
        self.pp              = preprocessor
        self.dm              = preprocessor.dm
        self.trained_models  = {}
        self.torch_trainers  = {}   # PyTorch trainer 저장 (plot_history용)
        self.results         = {}

    def fit(self) -> None:
        pp = self.pp
        dm = self.dm

        print("\n" + "=" * 55)
        print("  모델 학습")
        print("=" * 55)

        # ── RF / GBM ──────────────────────
        for name in FIXED_PARAMS:
            print(f"\n[{name}] 학습 중...")
            Xtr = dm.X_train
            Xte = dm.X_test

            m_ps  = _build_sklearn(name)
            m_uts = _build_sklearn(name)
            m_ps.fit(Xtr,  dm.y_ps_train)
            m_uts.fit(Xtr, dm.y_uts_train)

            pred_ps  = m_ps.predict(Xte)
            pred_uts = m_uts.predict(Xte)

            ps_m  = _eval(dm.y_ps_test,  pred_ps)
            uts_m = _eval(dm.y_uts_test, pred_uts)

            self.results[name] = {
                'ps_mae':  ps_m['mae'],  'ps_rmse':  ps_m['rmse'],  'ps_r2':  ps_m['r2'],
                'uts_mae': uts_m['mae'], 'uts_rmse': uts_m['rmse'], 'uts_r2': uts_m['r2'],
                'pred_ps':  pred_ps,
                'pred_uts': pred_uts,
            }
            self.trained_models[name] = {'ps': m_ps, 'uts': m_uts}
            print(f"  Proof Stress → MAE: {ps_m['mae']:.2f}, R²: {ps_m['r2']:.4f}")
            print(f"  UTS          → MAE: {uts_m['mae']:.2f}, R²: {uts_m['r2']:.4f}")

        # ── Neural Network (PyTorch) ───────
        print(f"\n[Neural Network] PyTorch 학습 중...")
        input_shape = pp.X_train_s.shape[1]

        # Proof Stress
        print("  Proof Stress 학습...")
        tt_ps = TorchTrainer(input_shape, **TORCH_PARAMS)
        tt_ps.fit(pp.X_train_s, dm.y_ps_train,
                  pp.X_val_s,   dm.y_ps_val)
        pred_ps = tt_ps.predict(pp.X_test_s)

        # UTS
        print("  UTS 학습...")
        tt_uts = TorchTrainer(input_shape, **TORCH_PARAMS)
        tt_uts.fit(pp.X_train_s, dm.y_uts_train,
                   pp.X_val_s,   dm.y_uts_val)
        pred_uts = tt_uts.predict(pp.X_test_s)

        ps_m  = _eval(dm.y_ps_test,  pred_ps)
        uts_m = _eval(dm.y_uts_test, pred_uts)

        self.results['Neural Network'] = {
            'ps_mae':  ps_m['mae'],  'ps_rmse':  ps_m['rmse'],  'ps_r2':  ps_m['r2'],
            'uts_mae': uts_m['mae'], 'uts_rmse': uts_m['rmse'], 'uts_r2': uts_m['r2'],
            'pred_ps':  pred_ps,
            'pred_uts': pred_uts,
        }
        self.trained_models['Neural Network'] = {'ps': tt_ps, 'uts': tt_uts}
        self.torch_trainers = {'ps': tt_ps, 'uts': tt_uts}
        print(f"  Proof Stress → MAE: {ps_m['mae']:.2f}, R²: {ps_m['r2']:.4f}")
        print(f"  UTS          → MAE: {uts_m['mae']:.2f}, R²: {uts_m['r2']:.4f}")

    def plot_history(self) -> None:
        """
        ceramic trainer.plot_history()와 동일
        Train loss + Val loss 같이 출력
        """
        plt.rcParams['font.family']        = FONT_FAMILY
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Neural Network - 학습 Loss 커브', fontsize=14, fontweight='bold')

        for ax, key, title in zip(axes, ['ps', 'uts'], ['Proof Stress', 'UTS']):
            tt = self.torch_trainers[key]
            ax.plot(tt.train_losses, color='#1f77b4', linewidth=1.0,
                    alpha=0.7, label='Train loss')
            ax.plot(tt.val_losses,   color='#ff7f0e', linewidth=2.0,
                    label='Val loss')
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss (MAE)', fontsize=10)
            ax.set_title(f'Loss  ({title})', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/2_train_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("학습 곡선 저장 완료 → 2_train_history.png")

    def evaluate(self) -> None:
        """
        ceramic trainer.evaluate()와 동일
        테스트 세트 성능 출력 + 시각화
        """
        plt.rcParams['font.family']        = FONT_FAMILY
        plt.rcParams['axes.unicode_minus'] = False

        dm = self.dm

        # ── 성능 요약 출력 ────────────────
        print("\n" + "=" * 65)
        print("  모델 평가 결과 (테스트 세트)")
        print("=" * 65)
        print(f"{'모델':<22} {'PS_MAE':>8} {'PS_RMSE':>9} {'PS_R²':>8} "
              f"{'UTS_MAE':>9} {'UTS_RMSE':>10} {'UTS_R²':>8}")
        print("-" * 65)
        for name, r in self.results.items():
            print(f"{name:<22} {r['ps_mae']:>7.2f}  {r['ps_rmse']:>8.2f}  {r['ps_r2']:>7.4f}  "
                  f"{r['uts_mae']:>8.2f}  {r['uts_rmse']:>9.2f}  {r['uts_r2']:>7.4f}")
        print("=" * 65)

        # ── 예측 vs 실제 시각화 ───────────
        model_names = list(self.results.keys())
        n_models    = len(model_names)
        fig, axes   = plt.subplots(n_models, 4, figsize=(20, 6 * n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle('모델별 예측 성능 비교', fontsize=16, fontweight='bold')

        for row, name in enumerate(model_names):
            res   = self.results[name]
            color = COLORS[name]
            pairs = [
                (dm.y_ps_test,  res['pred_ps'],  f'Proof Stress (R²={res["ps_r2"]:.3f})'),
                (dm.y_uts_test, res['pred_uts'], f'UTS (R²={res["uts_r2"]:.3f})'),
            ]
            for col_offset, (y_true, y_pred, subtitle) in enumerate(pairs):
                ax = axes[row, col_offset]
                lo = min(y_true.min(), y_pred.min()) - 5
                hi = max(y_true.max(), y_pred.max()) + 5
                ax.scatter(y_true, y_pred, alpha=0.3, s=8, color=color)
                ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
                ax.set_xlabel('실제 (MPa)', fontsize=9)
                ax.set_ylabel('예측 (MPa)', fontsize=9)
                ax.set_title(f'{name}\n{subtitle}', fontsize=10, fontweight='bold')
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
                ax.grid(alpha=0.3)

                ax = axes[row, col_offset + 2]
                residuals = y_true - y_pred
                ax.scatter(y_pred, residuals, alpha=0.3, s=8, color=color)
                ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
                ax.set_xlabel('예측값 (MPa)', fontsize=9)
                ax.set_ylabel('잔차 (MPa)', fontsize=9)
                ax.set_title(f'{name}\n{subtitle.split("(")[0].strip()} 잔차',
                             fontsize=10, fontweight='bold')
                ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/3_model_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("예측 성능 시각화 저장 완료 → 3_model_performance.png")

        # ── 지표 비교 차트 ────────────────
        short_names = {'Random Forest':'RF','Gradient Boosting':'GBM','Neural Network':'MLP'}
        bar_colors  = [COLORS[n] for n in model_names]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('모델 성능 지표 비교', fontsize=14, fontweight='bold')

        for ax, (title, ps_key, uts_key) in zip(axes, [
            ('MAE (MPa)',  'ps_mae',  'uts_mae'),
            ('RMSE (MPa)', 'ps_rmse', 'uts_rmse'),
            ('R²',         'ps_r2',   'uts_r2'),
        ]):
            ps_vals  = [self.results[n][ps_key]  for n in model_names]
            uts_vals = [self.results[n][uts_key] for n in model_names]
            x, w = np.arange(len(model_names)), 0.35

            bars1 = ax.bar(x - w/2, ps_vals,  w, label='Proof Stress',
                           color=[c+'CC' for c in bar_colors], edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + w/2, uts_vals, w, label='UTS',
                           color=bar_colors, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([short_names.get(n, n) for n in model_names], fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(); ax.grid(axis='y', alpha=0.3)

            for bar in [*bars1, *bars2]:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01*abs(bar.get_height()),
                        f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/4_metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("지표 비교 차트 저장 완료 → 4_metrics_comparison.png")

    def save_models(self) -> None:
        os.makedirs(f'{OUTPUT_DIR}/models', exist_ok=True)
        best_name = max(self.results, key=lambda x: self.results[x]['ps_r2'])
        joblib.dump(self.trained_models[best_name]['ps'],  f'{OUTPUT_DIR}/models/best_model_ps.pkl')
        joblib.dump(self.trained_models[best_name]['uts'], f'{OUTPUT_DIR}/models/best_model_uts.pkl')
        print(f"최우수 모델 저장 완료: {best_name} → outputs/models/")
        return best_name