"""
visualizer.py - 학습 결과 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from config import (
    ALL_FEATURES, COLORS, FEATURE_DISPLAY_NAMES,
    FONT_FAMILY, OUTPUT_DIR, TARGET_PS, TARGET_UTS,
)


def _setup():
    plt.rcParams['font.family']        = FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = False


# ── 1. Optuna 최적 파라미터 요약 ──────────

def plot_optuna_params(best_params_all: dict) -> None:
    _setup()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Optuna 최적 파라미터 요약 (Proof Stress 기준)', fontsize=14, fontweight='bold')

    for ax, name in zip(axes, best_params_all):
        bp   = best_params_all[name]['ps']
        text = '\n'.join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in bp.items()
        )
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(facecolor=COLORS[name], alpha=0.15, edgecolor=COLORS[name]))
        ax.set_title(f'{name}\n최적 파라미터 (PS)', fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/2a_optuna_best_params.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Optuna 파라미터 시각화 저장 완료 → 2a_optuna_best_params.png")


# ── 2. 예측 vs 실제 / 잔차 ────────────────

def plot_model_performance(results: dict, y_ps_test, y_uts_test) -> None:
    _setup()
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 4, figsize=(20, 6 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('모델별 예측 성능 비교 (Optuna 최적화)', fontsize=16, fontweight='bold')

    for row, (name, res) in enumerate(results.items()):
        color = COLORS[name]
        pairs = [
            (y_ps_test,  res['pred_ps'],  f'Proof Stress (R²={res["ps_r2"]:.3f})'),
            (y_uts_test, res['pred_uts'], f'UTS (R²={res["uts_r2"]:.3f})'),
        ]
        for col_offset, (y_true, y_pred, subtitle) in enumerate(pairs):
            # 예측 vs 실제
            ax = axes[row, col_offset]
            lo = min(y_true.min(), y_pred.min()) - 5
            hi = max(y_true.max(), y_pred.max()) + 5
            ax.scatter(y_true, y_pred, alpha=0.3, s=8, color=color)
            ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
            ax.set_xlabel('실제 (MPa)', fontsize=9); ax.set_ylabel('예측 (MPa)', fontsize=9)
            ax.set_title(f'{name}\n{subtitle}', fontsize=10, fontweight='bold')
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.grid(alpha=0.3)

            # 잔차
            ax = axes[row, col_offset + 2]
            residuals = y_true - y_pred
            ax.scatter(y_pred, residuals, alpha=0.3, s=8, color=color)
            ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
            ax.set_xlabel('예측값 (MPa)', fontsize=9); ax.set_ylabel('잔차 (MPa)', fontsize=9)
            ax.set_title(f'{name}\n{subtitle.split("(")[0].strip()} 잔차', fontsize=10, fontweight='bold')
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/2_model_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("모델 성능 시각화 저장 완료 → 2_model_performance.png")


# ── 3. 지표 비교 막대 차트 ────────────────

def plot_metrics_comparison(results: dict) -> None:
    _setup()
    model_names = list(results.keys())
    short_names = [{'Random Forest': 'RF', 'Gradient Boosting': 'GBM', 'Neural Network': 'MLP'}.get(n, n) for n in model_names]
    bar_colors  = [COLORS[n] for n in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('모델 성능 지표 비교 (Optuna 최적화)', fontsize=14, fontweight='bold')

    for ax, (title, ps_key, uts_key) in zip(axes, [
        ('MAE (MPa)',  'ps_mae',  'uts_mae'),
        ('RMSE (MPa)', 'ps_rmse', 'uts_rmse'),
        ('R²',         'ps_r2',   'uts_r2'),
    ]):
        ps_vals  = [results[n][ps_key]  for n in model_names]
        uts_vals = [results[n][uts_key] for n in model_names]
        x, w = np.arange(len(model_names)), 0.35

        bars1 = ax.bar(x - w/2, ps_vals,  w, label='Proof Stress',
                       color=[c + 'CC' for c in bar_colors], edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + w/2, uts_vals, w, label='UTS',
                       color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)

        for bar in [*bars1, *bars2]:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * abs(bar.get_height()),
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/3_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("지표 비교 차트 저장 완료 → 3_metrics_comparison.png")


# ── 4. 피처 중요도 ────────────────────────

def plot_feature_importance(trained_models: dict) -> None:
    _setup()
    rf_ps  = trained_models['Random Forest']['ps']
    rf_uts = trained_models['Random Forest']['uts']

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Random Forest 피처 중요도 (Optuna 최적화)', fontsize=14, fontweight='bold')

    for ax, model, title in zip(axes, [rf_ps, rf_uts], ['항복강도 (Proof Stress)', 'UTS']):
        imp        = model.feature_importances_
        sorted_idx = np.argsort(imp)
        bar_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(imp)))

        ax.barh(np.array(FEATURE_DISPLAY_NAMES)[sorted_idx], imp[sorted_idx],
                color=bar_colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('중요도 (Feature Importance)', fontsize=11)
        ax.set_title(f'{title} 예측\n피처 중요도', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for pos in range(len(sorted_idx) - 5, len(sorted_idx)):
            ax.get_children()[pos].set_edgecolor('red')
            ax.get_children()[pos].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/4_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("피처 중요도 저장 완료 → 4_feature_importance.png")


# ── 5. 온도별 오차 분석 ───────────────────

def plot_temp_error(trained_models: dict, X_test, y_ps_test, y_uts_test) -> None:
    _setup()
    import pandas as pd
    rf_ps  = trained_models['Random Forest']['ps']
    rf_uts = trained_models['Random Forest']['uts']

    pred_ps_rf  = rf_ps.predict(X_test)
    pred_uts_rf = rf_uts.predict(X_test)
    test_temps  = pd.DataFrame(X_test, columns=ALL_FEATURES)['Temperature (K)'].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Random Forest - 온도별 예측 오차 분석 (Optuna 최적화)', fontsize=14, fontweight='bold')

    for ax, pred, actual, title in zip(
        axes,
        [pred_ps_rf, pred_uts_rf],
        [y_ps_test,  y_uts_test],
        ['항복강도 (Proof Stress)', 'UTS'],
    ):
        mae_by_temp = {
            t: mean_absolute_error(actual[test_temps == t], pred[test_temps == t])
            for t in sorted(np.unique(test_temps))
            if (test_temps == t).sum() >= 3
        }
        ts   = sorted(mae_by_temp)
        maes = [mae_by_temp[t] for t in ts]

        ax.bar(range(len(ts)), maes, color='#4C72B0', alpha=0.8, edgecolor='white')
        ax.set_xticks(range(len(ts)))
        ax.set_xticklabels([str(int(t)) for t in ts], rotation=45, fontsize=8)
        ax.set_xlabel('Temperature (K)', fontsize=11)
        ax.set_ylabel('MAE (MPa)', fontsize=11)
        ax.set_title(f'{title}\n온도별 MAE', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        avg = np.mean(maes)
        ax.axhline(avg, color='red', linestyle='--', linewidth=1.5, label=f'평균 MAE: {avg:.1f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/5_temp_error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("온도별 오차 분석 저장 완료 → 5_temp_error_analysis.png")