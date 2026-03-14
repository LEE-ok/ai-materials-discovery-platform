"""
predictor.py - 예측 및 피처 중요도 분석
ceramic notebook의 Predictor 역할
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error

from config import (
    ALL_FEATURES, COLORS, FEATURE_DISPLAY_NAMES,
    FONT_FAMILY, OUTPUT_DIR, TARGET_PS, TARGET_UTS,
)


class Predictor:
    """
    ceramic notebook의 Predictor와 동일한 인터페이스
    predict() → 역스케일링 → 피처 중요도 분석
    """

    def __init__(self, trainer, preprocessor):
        self.trainer = trainer
        self.pp      = preprocessor
        self.dm      = preprocessor.dm

    def predict(self, inverse_scaling: bool = True) -> pd.DataFrame:
        """
        테스트 세트 예측 + 역스케일링
        ceramic의 predictor.predict(x_inverse_scaling=True, y_inverse_scaling=True)와 동일
        """
        plt.rcParams['font.family']        = FONT_FAMILY
        plt.rcParams['axes.unicode_minus'] = False

        best_name = max(self.trainer.results, key=lambda x: self.trainer.results[x]['ps_r2'])
        m_ps      = self.trainer.trained_models[best_name]['ps']
        m_uts     = self.trainer.trained_models[best_name]['uts']

        use_scaled = (best_name == 'Neural Network')
        Xte = self.pp.X_test_s if use_scaled else self.dm.X_test

        pred_ps_s  = m_ps.predict(Xte)
        pred_uts_s = m_uts.predict(Xte)

        # 역스케일링
        if inverse_scaling:
            pred_ps  = self.pp.scaler_ps.inverse_transform(pred_ps_s.reshape(-1,1)).ravel()
            pred_uts = self.pp.scaler_uts.inverse_transform(pred_uts_s.reshape(-1,1)).ravel()
            actual_ps  = self.pp.scaler_ps.inverse_transform(
                self.pp.scaler_ps.transform(self.dm.y_ps_test.reshape(-1,1))).ravel()
            actual_uts = self.pp.scaler_uts.inverse_transform(
                self.pp.scaler_uts.transform(self.dm.y_uts_test.reshape(-1,1))).ravel()
        else:
            pred_ps, pred_uts = pred_ps_s, pred_uts_s
            actual_ps, actual_uts = self.dm.y_ps_test, self.dm.y_uts_test

        # 결과 DataFrame
        result_df = pd.DataFrame({
            '실제_PS':  actual_ps,
            '예측_PS':  pred_ps,
            '오차_PS':  actual_ps - pred_ps,
            '실제_UTS': actual_uts,
            '예측_UTS': pred_uts,
            '오차_UTS': actual_uts - pred_uts,
        })

        print(f"\n[{best_name}] 역스케일링 예측 결과 (상위 10개)")
        print(result_df.head(10).round(2).to_string(index=False))
        print(f"\nPS  MAE: {mean_absolute_error(actual_ps, pred_ps):.2f} MPa")
        print(f"UTS MAE: {mean_absolute_error(actual_uts, pred_uts):.2f} MPa")

        result_df.to_csv(f'{OUTPUT_DIR}/predictions.csv', index=False, encoding='utf-8-sig')
        print("예측 결과 저장 완료 → outputs/predictions.csv")
        return result_df

    def plot_feature_importance(self) -> None:
        """Random Forest 피처 중요도"""
        plt.rcParams['font.family']        = FONT_FAMILY
        plt.rcParams['axes.unicode_minus'] = False

        if 'Random Forest' not in self.trainer.trained_models:
            print("Random Forest 모델이 없어 피처 중요도를 출력할 수 없습니다.")
            return

        rf_ps  = self.trainer.trained_models['Random Forest']['ps']
        rf_uts = self.trainer.trained_models['Random Forest']['uts']

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Random Forest 피처 중요도 (Optuna 최적화)', fontsize=14, fontweight='bold')

        for ax, model, title in zip(axes, [rf_ps, rf_uts], ['항복강도 (Proof Stress)', 'UTS']):
            imp        = model.feature_importances_
            sorted_idx = np.argsort(imp)
            colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(imp)))

            ax.barh(np.array(FEATURE_DISPLAY_NAMES)[sorted_idx], imp[sorted_idx],
                    color=colors_bar, edgecolor='white', linewidth=0.5)
            ax.set_xlabel('중요도 (Feature Importance)', fontsize=11)
            ax.set_title(f'{title}\n피처 중요도', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            for pos in range(len(sorted_idx) - 5, len(sorted_idx)):
                ax.get_children()[pos].set_edgecolor('red')
                ax.get_children()[pos].set_linewidth(1.5)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/5_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("피처 중요도 저장 완료 → 5_feature_importance.png")

    def plot_temp_error(self) -> None:
        """온도별 예측 오차 분석"""
        plt.rcParams['font.family']        = FONT_FAMILY
        plt.rcParams['axes.unicode_minus'] = False

        best_name = max(self.trainer.results, key=lambda x: self.trainer.results[x]['ps_r2'])
        res       = self.trainer.results[best_name]
        test_temps = pd.DataFrame(self.dm.X_test, columns=ALL_FEATURES)['Temperature (K)'].values

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{best_name} - 온도별 예측 오차 분석', fontsize=14, fontweight='bold')

        for ax, pred, actual, title in zip(
            axes,
            [res['pred_ps'],      res['pred_uts']],
            [self.dm.y_ps_test,   self.dm.y_uts_test],
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
        plt.savefig(f'{OUTPUT_DIR}/6_temp_error_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("온도별 오차 분석 저장 완료 → 6_temp_error_analysis.png")
