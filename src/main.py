"""
main.py - 파이프라인 진입점

실행:
    cd src
    python main.py
"""

import joblib
import os
import sys

# src 디렉토리를 경로에 추가
sys.path.insert(0, os.path.dirname(__file__))

from config import N_TRIALS_MAP, OUTPUT_DIR, RANDOM_STATE
from data_loader import load_raw, preprocess
from eda import run_eda
from optimizer import optimize_all
from trainer import train_and_evaluate
from visualizer import (
    plot_feature_importance,
    plot_metrics_comparison,
    plot_model_performance,
    plot_optuna_params,
    plot_temp_error,
)

import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. 데이터 로드 & EDA ─────────────
    print("=" * 55)
    print("  STEP 1 : 데이터 로드 및 EDA")
    print("=" * 55)
    df = load_raw()
    run_eda(df)

    # ── 2. 전처리 ────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 2 : 데이터 전처리")
    print("=" * 55)
    (
        X_train, X_test, X_train_s, X_test_s,
        y_ps_train, y_ps_test, y_uts_train, y_uts_test,
        scaler, imputer, df_model,
    ) = preprocess(df)

    # ── 3. Optuna 최적화 ─────────────────
    print("\n" + "=" * 55)
    print(f"  STEP 3 : Optuna 하이퍼파라미터 최적화 (trials={N_TRIALS_MAP})")
    print("=" * 55)
    best_params_all = optimize_all(X_train, X_train_s, y_ps_train, y_uts_train)

    # ── 4. 학습 & 평가 ───────────────────
    print("\n" + "=" * 55)
    print("  STEP 4 : 최적 파라미터로 학습 & 평가")
    print("=" * 55)
    results, trained_models = train_and_evaluate(
        best_params_all,
        X_train, X_test, X_train_s, X_test_s,
        y_ps_train, y_ps_test, y_uts_train, y_uts_test,
    )

    # ── 5. 시각화 ────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 5 : 시각화")
    print("=" * 55)
    plot_optuna_params(best_params_all)
    plot_model_performance(results, y_ps_test, y_uts_test)
    plot_metrics_comparison(results)
    plot_feature_importance(trained_models)
    plot_temp_error(trained_models, X_test, y_ps_test, y_uts_test)

    # ── 6. 최종 요약 & 모델 저장 ─────────
    print("\n" + "=" * 55)
    print("       최종 성능 요약 (Optuna 최적화 / 테스트 세트)")
    print("=" * 55)
    print(f"{'모델':<22} {'PS_MAE':>8} {'PS_R²':>8} {'UTS_MAE':>9} {'UTS_R²':>8}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<22} {r['ps_mae']:>7.2f}  {r['ps_r2']:>7.4f}  {r['uts_mae']:>8.2f}  {r['uts_r2']:>7.4f}")
    print("=" * 55)

    best_name = max(results, key=lambda x: results[x]['ps_r2'])
    print(f"\n최우수 모델: {best_name}")

    joblib.dump(trained_models[best_name]['ps'],  f'{OUTPUT_DIR}/best_model_ps.pkl')
    joblib.dump(trained_models[best_name]['uts'], f'{OUTPUT_DIR}/best_model_uts.pkl')
    joblib.dump(scaler,  f'{OUTPUT_DIR}/scaler.pkl')
    joblib.dump(imputer, f'{OUTPUT_DIR}/imputer.pkl')
    print("모델 저장 완료: best_model_ps.pkl, best_model_uts.pkl")
    print("\n✅ 모든 분석 완료!")


if __name__ == '__main__':
    main()