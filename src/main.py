"""
main.py - 파이프라인 진입점
오스테나이트계 철강 고온 강도 예측 AI 모델

ceramic notebook 흐름과 동일:
  DataManager → Analyzer → Preprocessor → Trainer → Predictor

실행:
    cd src
    python main.py
"""

import os
import sys
import warnings
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from config import OUTPUT_DIR
from data_loader  import DataManager
from analyzer     import Analyzer
from preprocessor import Preprocessor
from trainer      import Trainer
from predictor    import Predictor


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ══════════════════════════════════════════
    # STEP 1. 데이터 로드
    # ceramic: data = DataManager()
    #          data.load_csv(...)
    #          data.summary()
    # ══════════════════════════════════════════
    print("=" * 55)
    print("  STEP 1 : 데이터 로드")
    print("=" * 55)
    data = DataManager()
    data.load()
    data.summary()

    # ══════════════════════════════════════════
    # STEP 2. EDA
    # ceramic: analyzer = Analyzer(data)
    #          analyzer.plot_column()
    #          analyzer.violinplot_normalized_column()
    #          analyzer.plot_correlation_matrix()
    #          analyzer.plot_histogram_column()
    # ══════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  STEP 2 : EDA (탐색적 데이터 분석)")
    print("=" * 55)
    analyzer = Analyzer(data)
    analyzer.plot_column()
    analyzer.violinplot_normalized_column()
    analyzer.plot_correlation_matrix()
    analyzer.plot_histogram_column()
    analyzer.plot_temp_distribution()

    # ══════════════════════════════════════════
    # STEP 3. 전처리 (스케일링)
    # ceramic: preprocessor = Preprocessor(data)
    #          preprocessor.standard_scaling(x=True, y=True)
    # ══════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  STEP 3 : 전처리 (StandardScaling)")
    print("=" * 55)
    preprocessor = Preprocessor(data)
    preprocessor.standard_scaling()
    preprocessor.save_scalers()

    # ══════════════════════════════════════════
    # STEP 4. 학습
    # ceramic: model   = Model(...)
    #          trainer = Trainer(model=model, ...)
    #          trainer.fit()
    #          trainer.plot_history()
    # ══════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  STEP 4 : 모델 학습 (Optuna 최적화)")
    print("=" * 55)
    trainer = Trainer(preprocessor)
    trainer.fit()
    trainer.plot_history()

    # ══════════════════════════════════════════
    # STEP 5. 평가
    # ceramic: trainer.evaluate()
    # ══════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  STEP 5 : 모델 평가")
    print("=" * 55)
    trainer.evaluate()
    trainer.save_models()

    # ══════════════════════════════════════════
    # STEP 6. 예측
    # ceramic: predictor = Predictor(model=model, ...)
    #          predictor.predict(x_inverse_scaling=True,
    #                            y_inverse_scaling=True)
    # ══════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  STEP 6 : 예측 (역스케일링 포함)")
    print("=" * 55)
    predictor = Predictor(trainer, preprocessor)
    predictor.predict(inverse_scaling=True)
    predictor.plot_feature_importance()
    predictor.plot_temp_error()

    print("\n" + "=" * 55)
    print("  ✅ 모든 분석 완료!")
    print(f"     결과물 위치: {OUTPUT_DIR}/")
    print("=" * 55)


if __name__ == '__main__':
    main()
