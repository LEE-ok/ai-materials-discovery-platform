# 🧪 AI 소재 고속 발견 플랫폼 (AI Materials Discovery Platform)

본 프로젝트는 머신러닝 알고리즘을 활용하여 신소재 함량에 따른 **물성(항복강도, Yield Strength)**을 예측하고, 최적의 성분 배합 비율을 자동으로 도출해주는 완성형 AI 솔루션입니다.

## 🚀 주요 기능 (What We Built)

1.  **다중 AI 모델 가동 엔진**:
    *   **TFP (확률형 신경망)**: 예측값과 함께 '신뢰도(Uncertainty)'를 제공하여 고부가가치 연구에 최적화된 모델
    *   **MLP (일반 신경망)**: 고성능 딥러닝 기반 예측 모델
    *   **RF (랜덤 포레스트)**: 데이터가 적을 때도 안정적인 성능을 보이는 기계학습 모델
    *   **GBM (그래디언트 부스팅)**: 최고의 예측 정밀도를 목표로 하는 부스팅 모델
2.  **지능형 데이터 분석 (EDA)**:
    *   전체 성분 분포 히스토그램, 상관관계 히트맵, 바이올린 플롯을 통한 데이터 정규성 분석
3.  **최적 소재 설계 (Optimization)**:
    *   강도를 극대화하는 성분 비율을 수학적으로 탐색하며, 진행 과정을 **프로그레스 바**로 실시간 확인 가능
4.  **AI 분석 리포트 자동 생성**:
    *   데이터 분석 결과부터 AI의 학습 성과, 최적화 근거까지 한글로 된 보고서를 실시간 생성

## 🛠️ 실행 방법 (How to Run)

1.  **환경 준비**:
    ```bash
    # 가상환경 활성화 (Windows 기준)
    .\venv\Scripts\activate
    
    # 필수 패키지 설치
    pip install -r requirements.txt
    ```
2.  **프로그램 실행**:
    *   방법 1: `python main.py` 실행 (GUI와 API 서버가 동시에 켜집니다.)
    *   방법 2: 수동 실행
        *   API 서버: `python src/api/app.py` (포트 5000)
        *   GUI 메인: `python src/gui/main_window.py`

## 📊 사용된 AI 모델 상세

*   **Deep Learning (TensorFlow)**: 복잡한 성분 간의 비선형 시너지를 파악하는 데 사용
*   **Probabilistic AI (TFP)**: 소재 데이터의 부족함을 보완하기 위해 '확률' 개념을 도입하여 정밀도 향상
*   **Ensemble (Scikit-learn)**: 안정적인 통계 기반 예측 수행

---
본 결과물은 `branch_branch`에 안전하게 기록되어 있으며, 기존 `main` 브랜치의 기록을 유지하면서도 최신 AI 기능을 모두 포함하고 있습니다.
