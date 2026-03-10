# AI Materials Discovery Platform


AI 기반 계산 재료 탐색(Materials Discovery) 및 물성 예측(Property Prediction)을 위한 연구성 프로젝트입니다.

## 프로젝트 목적
- 노트북 중심으로 빠르게 실험을 반복합니다.
- 검증된 로직은 `src/`로 분리해 재사용성을 높입니다.
- 학습 결과와 산출물을 체계적으로 관리합니다.

## 저장소 구조

```text
ai-materials-discovery-platform/
├── notebooks/      # 실험/분석용 Jupyter 노트북
├── data/
│   ├── raw/        # 원본 데이터
│   └── processed/  # 전처리/가공 데이터
├── src/            # 재사용 가능한 Python 모듈
├── models/         # 학습된 모델 및 체크포인트
├── outputs/        # 시각화, 로그, 예측 결과물
├── scripts/        # 학습/추론/유틸 실행 스크립트
└── docs/           # 프로젝트 문서
```

## 권장 워크플로우
1. `notebooks/`에서 아이디어를 빠르게 검증합니다.
2. 반복 사용되는 코드를 `src/`로 이동해 모듈화합니다.
3. 모델 학습 결과를 `models/`에 저장합니다.
4. 그래프/리포트/예측값을 `outputs/`에 정리합니다.
5. 사용 방법과 의사결정을 `docs/`에 문서화합니다.

## 참고
- 빈 디렉터리 유지를 위해 각 폴더에 `.gitkeep` 파일이 포함되어 있습니다.
