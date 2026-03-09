# AI Materials Discovery Platform

AI 기반 계산 재료 탐색 및 물성 예측 플랫폼입니다.

## 개요
이 저장소는 머신러닝과 계산 재료 데이터를 활용하여 새로운 재료를 발굴하기 위한 실험, 모델, 소프트웨어 컴포넌트를 포함합니다.

워크플로우는 노트북 중심으로 진행되며, 점진적으로 재사용 가능한 Python 코드로 모듈화됩니다.

## 저장소 구조

ai-materials-discovery-platform/
├── notebooks        # Jupyter 실험
├── data             # 데이터셋
│   ├── raw
│   └── processed
├── src              # 재사용 가능한 Python 모듈
├── models           # 학습된 모델
├── outputs          # 그림, 로그, 예측 결과
├── scripts          # 학습 / 추론 스크립트
└── docs             # 문서

## 워크플로우

1. 노트북에서 탐색 및 실험 수행
2. 재사용 가능한 코드를 `src`로 이동
3. 모델을 학습하고 `models`에 저장
4. 결과를 `outputs`에 저장
