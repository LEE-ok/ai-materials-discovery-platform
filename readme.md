# AI Materials Discovery Platform

전산재료를 통해 신소재를 찾기 위한 인공지능모델 및 플랫폼 개발 (Development of an AI model and platform to find new materials through computational materials).

## 개요 (Overview)
본 프로젝트는 동아대학교 SW중심대학사업 현장미러형 연계 프로젝트의 일환으로 진행됩니다. 
소재 데이터(MatCalc, koMAP, 문헌 데이터)를 수집, 정량화하여 신소재 탐색과 특성 예측이 가능한 통합형 인공지능 모델 및 플랫폼을 구축합니다.

## 주요 기능 (Features)
1. **데이터 파이프라인**: MatCalc 시뮬레이션 결과 및 koMAP 등에서 소재 데이터 자동 추출 및 정량화.
2. **AI 예측 모델**: 합금 성분, 열처리 조건, 미세조직 특성 등을 입력받아 기계적 물성(강도, 경도, 인성 등)과 예측 신뢰도 도출.
3. **학습용 소프트웨어**: 모델 구조, 하이퍼파라미터 실험 및 시각화 도구.
4. **추론용 소프트웨어 (GUI)**: 사용자가 조성 및 조건값을 입력하면 예측 물성값, 오차범위, 신뢰도를 확인 가능한 데스크탑 어플리케이션.

## 구조 (Architecture)
- `src/data_pipeline/`: 데이터 수집 및 정처리 모듈
- `src/models/`: TensorFlow 및 TensorFlow Probability 기반 AI 모델
- `src/api/`: Flask 기반 추론용 API 서버
- `src/gui/`: PyQt6 기반 클라이언트 어플리케이션
- `main.py`: 프로그램 실행 진입점

## 설치 및 실행 (Setup & Run)
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```
