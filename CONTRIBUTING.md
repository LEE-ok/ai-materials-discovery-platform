# Branch & PR Rules

이 문서는 `ai-materials-discovery-platform` 저장소의 브랜치 전략, 머지 순서, PR 규칙을 정의합니다.

## 1) 기본 원칙
- `main`은 항상 실행 가능한 안정 상태를 유지합니다.
- 개발은 개인 기준이 아니라 기능/파트 기준으로 진행합니다.
- 브랜치는 짧게 유지합니다(권장 1~3일, 최대 5일).
- PR은 작고 자주 올립니다(1 PR = 1 목적).
- 장기 브랜치와 대형 PR을 피합니다.

## 2) 브랜치 종류
- `main`: 운영/기준 브랜치
- `feature/*`: 신규 기능
- `fix/*`: 버그 수정
- `docs/*`: 문서 수정
- `exp/*`: 실험 코드(검증 후 feature로 정리)
- `hotfix/*`: `main` 긴급 수정
- `release/*`: 릴리즈 준비(선택)

## 3) 브랜치 네이밍 규칙
- 형식: `<type>/<area>-<topic>`
- 예시:
- `feature/gui-training-tab`
- `feature/model-uncertainty`
- `feature/api-predict-endpoint`
- `fix/data-xls-header-parse`
- `docs/branch-policy-update`

권장 area:
- `data`, `model`, `gui`, `api`, `docs`, `infra`

## 4) 파트 브랜치와 세부 브랜치 운영
Git은 "브랜치 안의 브랜치" 개념이 없지만, 특정 브랜치에서 새 브랜치를 만들면 세부 분기처럼 운용할 수 있습니다.

예시:
```bash
git checkout feature/gui
git checkout -b feature/gui-performance-tab
```

운영 규칙:
- 세부 브랜치는 상위 브랜치 기준으로 생성합니다.
- 세부 브랜치 완료 후 상위 브랜치로 먼저 머지합니다.
- 상위 브랜치 안정화 후 `main`으로 머지합니다.
- 계층 깊이는 1단계까지만 권장합니다.

## 5) 표준 작업 흐름
### 5-1. 새 작업 시작
```bash
git checkout main
git pull origin main
git checkout -b feature/<area>-<topic>
```

### 5-2. 작업 중 동기화 (드리프트 방지)
최소 하루 1회:
```bash
git fetch origin
git checkout <my-branch>
git merge origin/main
```
팀 합의 시 `rebase` 사용 가능:
```bash
git fetch origin
git rebase origin/main
```

### 5-3. PR 생성 후
- 리뷰 반영
- 충돌 해결
- 테스트 재실행
- 머지

### 5-4. 머지 후 정리
```bash
git checkout main
git pull origin main
git branch -d <my-branch>
```

## 6) PR 규칙
- PR 1개 = 기능/수정 1개
- Draft PR 적극 사용
- 제목 형식:
- `[DATA] xls 전처리 파이프라인 개선`
- `[MODEL] UTS 튜닝 전략 추가`
- `[GUI] 추론 탭 파일 선택 UX 개선`

PR 본문 필수 항목:
- 목적
- 변경 파일/모듈
- 테스트 방법 및 결과
- 영향 범위(`data/model/gui/api/docs`)
- 리스크 및 롤백 방법

권장 PR 크기:
- 변경 파일 3~12개 권장
- 500라인 이상이면 분할 검토

## 7) 커밋 메시지 규칙
- 형식: `<type>(<area>): <summary>`
- 예시:
- `feat(model): add uncertainty-aware prediction output`
- `fix(gui): handle missing input file extension`
- `docs(branch): add hierarchical branch workflow`

권장 type:
- `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

## 8) 충돌 대응 규칙
- 공통 파일 변경 시 담당자와 사전 공유합니다.
- 충돌 발생 시 최신 `main` 반영 후 해결합니다.
- 해결 후 반드시 실행 검증합니다.
- 특히 아래 계약 파일은 먼저 합의 후 수정합니다:
- 데이터 스키마(컬럼명/단위)
- 모델 입출력 형식
- API 응답 키

## 9) release/hotfix 규칙
### 9-1. 릴리즈(선택)
```bash
git checkout -b release/v0.2.0 main
```
- 문서/버전 정리 후 `main` 머지
- 태그 생성: `v0.2.0`

### 9-2. 긴급 수정
```bash
git checkout -b hotfix/api-response-error main
```
- 수정 완료 후 `main`으로 즉시 PR
- 필요 시 관련 feature 브랜치에도 동일 수정 반영

## 10) 금지 사항
- `main` 직접 커밋/푸시 금지
- 리뷰 없이 머지 금지
- 대규모 리팩토링과 기능 추가를 한 PR에 혼합 금지
- 임시 산출물(`__pycache__`, 임시 csv/png/log) 커밋 금지

## 11) PR 전 체크리스트
- [ ] `git fetch origin` 후 `main` 동기화 반영 완료
- [ ] 로컬 실행 또는 테스트 통과 확인
- [ ] 영향 범위/리스크를 PR에 기재
- [ ] 문서 변경 필요 시 함께 반영
- [ ] 불필요 파일 제외 확인
