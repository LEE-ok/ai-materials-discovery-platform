"""
api/config.py - 전역 설정
오스테나이트계 철강 고온 강도 예측 AI 모델
"""

# ── 데이터 ────────────────────────────────
SHEET_NAME = 'alldata'
HEADER_ROW = 5

COMPOSITION_FEATURES = [
    'Cr','Ni','Mo','Mn','Si','Nb','Ti','Zr','Ta','V','W',
    'Cu','N','C','B','P','S','Co','Al','Sn','Pb'
]

PROCESS_FEATURES = [
    'Solution_treatment_temperature', 'Temperature (K)',
    'Product form', 'Type of melting'
]

NUMERIC_COLS = [
    'Solution_treatment_temperature', 'Solution_treatment_time(s)',
    'Grains mm-2', 'Elongation (%)', 'Area_reduction (%)'
]

ALL_FEATURES = COMPOSITION_FEATURES + PROCESS_FEATURES
TARGET_PS    = '0.2%proof_stress (M Pa)'
TARGET_UTS   = 'UTS (M Pa)'
TARGETS      = [TARGET_PS, TARGET_UTS]

FEATURE_DISPLAY_NAMES = COMPOSITION_FEATURES + ['용체화 온도', '시험 온도(K)', '제품 형태', '용해 방법']

# ── 학습 ──────────────────────────────────
TRAIN_SIZE   = 0.7
VAL_SIZE     = 0.15
TEST_SIZE    = 0.15
RANDOM_STATE = 42

# ── 모델 파라미터 ─────────────────────────
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

TORCH_PARAMS = dict(
    epochs     = 100,
    batch_size = 64,
    lr         = 1e-3,
    patience   = 30,
)
BNN_PARAMS = dict(
    epochs       = 100,
    batch_size   = 64,
    lr           = 1e-3,
    patience     = 30,
    dropout_rate = 0.1,
    n_samples    = 100,  # MC Dropout 샘플 수 (높을수록 정확하지만 느림)
)
# ── 시각화 ────────────────────────────────
COLORS = {
    'Random Forest':     '#4C72B0',
    'Gradient Boosting': '#DD8452',
    'Neural Network':    '#55A868',
}
FONT_FAMILY = 'Malgun Gothic'
OUTPUT_DIR = 'outputs'
