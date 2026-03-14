"""
config.py - 전역 설정
"""

# ── 경로 ──────────────────────────────────
DATA_PATH    = '../data/STMECH_AUS_SS.xls'
OUTPUT_DIR   = '../outputs'

# ── 데이터 ────────────────────────────────
SHEET_NAME   = 'alldata'
HEADER_ROW   = 5

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

ALL_FEATURES  = COMPOSITION_FEATURES + PROCESS_FEATURES
TARGET_PS     = '0.2%proof_stress (M Pa)'
TARGET_UTS    = 'UTS (M Pa)'

FEATURE_DISPLAY_NAMES = COMPOSITION_FEATURES + ['용체화 온도', '시험 온도(K)', '제품 형태', '용해 방법']

# ── 학습 ──────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5

# 모델별 Optuna trials (MLP는 느리므로 적게 설정)
N_TRIALS_MAP = {
    'Random Forest':     20,
    'Gradient Boosting': 20,
    'Neural Network':    10,  # MLP는 느려서 적게
}

# ── 시각화 ────────────────────────────────
COLORS = {
    'Random Forest':     '#4C72B0',
    'Gradient Boosting': '#DD8452',
    'Neural Network':    '#55A868',
}
FONT_FAMILY = 'Malgun Gothic'