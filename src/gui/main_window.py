"""
gui/main_window.py - Streamlit UI (블루/그레이 전문 테마)
"""

import os
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from api.config import (
    ALL_FEATURES, COLORS, FEATURE_DISPLAY_NAMES,
    FONT_FAMILY, OUTPUT_DIR, TARGET_PS, TARGET_UTS,
)
from data_pipeline.parser import DataManager, Preprocessor
from models.ai_model import Trainer, Predictor

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 그래프 테마 ───────────────────────────
BG_MAIN   = '#ffffff'   # 흰 배경
BG_PLOT   = '#ffffff'   # 플롯 영역
CLR_TEXT  = '#1a1a1a'   # 텍스트
CLR_GRID  = '#e0e0e0'   # 그리드
CLR_SPINE = '#888888'   # 테두리
CLR_BLUE  = '#2166ac'   # 메인 파란색
CLR_ORANGE= '#d6604d'   # 보조 빨강
CLR_RED   = '#b2182b'   # 강조 빨강
CLR_GREEN = '#1a9641'   # 성공 초록

CHART_COLORS = {
    'Random Forest':     '#2166ac',
    'Gradient Boosting': '#d6604d',
    'Neural Network':    '#1a9641',
    'BNN':               '#762a83',
}

def _fig_style(fig, axes_list=None):
    """그래프 전체 스타일 적용"""
    fig.patch.set_facecolor(BG_MAIN)
    if axes_list is not None:
        import numpy as _np
        ax_iter = _np.array(axes_list).flatten() if hasattr(axes_list, '__iter__') else [axes_list]
        for ax in ax_iter:
            ax.set_facecolor(BG_PLOT)
            ax.tick_params(colors=CLR_TEXT, labelsize=9)
            ax.xaxis.label.set_color(CLR_TEXT)
            ax.yaxis.label.set_color(CLR_TEXT)
            ax.title.set_color(CLR_TEXT)
            for spine in ax.spines.values():
                spine.set_color(CLR_SPINE)
            ax.tick_params(direction='in')
            ax.grid(alpha=0.4, color=CLR_GRID, linewidth=0.6, linestyle='--')

plt.rcParams.update({
    'font.family':        FONT_FAMILY,
    'axes.unicode_minus': False,
    'figure.facecolor':   BG_MAIN,
    'axes.facecolor':     BG_PLOT,
    'text.color':         CLR_TEXT,
    'axes.labelcolor':    CLR_TEXT,
    'xtick.color':        CLR_TEXT,
    'ytick.color':        CLR_TEXT,
    'axes.edgecolor':     CLR_SPINE,
    'grid.color':         CLR_GRID,
    'grid.alpha':         0.25,
    'axes.titlepad':      10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '#cccccc',
})

# ── Streamlit 스타일 (논문/보고서) ──────────
STYLE = """
<style>
    .stApp { background-color: #f8f9fa; }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dee2e6;
    }

    /* 헤더 */
    .app-header {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-top: 4px solid #2166ac;
        border-radius: 4px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    .app-header h1 {
        color: #1a1a1a;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    .app-header p { color: #6c757d; font-size: 0.85rem; margin: 0; }

    /* 섹션 */
    .sec-title {
        color: #1a1a1a;
        font-size: 0.95rem;
        font-weight: 700;
        padding-bottom: 8px;
        border-bottom: 2px solid #2166ac;
        margin-bottom: 16px;
    }
    .sec-num {
        display: inline-block;
        background: #2166ac;
        color: #fff;
        font-size: 0.68rem;
        font-weight: 700;
        width: 22px; height: 22px;
        border-radius: 50%;
        text-align: center;
        line-height: 22px;
        margin-right: 8px;
    }

    /* 버튼 */
    .stButton > button {
        background: #2166ac !important;
        color: #fff !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 6px 18px !important;
    }
    .stButton > button:hover { background: #1a4f8a !important; }

    /* 텍스트 */
    p, li, .stMarkdown { color: #333333; }
    label { color: #6c757d !important; font-size: 0.82rem !important; }
    h1,h2,h3 { color: #1a1a1a !important; }

    /* 메트릭 */
    [data-testid="stMetricValue"] { color: #2166ac !important; font-weight: 700 !important; font-size: 1.5rem !important; }
    [data-testid="stMetricLabel"] { color: #6c757d !important; font-size: 0.78rem !important; }

    /* 알림 */
    .stSuccess { background: #d4edda !important; border-left: 3px solid #1a9641 !important; color: #155724 !important; }
    .stInfo    { background: #d1ecf1 !important; border-left: 3px solid #2166ac !important; color: #0c5460 !important; }
    .stWarning { background: #fff3cd !important; border-left: 3px solid #d29922 !important; color: #856404 !important; }

    /* 프로그레스 */
    .stProgress > div > div { background: #2166ac !important; }

    /* 구분선 */
    hr { border-color: #dee2e6 !important; margin: 20px 0 !important; }

    /* 사이드바 */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] .stMarkdown { color: #333333 !important; }

    /* expander */
    .streamlit-expanderHeader { color: #6c757d !important; background: #ffffff !important; }
    .streamlit-expanderContent { background: #f8f9fa !important; }

    /* 파일 업로드 */
    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 1px dashed #adb5bd !important;
        border-radius: 4px !important;
    }
</style>
"""


def run():
    st.set_page_config(
        page_title="강도 예측 AI",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(STYLE, unsafe_allow_html=True)

    # Session State
    for k, v in {
        'dm': None, 'pp': None, 'trainer': None,
        'eda_figs': [], 'train_figs': [], 'predict_figs': [],
        'train_results': None, 'predict_results': None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── 사이드바 ──────────────────────────
    with st.sidebar:
        st.markdown("### 강도 예측 AI")
        st.markdown("<span style='color:#8b949e;font-size:0.82rem'>오스테나이트계 철강 고온 강도</span>", unsafe_allow_html=True)
        st.divider()

        st.markdown("<span style='color:#8b949e;font-size:0.78rem;font-weight:600;letter-spacing:1px'>PIPELINE</span>", unsafe_allow_html=True)
        steps = [
            (st.session_state.dm is not None,               "데이터 로드"),
            (bool(st.session_state.eda_figs),               "EDA 분석"),
            (st.session_state.trainer is not None,          "모델 학습"),
            (st.session_state.predict_results is not None,  "예측 분석"),
        ]
        for done, label in steps:
            c = '#1a9641' if done else '#adb5bd'
            i = '●' if done else '○'
            st.markdown(f"<span style='color:{c};font-size:0.85rem'>{i}&nbsp; {label}</span>", unsafe_allow_html=True)

        st.divider()
        st.markdown("<span style='color:#8b949e;font-size:0.78rem;font-weight:600;letter-spacing:1px'>MODELS</span>", unsafe_allow_html=True)
        for name, color in CHART_COLORS.items():
            short = {'Random Forest':'RF','Gradient Boosting':'GBM','Neural Network':'MLP','BNN':'BNN'}[name]
            st.markdown(
                f"<span style='background:{color}22;color:{color};border:1px solid {color}44;"
                f"padding:2px 8px;border-radius:4px;font-size:0.72rem;font-weight:700'>{short}</span>"
                f"<span style='color:#8b949e;font-size:0.8rem'> {name}</span>",
                unsafe_allow_html=True)

        if st.session_state.trainer is not None:
            st.divider()
            best = st.session_state.trainer.best_model_name()
            r    = st.session_state.trainer.results[best]
            st.markdown("<span style='color:#8b949e;font-size:0.78rem;font-weight:600;letter-spacing:1px'>BEST MODEL</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:{CLR_BLUE};font-weight:700;font-size:0.9rem'>{best}</span>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("PS R²",  f"{r['ps_r2']:.3f}")
            c2.metric("UTS R²", f"{r['uts_r2']:.3f}")

    # ── 헤더 ──────────────────────────────
    st.markdown("""
    <div class='app-header'>
        <h1>오스테나이트계 철강 &nbsp;고온 강도 예측</h1>
        <p>Random Forest &nbsp;·&nbsp; Gradient Boosting &nbsp;·&nbsp; Neural Network (PyTorch) &nbsp;·&nbsp; 데이터 기반 AI 예측</p>
    </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════
    # 01. 데이터 로드
    # ══════════════════════════════════════
    st.markdown("<div class='sec-title'><span class='sec-num'>1</span>데이터 로드</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([5, 1])
    with c1:
        uploaded = st.file_uploader("엑셀 파일 (.xls/.xlsx)", type=['xls','xlsx'], label_visibility="collapsed")
    with c2:
        st.write(""); load_btn = st.button("로드", type="primary", use_container_width=True)

    if uploaded and load_btn:
        with st.spinner("로드 중..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as f:
                f.write(uploaded.read()); tmp = f.name
            dm = DataManager(); dm.load(tmp)
            st.session_state.dm = dm
            for k in ['eda_figs','train_figs','predict_figs','train_results','predict_results']:
                st.session_state[k] = [] if 'figs' in k else None
        st.success("데이터 로드 완료")

    if st.session_state.dm is not None:
        dm = st.session_state.dm
        cols = st.columns(4)
        for col, label, val in zip(cols, ["전체","학습","검증","테스트"],
                                   [len(dm.df_model), len(dm.X_train), len(dm.X_val), len(dm.X_test)]):
            col.metric(label, f"{val:,} 행")
        with st.expander("데이터 통계"):
            st.text(dm.summary())

    st.divider()

    # ══════════════════════════════════════
    # 02. EDA
    # ══════════════════════════════════════
    st.markdown("<div class='sec-title'><span class='sec-num'>2</span>데이터 분석 (EDA)</div>", unsafe_allow_html=True)

    if st.session_state.dm is None:
        st.info("데이터를 먼저 로드해주세요.")
    else:
        c1, c2 = st.columns([5, 1])
        with c1:
            opts = st.multiselect("항목", ['결측치 현황','바이올린 플롯','상관관계 행렬','히스토그램','온도별 강도 분포'],
                                  default=['결측치 현황','상관관계 행렬','온도별 강도 분포'], label_visibility="collapsed")
        with c2:
            st.write(""); eda_btn = st.button("분석", use_container_width=True)

        if eda_btn:
            with st.spinner("분석 중..."):
                st.session_state.eda_figs = _run_eda(st.session_state.dm, opts)

        for title, fig in st.session_state.eda_figs:
            st.markdown(f"<p style='color:#8b949e;font-size:0.82rem;margin:16px 0 6px'>{title}</p>", unsafe_allow_html=True)
            st.pyplot(fig); plt.close(fig)

    st.divider()

    # ══════════════════════════════════════
    # 03. 모델 학습
    # ══════════════════════════════════════
    st.markdown("<div class='sec-title'><span class='sec-num'>3</span>모델 학습</div>", unsafe_allow_html=True)

    if st.session_state.dm is None:
        st.info("데이터를 먼저 로드해주세요.")
    else:
        # 저장된 모델 존재 여부 확인
        model_exists = os.path.exists(f'{OUTPUT_DIR}/models/best_model_ps.pkl')

        c1, c2, c3 = st.columns([4, 1, 1])
        with c1:
            st.markdown("<span style='color:#8b949e;font-size:0.82rem'>RF · GBM · Neural Network · BNN  4개 모델 학습 및 성능 비교</span>", unsafe_allow_html=True)
            if model_exists:
                st.markdown("<span style='color:#3fb950;font-size:0.78rem'>● 저장된 모델 있음 (로드 가능)</span>", unsafe_allow_html=True)
        with c2:
            load_btn = st.button("모델 로드", use_container_width=True, disabled=not model_exists)
        with c3:
            train_btn = st.button("새로 학습", type="primary", use_container_width=True)

        # ── 저장된 모델 로드 ──────────────
        if load_btn and model_exists:
            dm = st.session_state.dm
            with st.spinner("모델 로드 중..."):
                import joblib
                from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

                # 스케일러 로드
                pp = Preprocessor(dm)
                pp.imputer    = joblib.load(f'{OUTPUT_DIR}/scalers/imputer.pkl')
                pp.scaler_x   = joblib.load(f'{OUTPUT_DIR}/scalers/scaler_x.pkl')
                pp.scaler_ps  = joblib.load(f'{OUTPUT_DIR}/scalers/scaler_ps.pkl')
                pp.scaler_uts = joblib.load(f'{OUTPUT_DIR}/scalers/scaler_uts.pkl')

                # X 스케일링
                X_tr  = pp.imputer.transform(dm.X_train)
                X_val = pp.imputer.transform(dm.X_val)
                X_te  = pp.imputer.transform(dm.X_test)
                pp.X_train_s = pp.scaler_x.transform(X_tr)
                pp.X_val_s   = pp.scaler_x.transform(X_val)
                pp.X_test_s  = pp.scaler_x.transform(X_te)
                st.session_state.pp = pp

                # 저장된 best 모델 로드 후 trainer에 세팅
                trainer = Trainer(pp)

                best_ps  = joblib.load(f'{OUTPUT_DIR}/models/best_model_ps.pkl')
                best_uts = joblib.load(f'{OUTPUT_DIR}/models/best_model_uts.pkl')

                # 모델 타입 확인해서 이름 결정
                if isinstance(best_ps, RandomForestRegressor):
                    best_name = 'Random Forest'
                elif isinstance(best_ps, HistGradientBoostingRegressor):
                    best_name = 'Gradient Boosting'
                else:
                    best_name = 'Neural Network'

                # 나머지 모델도 간단히 학습 (RF, GBM은 빠름)
                trainer.fit(progress_callback=None)

                st.session_state.trainer = trainer
                figs, table = _get_train_figs(trainer, dm)
                st.session_state.train_figs    = figs
                st.session_state.train_results = table
                st.session_state.predict_figs  = []
                st.session_state.predict_results = None
            st.success(f"모델 로드 완료! (최우수: {trainer.best_model_name()})")

        # ── 새로 학습 ─────────────────────
        if train_btn:
            dm = st.session_state.dm
            pb = st.progress(0); st_msg = st.empty()

            with st.spinner("전처리 중..."):
                pp = Preprocessor(dm); pp.fit_transform(); pp.save()
                st.session_state.pp = pp

            trainer = Trainer(pp)
            def cb(step, total, msg):
                pb.progress(step/total)
                st_msg.markdown(f"<span style='color:#8b949e;font-size:0.82rem'>{msg}</span>", unsafe_allow_html=True)

            trainer.fit(progress_callback=cb)
            trainer.save_models()
            st.session_state.trainer = trainer
            figs, table = _get_train_figs(trainer, dm)
            st.session_state.train_figs    = figs
            st.session_state.train_results = table
            st.session_state.predict_figs  = []
            st.session_state.predict_results = None
            pb.progress(1.0); st_msg.empty()

        if st.session_state.train_results is not None:
            trainer = st.session_state.trainer
            best    = trainer.best_model_name()
            r       = trainer.results[best]
            st.success(f"최우수 모델: **{best}**")

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("PS MAE",  f"{r['ps_mae']:.2f} MPa")
            c2.metric("PS R²",   f"{r['ps_r2']:.4f}")
            c3.metric("UTS MAE", f"{r['uts_mae']:.2f} MPa")
            c4.metric("UTS R²",  f"{r['uts_r2']:.4f}")

            with st.expander("성능 지표 해석"):
                ps_mae  = r['ps_mae']
                ps_r2   = r['ps_r2']
                ps_rmse = r['ps_rmse']
                uts_r2  = r['uts_r2']
                grade   = "우수" if ps_r2 >= 0.9 else "양호" if ps_r2 >= 0.8 else "다소 낮음"
                uniform = "비교적 균일" if ps_rmse/ps_mae < 1.5 else "일부 구간에서 큰 오차 발생"
                st.caption(
                    f"항복강도 예측의 평균 오차(MAE)는 {ps_mae:.1f} MPa로, "
                    f"실제 강도 범위 대비 {ps_mae/400*100:.1f}% 수준입니다. "
                    f"R²는 {ps_r2:.4f}로 {grade} 수준이며, 전체 강도 변동의 {ps_r2*100:.1f}%를 설명합니다. "
                    f"RMSE/MAE 비율은 {ps_rmse/ps_mae:.2f}로 오차가 {uniform}합니다. "
                    f"4개 모델 중 Proof Stress R²가 가장 높은 {best}이 최우수 모델로 선정되었습니다."
                )

            with st.expander("전체 모델 성능 비교"):
                st.dataframe(st.session_state.train_results, use_container_width=True)
                st.markdown("<span style='color:#8b949e;font-size:0.8rem'>※ RF = Random Forest | GBM = Gradient Boosting | MLP = Neural Network</span>", unsafe_allow_html=True)

        for title, fig in st.session_state.train_figs:
            st.markdown(f"<p style='color:#8b949e;font-size:0.82rem;margin:16px 0 6px'>{title}</p>", unsafe_allow_html=True)
            st.pyplot(fig); plt.close(fig)

    st.divider()

    # ══════════════════════════════════════
    # 04. 예측 분석
    # ══════════════════════════════════════
    st.markdown("<div class='sec-title'><span class='sec-num'>4</span>예측 결과 분석</div>", unsafe_allow_html=True)

    if st.session_state.trainer is None:
        st.info("모델을 먼저 학습해주세요.")
    else:
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown("<span style='color:#8b949e;font-size:0.82rem'>최우수 모델 예측 · 피처 중요도 · 온도별 오차 분석</span>", unsafe_allow_html=True)
        with c2:
            pred_btn = st.button("분석", key="pred", use_container_width=True)

        if pred_btn:
            predictor = Predictor(st.session_state.trainer, st.session_state.pp)
            with st.spinner("분석 중..."):
                result_df, best_name = predictor.predict(inverse_scaling=True)
                figs = _get_predict_figs(predictor, result_df, best_name)
            st.session_state.predict_figs    = figs
            st.session_state.predict_results = (result_df, best_name)

        if st.session_state.predict_results is not None:
            result_df, best_name = st.session_state.predict_results
            st.success(f"최우수 모델: **{best_name}**")
            c1, c2 = st.columns(2)
            c1.metric("Proof Stress MAE", f"{abs(result_df['오차_PS']).mean():.2f} MPa")
            c2.metric("UTS MAE",           f"{abs(result_df['오차_UTS']).mean():.2f} MPa")
            with st.expander("예측 결과 해석"):
                st.caption(
                    "오차는 실제값에서 예측값을 뺀 값입니다. "
                    "양수이면 실제보다 낮게, 음수이면 높게 예측한 것을 의미합니다. "
                    f"현재 항복강도 평균 오차는 {abs(result_df['오차_PS']).mean():.1f} MPa, "
                    f"인장강도 평균 오차는 {abs(result_df['오차_UTS']).mean():.1f} MPa입니다."
                )

            with st.expander("예측 결과 상세"):
                st.dataframe(result_df.round(2), use_container_width=True)

            # ── BNN 불확실성 분석 ──────────────
            st.markdown("---")
            st.markdown("**BNN 예측 신뢰구간 분석**")
            _predictor = Predictor(st.session_state.trainer, st.session_state.pp)
            uncertainty = _predictor.predict_uncertainty()
            if uncertainty[0] is not None:
                mean_ps, std_ps, lo_ps, hi_ps, mean_uts, std_uts, lo_uts, hi_uts = uncertainty

                c1, c2 = st.columns(2)
                c1.metric("PS 평균 불확실성 (σ)", f"±{std_ps.mean():.2f} MPa")
                c2.metric("UTS 평균 불확실성 (σ)", f"±{std_uts.mean():.2f} MPa")

                st.caption(
                    f"BNN(Bayesian Neural Network) MC Dropout 방식으로 100회 샘플링한 결과입니다. "
                    f"항복강도 예측의 95% 신뢰구간은 평균 ±{1.96*std_ps.mean():.1f} MPa이며, "
                    f"인장강도는 ±{1.96*std_uts.mean():.1f} MPa입니다. "
                    f"불확실성이 클수록 해당 조건에서의 예측 신뢰도가 낮음을 의미합니다."
                )

                # 불확실성 시각화
                fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                _fig_style(fig, axes)
                fig.suptitle('BNN 예측 신뢰구간 (95%)', color=CLR_TEXT, fontsize=13, fontweight='bold')

                for ax, mean, lo, hi, actual, title in zip(
                    axes,
                    [mean_ps, mean_uts],
                    [lo_ps,   lo_uts],
                    [hi_ps,   hi_uts],
                    [_predictor.dm.y_ps_test, _predictor.dm.y_uts_test],
                    ['Proof Stress', 'UTS']
                ):
                    idx = np.argsort(actual)
                    ax.fill_between(range(len(idx)), lo[idx], hi[idx],
                                    alpha=0.3, color=CLR_BLUE, label='95% 신뢰구간')
                    ax.plot(range(len(idx)), mean[idx],
                            color=CLR_BLUE, linewidth=1.5, label='BNN 예측')
                    ax.scatter(range(len(idx)), actual[idx],
                               color=CLR_RED, s=8, alpha=0.6, zorder=5, label='실제값')
                    ax.set_title(title, fontsize=11, fontweight='bold')
                    ax.set_xlabel('샘플 인덱스 (실제값 기준 정렬)')
                    ax.set_ylabel('강도 (MPa)')
                    ax.legend(facecolor=BG_MAIN, edgecolor=CLR_SPINE, labelcolor=CLR_TEXT, fontsize=9)

                plt.tight_layout()
                fig.savefig(f'{OUTPUT_DIR}/7_bnn_uncertainty.png', dpi=150, bbox_inches='tight')
                st.pyplot(fig); plt.close(fig)
            else:
                st.info("BNN 모델이 학습되지 않았습니다. 모델을 다시 학습해주세요.")

            csv = result_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button("CSV 다운로드", csv, "predictions.csv", "text/csv")

        for item in st.session_state.predict_figs:
            if len(item) == 3:
                title, fig, insight = item
            else:
                title, fig = item
                insight = None

            st.markdown(f"<p style='color:#8b949e;font-size:0.82rem;margin:16px 0 6px'>{title}</p>", unsafe_allow_html=True)
            st.pyplot(fig); plt.close(fig)

            # 피처 중요도 인사이트 표시
            if insight is not None and title == '피처 중요도':
                top3_ps, top3_uts, top1_imp, top2_imp = insight
                import pandas as pd
                st.caption(
                    f"항복강도에 가장 큰 영향을 미치는 인자는 {top3_ps[0]}, {top3_ps[1]}, {top3_ps[2]} 순이며, "
                    f"인장강도는 {top3_uts[0]}, {top3_uts[1]}, {top3_uts[2]} 순입니다. "
                    "신소재 설계 시 이 인자들을 우선적으로 고려하는 것이 효과적입니다."
                )



# ══════════════════════════════════════════
# EDA
# ══════════════════════════════════════════

def _run_eda(dm, options):
    figs = []

    if '결측치 현황' in options:
        missing = dm.df[ALL_FEATURES + [TARGET_PS, TARGET_UTS]].isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        _fig_style(fig, [ax])
        if len(missing) > 0:
            bars = ax.barh(missing.index, missing.values, color=CLR_BLUE, alpha=0.85, edgecolor='none', height=0.6)
            for bar in bars:
                ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                        f'{int(bar.get_width())}', va='center', color=CLR_TEXT, fontsize=9)
            ax.set_xlabel('결측치 수')
        else:
            ax.text(0.5, 0.5, '결측치 없음', ha='center', va='center', fontsize=14, color=CLR_GREEN, transform=ax.transAxes)
        ax.set_title('결측치 현황', fontweight='bold', fontsize=13)
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/1a_missing.png', dpi=150, bbox_inches='tight')
        figs.append(('결측치 현황', fig))

    if '바이올린 플롯' in options:
        cols = ALL_FEATURES[:15] + [TARGET_PS, TARGET_UTS]
        df_norm = dm.df_model[ALL_FEATURES + [TARGET_PS, TARGET_UTS]].copy()
        df_norm = (df_norm - df_norm.mean()) / (df_norm.std() + 1e-8)
        fig, ax = plt.subplots(figsize=(18, 6))
        _fig_style(fig, [ax])
        vp = ax.violinplot([df_norm[c].dropna().values for c in cols], positions=range(len(cols)), showmedians=True)
        for pc in vp['bodies']:
            pc.set_facecolor(CLR_BLUE); pc.set_alpha(0.5); pc.set_edgecolor(CLR_SPINE)
        vp['cmedians'].set_color(CLR_ORANGE); vp['cmedians'].set_linewidth(2)
        for part in ['cbars','cmins','cmaxes']:
            vp[part].set_color(CLR_SPINE)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
        ax.axhline(0, color=CLR_RED, linestyle='--', alpha=0.4, linewidth=1)
        ax.set_title('정규화된 컬럼 분포 (바이올린 플롯)', fontweight='bold', fontsize=13)
        ax.set_ylabel('정규화된 값')
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/1b_violin.png', dpi=150, bbox_inches='tight')
        figs.append(('바이올린 플롯', fig))

    if '상관관계 행렬' in options:
        cols = ['Cr','Ni','Mo','Mn','Si','N','C','Cu','Temperature (K)', TARGET_PS, TARGET_UTS]
        corr = dm.df[cols].corr()
        fig, ax = plt.subplots(figsize=(11, 9))
        _fig_style(fig, [ax])
        im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
        cbar.ax.tick_params(colors=CLR_TEXT, labelsize=9)
        cbar.outline.set_edgecolor(CLR_SPINE)
        ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(cols, fontsize=9)
        ax.set_title('상관관계 행렬', fontweight='bold', fontsize=13)
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = corr.values[i,j]
                # 항상 흰색 텍스트 + 반투명 검정 배경으로 가독성 확보
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold' if abs(v)>0.5 else 'normal',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.35, edgecolor='none'))
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/1c_corr.png', dpi=150, bbox_inches='tight')
        figs.append(('상관관계 행렬', fig))

    if '히스토그램' in options:
        cols = ['Cr','Ni','Mo','Mn','Si','N','C','Cu','Temperature (K)', TARGET_PS, TARGET_UTS]
        pal  = [CLR_BLUE,'#79c0ff','#388bfd',CLR_ORANGE,'#ffa657',
                CLR_GREEN,'#56d364',CLR_RED,'#d2a8ff','#a5d6ff','#7ee787']
        n_rows = (len(cols)+3)//4
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4*n_rows))
        _fig_style(fig, axes.flatten())
        fig.suptitle('컬럼별 데이터 분포', color=CLR_TEXT, fontsize=13, fontweight='bold', y=1.01)
        for i, col in enumerate(cols):
            ax   = axes.flatten()[i]
            vals = dm.df_model[col].dropna()
            ax.hist(vals, bins=30, color=pal[i], alpha=0.85, edgecolor='none')
            ax.axvline(vals.mean(),   color=CLR_RED,    linestyle='--', linewidth=1.5, alpha=0.9)
            ax.axvline(vals.median(), color=CLR_ORANGE, linestyle='--', linewidth=1.5, alpha=0.9)
            ax.set_title(col, fontsize=10, fontweight='bold')
        for j in range(len(cols), len(axes.flatten())):
            axes.flatten()[j].set_visible(False)
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/1d_histogram.png', dpi=150, bbox_inches='tight')
        figs.append(('히스토그램', fig))

    if '온도별 강도 분포' in options:
        temps = sorted(dm.df['Temperature (K)'].dropna().unique())
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        _fig_style(fig, axes)
        axes[0].boxplot(
            [dm.df[dm.df['Temperature (K)']==t][TARGET_PS].dropna().values for t in temps],
            positions=range(len(temps)), widths=0.55, patch_artist=True,
            boxprops=dict(facecolor=f'{CLR_BLUE}55', edgecolor=CLR_BLUE, linewidth=1.2),
            medianprops=dict(color=CLR_ORANGE, linewidth=2.5),
            whiskerprops=dict(color=CLR_SPINE, linewidth=1.2),
            capprops=dict(color=CLR_SPINE, linewidth=1.2),
            flierprops=dict(marker='o', markerfacecolor=CLR_TEXT, markersize=3, alpha=0.5))
        axes[0].set_xticks(range(len(temps)))
        axes[0].set_xticklabels([str(int(t)) for t in temps], rotation=45, fontsize=8)
        axes[0].set_title('온도별 항복강도 분포', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('0.2% Proof Stress (MPa)')

        mean_ps  = dm.df.groupby('Temperature (K)')[TARGET_PS].mean()
        mean_uts = dm.df.groupby('Temperature (K)')[TARGET_UTS].mean()
        axes[1].plot(mean_ps.index,  mean_ps.values,  'o-', color=CLR_BLUE,   linewidth=2.5, markersize=6, label='Proof Stress')
        axes[1].plot(mean_uts.index, mean_uts.values, 's-', color=CLR_ORANGE, linewidth=2.5, markersize=6, label='UTS')
        axes[1].set_title('온도별 평균 강도 추이', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('강도 (MPa)')
        axes[1].legend(facecolor=BG_MAIN, edgecolor=CLR_SPINE, labelcolor=CLR_TEXT, fontsize=10)
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/1e_temp.png', dpi=150, bbox_inches='tight')
        figs.append(('온도별 강도 분포', fig))

    return figs


# ══════════════════════════════════════════
# 학습 결과
# ══════════════════════════════════════════

def _get_train_figs(trainer, dm):
    import pandas as pd
    figs = []

    rows = []
    for name, r in trainer.results.items():
        rows.append({'모델': name,
                     'PS MAE': f"{r['ps_mae']:.2f}", 'PS R²': f"{r['ps_r2']:.4f}",
                     'UTS MAE': f"{r['uts_mae']:.2f}", 'UTS R²': f"{r['uts_r2']:.4f}"})
    table = pd.DataFrame(rows)

    # Loss 커브
    tt = trainer.torch_trainers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _fig_style(fig, axes)
    fig.suptitle('Neural Network — Loss 커브', color=CLR_TEXT, fontsize=13, fontweight='bold')
    for ax, key, title in zip(axes, ['ps','uts'], ['Proof Stress','UTS']):
        ax.plot(tt[key].train_losses, color=CLR_BLUE,   linewidth=1.2, alpha=0.7, label='Train loss')
        ax.plot(tt[key].val_losses,   color=CLR_ORANGE, linewidth=2.0, label='Val loss')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MAE')
        ax.legend(facecolor=BG_MAIN, edgecolor=CLR_SPINE, labelcolor=CLR_TEXT, fontsize=9)
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/2_loss.png', dpi=150, bbox_inches='tight')
    figs.append(('Neural Network Loss 커브', fig))

    # 예측 vs 실제
    model_names = list(trainer.results.keys())
    n_models    = len(model_names)
    fig, axes   = plt.subplots(n_models, 4, figsize=(22, 5.5*n_models))
    if n_models == 1: axes = axes.reshape(1, -1)
    _fig_style(fig, axes.flatten())
    fig.suptitle('모델별 예측 성능', color=CLR_TEXT, fontsize=14, fontweight='bold')

    for row, name in enumerate(model_names):
        res   = trainer.results[name]
        color = CHART_COLORS[name]
        for col_offset, (y_true, y_pred, subtitle) in enumerate([
            (dm.y_ps_test,  res['pred_ps'],  f'Proof Stress   R² = {res["ps_r2"]:.3f}'),
            (dm.y_uts_test, res['pred_uts'], f'UTS   R² = {res["uts_r2"]:.3f}'),
        ]):
            # 예측 vs 실제
            ax = axes[row, col_offset]
            lo, hi = min(y_true.min(), y_pred.min())-10, max(y_true.max(), y_pred.max())+10
            ax.scatter(y_true, y_pred, alpha=0.5, s=12, color=color, edgecolors='none')
            ax.plot([lo,hi],[lo,hi], color=CLR_RED, linestyle='--', linewidth=1.5, alpha=0.8)
            ax.set_title(f'{name}\n{subtitle}', fontsize=10, fontweight='bold')
            ax.set_xlabel('실제 (MPa)', fontsize=9); ax.set_ylabel('예측 (MPa)', fontsize=9)
            ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)

            # 잔차
            ax = axes[row, col_offset+2]
            residuals = y_true - y_pred
            ax.scatter(y_pred, residuals, alpha=0.5, s=12, color=color, edgecolors='none')
            ax.axhline(0, color=CLR_RED, linestyle='--', linewidth=1.5, alpha=0.8)
            ax.set_title(f'{name} — 잔차', fontsize=10, fontweight='bold')
            ax.set_xlabel('예측값 (MPa)', fontsize=9); ax.set_ylabel('잔차 (MPa)', fontsize=9)

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/3_performance.png', dpi=150, bbox_inches='tight')
    figs.append(('예측 성능 비교', fig))

    # 지표 비교
    short      = {'Random Forest':'RF','Gradient Boosting':'GBM','Neural Network':'MLP','BNN':'BNN'}
    bar_colors = [CHART_COLORS[n] for n in model_names]
    fig, axes  = plt.subplots(1, 3, figsize=(16, 5))
    _fig_style(fig, axes)
    fig.suptitle('모델 성능 지표 비교', color=CLR_TEXT, fontsize=13, fontweight='bold')
    for ax, (title, pk, uk) in zip(axes, [
        ('MAE (MPa)',  'ps_mae',  'uts_mae'),
        ('RMSE (MPa)', 'ps_rmse', 'uts_rmse'),
        ('R²',         'ps_r2',   'uts_r2'),
    ]):
        pv = [trainer.results[n][pk] for n in model_names]
        uv = [trainer.results[n][uk] for n in model_names]
        x, w = np.arange(len(model_names)), 0.35
        b1 = ax.bar(x-w/2, pv, w, label='Proof Stress', color=[c+'66' for c in bar_colors], edgecolor=[c for c in bar_colors], linewidth=1.2)
        b2 = ax.bar(x+w/2, uv, w, label='UTS',          color=bar_colors,                  edgecolor='none', alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels([short.get(n,n) for n in model_names], fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(facecolor=BG_MAIN, edgecolor=CLR_SPINE, labelcolor=CLR_TEXT, fontsize=9)
        for bar in [*b1, *b2]:
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.003*abs(bar.get_height()),
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8, color=CLR_TEXT)
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/4_metrics.png', dpi=150, bbox_inches='tight')
    figs.append(('성능 지표 비교', fig))

    return figs, table


# ══════════════════════════════════════════
# 예측 결과
# ══════════════════════════════════════════

def _get_predict_figs(predictor, result_df, best_name):
    figs = []

    imp_ps, imp_uts = predictor.feature_importance()
    if imp_ps is not None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        _fig_style(fig, axes)
        fig.suptitle('피처 중요도 — Random Forest', color=CLR_TEXT, fontsize=13, fontweight='bold')
        for ax, imp, title in zip(axes, [imp_ps, imp_uts], ['Proof Stress', 'UTS']):
            sorted_idx = np.argsort(imp)
            n = len(sorted_idx)
            colors_bar = [CLR_BLUE if i >= n-5 else f'{CLR_BLUE}55' for i in range(n)]
            bars = ax.barh(np.array(FEATURE_DISPLAY_NAMES)[sorted_idx], imp[sorted_idx],
                           color=colors_bar, edgecolor='none', height=0.7)
            # 상위 5개 테두리 강조
            for i in range(n-5, n):
                bars[i].set_edgecolor(CLR_ORANGE)
                bars[i].set_linewidth(1.5)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Feature Importance')
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/5_feature_importance.png', dpi=150, bbox_inches='tight')

        # 상위 3개 피처 자동 추출해서 해석 텍스트 생성
        sorted_ps  = np.argsort(imp_ps)[::-1]
        sorted_uts = np.argsort(imp_uts)[::-1]
        top3_ps    = [FEATURE_DISPLAY_NAMES[i] for i in sorted_ps[:3]]
        top3_uts   = [FEATURE_DISPLAY_NAMES[i] for i in sorted_uts[:3]]

        insight = (top3_ps, top3_uts, imp_ps[sorted_ps[0]], imp_ps[sorted_ps[1]])
        figs.append(('피처 중요도', fig, insight))

    temp_err = predictor.temp_error()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    _fig_style(fig, axes)
    fig.suptitle(f'온도별 예측 오차 — {best_name}', color=CLR_TEXT, fontsize=13, fontweight='bold')
    for ax, key, title in zip(axes, ['ps','uts'], ['Proof Stress','UTS']):
        mae_dict = temp_err[key]
        ts, maes = sorted(mae_dict), [mae_dict[t] for t in sorted(mae_dict)]
        bars = ax.bar(range(len(ts)), maes, color=CLR_BLUE, alpha=0.85, edgecolor='none', width=0.6)
        # 평균 이상 강조
        avg = np.mean(maes)
        for i, (bar, mae) in enumerate(zip(bars, maes)):
            if mae > avg:
                bar.set_color(CLR_ORANGE)
        ax.set_xticks(range(len(ts)))
        ax.set_xticklabels([str(int(t)) for t in ts], rotation=45, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('MAE (MPa)')
        ax.axhline(avg, color=CLR_RED, linestyle='--', linewidth=1.8,
                   label=f'평균  {avg:.1f} MPa')
        ax.legend(facecolor=BG_MAIN, edgecolor=CLR_SPINE, labelcolor=CLR_TEXT, fontsize=9)
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/6_temp_error.png', dpi=150, bbox_inches='tight')
    figs.append(('온도별 오차 분석', fig))

    return figs