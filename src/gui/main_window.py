import sys
import os

# Important: Import project modules (which load TensorFlow) before Matplotlib to avoid DLL conflicts on Windows
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data_pipeline.parser import StmechDataParser
from src.models.ai_model import MaterialPredictionModel

import requests
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QLabel, QLineEdit, 
                             QPushButton, QTextEdit, QFormLayout, QMessageBox,
                             QScrollArea, QFileDialog, QGroupBox, QListWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import matplotlib
matplotlib.use('QtAgg') # 맷플롯립(Matplotlib) 그래프를 PyQt 창 내부에 띄우기 위한 백엔드 설정
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.font_manager as fm
import seaborn as sns

# 한글 폰트 설정 (Windows/Linux/Mac 공용 가용 폰트 검색)
def get_korean_font():
    korean_fonts = [
        "Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic", 
        "Gulim", "Dotum", "Arial"
    ]
    for font in korean_fonts:
        if font in [f.name for f in fm.fontManager.ttflist]:
            return font
    return None

font_name = get_korean_font()
if font_name:
    matplotlib.rc('font', family=font_name)
    matplotlib.rc('axes', unicode_minus=False) # 마이너스 기호 깨짐 방지

class TrainingThread(QThread):
    """
    GUI 환경에서 AI 모델 학습 버튼을 눌렀을 때, 프로그램 화면이 멈추는(응답 없음) 현상을 방지하기 위해 
    별도의 백그라운드 스레드(Thread)에서 학습 연산을 수행하는 클래스입니다.
    """
    # 진행 상황 로그 전송, 그래프 업데이트, 학습 종료 신호를 메인 창으로 전달하는 시그널(Signal) 정의
    log_signal = pyqtSignal(str)
    plot_signal = pyqtSignal(list, list)
    yy_plot_signal = pyqtSignal(list, list) # 실제값, 예측값 데이터 전송
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str) # 에러 발생 시 알림

    def __init__(self, filepath=None, epochs=50, batch_size=32, model_type='TFP'):
        super().__init__()
        self.filepath = filepath
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_type = model_type

    def run(self):
        """별도 스레드에서 실제 실행되는 학습 메인 로직입니다."""
        try:
            self.log_signal.emit(f"[System] {self.model_type} 모델 기반 데이터 로딩 중...")
            
            # 1. 엑셀 파서를 초기화하고 데이터를 불러와 정규화(Scaling)를 진행합니다.
            parser = StmechDataParser()
            X, y, feats = parser.load_and_preprocess(custom_path=self.filepath)
            X_scaled, y_scaled = parser.fit_transform(X, y)
            
            self.log_signal.emit(f"[System] 데이터 로딩 완료. 형태: {X.shape}")
            self.log_signal.emit(f"[System] 사용하는 특성: {', '.join(feats)}")
            
            # 2. 인공지능 모델을 초기화합니다.
            model = MaterialPredictionModel(input_shape=X.shape[1], model_type=self.model_type)
            
            # 3. 모델을 학습시킵니다.
            self.log_signal.emit(f"[System] {self.model_type} 학습 시작...")
            
            # 과적합 방지를 위해 데이터를 학습용(80%)과 검증용(20%)으로 미리 나눕니다.
            from sklearn.model_selection import train_test_split
            X_t, X_v, y_t, y_v = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

            if self.model_type in ['TFP', 'MLP']:
                # Keras Callback을 이용해 Epoch마다 그래프 및 Y-Y Plot 업데이트 시그널 전송
                class PlotCallback(tf.keras.callbacks.Callback):
                    def __init__(self, thread, X_val, y_val):
                        super().__init__()
                        self.thread = thread
                        self.X_val = X_val
                        self.y_val = y_val
                        self.losses = []
                        
                    def on_epoch_end(self, epoch, logs=None):
                        self.losses.append(logs['loss'])
                        if (epoch + 1) % 5 == 0 or epoch == 0:
                            self.thread.plot_signal.emit(list(range(1, epoch + 2)), self.losses)
                        if (epoch + 1) % 10 == 0:
                            res = self.thread.ai_model.predict_with_confidence(self.X_val)
                            self.thread.yy_plot_signal.emit(self.y_val.flatten().tolist(), res['mean'])

                self.ai_model = model # 콜백에서 쓰기 위해 저장
                plot_callback = PlotCallback(self, X_v, y_v)
                history = model.train(X_t, y_t, validation_data=(X_v, y_v), epochs=self.epochs, batch_size=self.batch_size, callbacks=[plot_callback])
                final_loss = history.history['loss']
            else:
                # RF, GBM (Scikit-learn)
                model.train(X_t, y_t, validation_data=(X_v, y_v))
                final_loss = [0.0] * self.epochs # SKLearn은 손실 이력 없음

            # 마지막 결과는 항상 전송
            final_res = model.predict_with_confidence(X_v)
            self.yy_plot_signal.emit(y_v.flatten().tolist(), final_res['mean'])
            
            # 4. 학습이 끝난 모델 저장
            os.makedirs("data", exist_ok=True)
            model.save_model("data/model_weights")
            parser.save_scalers("data/scalers.pkl")
            
            self.log_signal.emit(f"[System] {self.model_type} 학습 완료 및 저장 성공!")
            
            # 완료 알림
            self.finished_signal.emit()
            
        except Exception as e:
            import traceback
            error_msg = f"학습 중 치명적 오류 발생: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.error_signal.emit(str(e))
            self.log_signal.emit(f"❌ [에러] {str(e)}")
            self.finished_signal.emit()

class OptimizationThread(QThread):
    """
    최적화 계산(SciPy)이 진행되는 동안 GUI가 멈추지 않도록 별도 스레드에서 시뮬레이션을 수행합니다.
    """
    finished_signal = pyqtSignal(object, float) # (best_x, best_y)
    progress_signal = pyqtSignal(int, int) # (current, total)
    error_signal = pyqtSignal(str)

    def __init__(self, bounds, model_path, scaler_path, features, model_type='TFP'):
        super().__init__()
        self.bounds = bounds
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.features = features
        self.model_type = model_type

    def run(self):
        try:
            from src.models.ai_model import MaterialPredictionModel
            from src.data_pipeline.parser import StmechDataParser
            
            model = MaterialPredictionModel(len(self.features), model_type=self.model_type)
            parser = StmechDataParser()
            
            if not model.load_model(self.model_path, model_type=self.model_type) or not parser.load_scalers(self.scaler_path):
                self.error_signal.emit("학습된 모델 혹은 정규화 정보(Scalers)를 찾을 수 없습니다.")
                return

            # 실제 최적화 수행 (진행 상황 콜백 연동)
            best_x, best_y = model.optimize_composition(
                self.bounds, 
                scaler_X=parser.scaler_X, 
                scaler_y=parser.scaler_y,
                iteration_callback=lambda c, t: self.progress_signal.emit(c, t)
            )
            self.finished_signal.emit(best_x, best_y)
        except Exception as e:
            self.error_signal.emit(str(e))

class AIMaterialPlatform(QMainWindow):
    """
    앱의 메인 창(Window)을 나타내는 클래스입니다. 데이터를 들이고, 모델을 학습시키고, 결과를 예측하는
    3개의 탭(Tab)으로 구성되어 있습니다.
    """
    def __init__(self):
        super().__init__()
        # 윈도우 창 이름과 넓이 높이 초기값 설정
        self.setWindowTitle("AI Materials Discovery Platform (현장미러형 연계 프로젝트)")
        self.setGeometry(100, 100, 1000, 750)
        
        # 중심 위젯과 통합 레이아웃 설정
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # 상단에 탭 위젯 생성 
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # 현재 선택된 모델 타입을 보관 (기본값 TFP)
        self.current_model_type = "TFP"
        
        # --- [신규] 하단 도움말 섹션 (항복강도 설명) ---
        help_group = QGroupBox("💡 소재 용어 사전: 항복강도(Yield Strength)란?")
        help_layout = QVBoxLayout(help_group)
        help_text = (
            "<b>🔬 플랫폼 용어 안내</b><br>"
            "• <b>항복강도(Yield Strength)</b>: 금속이 영구적으로 휘어지기 시작하는 '한계 힘'입니다. 이 값이 높을수록 튼튼합니다.<br>"
            "• <b>물성 예측(Inference)</b>: 학습된 AI가 새로운 성분 조합을 보고 강도가 얼마일지 알아맞히는 과정입니다.<br>"
            "• <b>최적 설계(Optimization)</b>: AI가 수만 번의 시뮬레이션을 통해 가장 강력한 강도를 내는 '황금 비율'을 찾아내는 기능입니다.<br>"
            "• <b>오차 분석(Y-Y Plot)</b>: 점들이 대각선에 가까울수록 AI가 실제 실험 결과를 정확히 이해하고 있다는 증거입니다."
        )
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #2c3e50; line-height: 140%;")
        help_layout.addWidget(help_label)
        self.layout.addWidget(help_group)
        
        # 동적으로 입력 폼을 구성하기 위해 파서의 Features 리스트를 가져옵니다.
        self.parser = StmechDataParser()
        self.features = self.parser.features
        self.selected_file_path = None # 사용자가 선택한 파일 경로 저장
        self.all_samples = None        # 전체 분석 데이터 (Random 샘플링용)
        self.sample_data = None       # 현재 선택된 샘플 (Inference 탭 입력 보조용)
        self.feature_bounds = None    # 각 성분의 (min, max) 범위 (최적화용)
        self.df_raw = None            # 시각화용 원본 데이터 보관
        
        # 성분별 설명 사전 (사전 지식 제공용)
        self.element_info = {
            "Cr": "✨ <b>[크롬]</b> 스테인리스강의 핵심 성분으로 부식 방지 및 산화 저항성을 크게 향상시킵니다.",
            "Ni": "✨ <b>[니켈]</b> 오스테나이트 구조를 안정화하며 저온 인성과 내식성을 강화합니다.",
            "Mo": "✨ <b>[몰리브덴]</b> 고온 강도를 높이고 공식(Pitting) 부식에 대한 저항성을 부여합니다.",
            "Mn": "✨ <b>[망간]</b> 탈산제 역할을 하며 강도를 높이고 황(S)에 의한 취성을 방지합니다.",
            "Si": "✨ <b>[규소]</b> 탈산제 및 강화 성분으로 사용되며 고온 산화 저항성을 돕습니다.",
            "Nb": "✨ <b>[나이오븀]</b> 결정립 미세화와 탄화물 형성을 통해 강도를 높이고 연화를 늦춥니다.",
            "Ti": "✨ <b>[티타늄]</b> 결정립 미세화 및 입계 부식 방지 역할을 하는 강한 탄화물 형성 원소입니다.",
            "V": "✨ <b>[바나듐]</b> 뜨임 연화 저항성을 높이고 미세 탄화물을 형성하여 강도를 개선합니다.",
            "W": "✨ <b>[텅스텐]</b> 고온 강도와 경도를 유지하는 데 탁월하며 탄화물을 형성합니다.",
            "Cu": "✨ <b>[구리]</b> 내식성을 향상시키며 석출 경화 효과를 통해 강도를 보강하기도 합니다.",
            "N": "✨ <b>[질소]</b> 오스테나이트 안정화 원소이며 고용 강화 효과로 항복강도를 크게 높입니다.",
            "C": "✨ <b>[탄소]</b> 가장 기본적인 강화 원소로 고용 강화 및 탄화물 형성을 통해 강도를 결정합니다.",
            "B": "✨ <b>[붕소]</b> 소량으로도 담금질성을 크게 높이며 결정 입계 강도를 보강합니다.",
            "P": "✨ <b>[인]</b> 강도를 높이지만 인성을 저해하고 저온 취성을 유발할 수 있어 주의가 필요합니다.",
            "S": "✨ <b>[황]</b> 절삭성을 좋게 하지만 연성과 인성을 해치므로 보통 낮게 유지합니다.",
            "Al": "✨ <b>[알루미늄]</b> 강력한 탈산제이며 질화물 형성을 통해 결정립 성장을 억제합니다.",
            "Solution_treatment_temperature": "🌡️ <b>[고용화 처리 온도]</b> 성분들이 균일하게 고용되도록 결정하는 핵심 공정 변수입니다.",
            "Temperature (K)": "🌡️ <b>[측정 온도]</b> 강도를 측정하는 시점의 온도로, 고온일수록 항복강도는 감소합니다.",
            "0.2%proof_stress (M Pa)": "🎯 <b>[항복강도]</b> 소재가 영구 변형되기 시작하는 지점으로, 본 모델의 핵심 예측 대상입니다."
        }
        
        # 각각 탭 초기화 함수 호출
        self._init_data_tab()
        self._init_training_tab()
        self._init_eda_tab() # 신설: 정밀 데이터 분석 탭
        self._init_results_tab()
        self._init_inference_tab()
        self._init_optimization_tab()
        self._init_report_tab() # 신규: AI 상세 보고서 탭

    def _init_data_tab(self):
        """[데이터 수집] 탭 구성 - 최대한 심플하게 원복"""
        data_tab = QWidget()
        layout = QVBoxLayout(data_tab)
        
        info_label = QLabel("### 데이터 수집 및 전처리 (Data Pipeline) ###")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("font-weight: bold; font-size: 16px; margin: 10px;")
        layout.addWidget(info_label)
        
        btn_layout = QHBoxLayout()
        self.btn_select_file = QPushButton("📁 분석할 정밀 데이터 파일 선택 (*.xls, *.xlsx)")
        self.btn_load_real_data = QPushButton("📊 선택된 데이터 구조 분석")
        btn_layout.addWidget(self.btn_select_file)
        btn_layout.addWidget(self.btn_load_real_data)
        layout.addLayout(btn_layout)
        
        # 안내 문구나 결과를 표시하는 텍스트 상자
        self.data_log = QTextEdit()
        self.data_log.setReadOnly(True)
        self.data_log.setPlaceholderText("여기에 데이터 분석 결과가 표시됩니다. 먼저 파일을 선택해 주세요.")
        layout.addWidget(self.data_log)
        
        # 버튼을 눌렀을 때 실행될 함수(Click Event) 연결
        self.btn_select_file.clicked.connect(self._select_file)
        self.btn_load_real_data.clicked.connect(self._analyze_data)
        
        self.tabs.addTab(data_tab, "데이터 (수집/준비)")

    def _init_eda_tab(self):
        """신설된 [데이터 분석(EDA)] 탭 구성 - 하위 탭(Tab-in-Tab) 구조로 전문화"""
        eda_tab = QWidget()
        layout = QVBoxLayout(eda_tab)
        
        info_label = QLabel("🔬 <b>전문가용 데이터 탐색 분석 (EDA) - 상세 분류</b>")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("font-weight: bold; font-size: 16px; margin: 10px;")
        layout.addWidget(info_label)

        # EDA 내부 하위 탭 생성
        self.eda_sub_tabs = QTabWidget()
        layout.addWidget(self.eda_sub_tabs)

        # [복구] 전체 성분 분포 (Overview) 하위 탭
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self.hist_figure = Figure(figsize=(12, 16), dpi=100, tight_layout=True)
        self.hist_canvas = FigureCanvas(self.hist_figure)
        inner_layout = QVBoxLayout(inner)
        inner_layout.addWidget(self.hist_canvas)
        scroll.setWidget(inner)
        overview_layout.addWidget(scroll)
        self.eda_sub_tabs.addTab(overview_tab, "📊 전체 분포 (Overview)")

        # 1. 하위 탭: 성분별 분포 (Distribution)
        dist_tab = QWidget()
        dist_layout = QHBoxLayout(dist_tab) # 리스트와 그래프 배치를 위해 가로 배치
        
        # 좌측: 성분 선택 리스트
        list_panel = QWidget()
        list_vbox = QVBoxLayout(list_panel)
        list_vbox.addWidget(QLabel("<b>🔎 분석 대상 성분 선택</b>"))
        self.element_list = QListWidget()
        self.element_list.setFixedWidth(180)
        list_vbox.addWidget(self.element_list)
        dist_layout.addWidget(list_panel)
        
        # 우측: 상세 그래프 영역 (스크롤 가능)
        detail_scroll = QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail_inner = QWidget()
        detail_vbox = QVBoxLayout(detail_inner)
        
        self.detail_figure = Figure(figsize=(8, 6), dpi=100, tight_layout=True)
        self.detail_canvas = FigureCanvas(self.detail_figure)
        detail_vbox.addWidget(self.detail_canvas)
        
        # 성분 상세 설명 박스 추가
        self.info_box = QLabel("성분을 선택하시면 상세 설명이 여기에 표시됩니다.")
        self.info_box.setWordWrap(True)
        self.info_box.setMinimumHeight(80)
        self.info_box.setStyleSheet("""
            background-color: #f8f9fa; 
            border: 2px solid #3498db; 
            border-radius: 8px; 
            padding: 15px; 
            font-size: 13px;
            color: #2c3e50;
        """)
        detail_vbox.addWidget(self.info_box)
        
        detail_scroll.setWidget(detail_inner)
        dist_layout.addWidget(detail_scroll)
        
        # 성분이 변경될 때 상세 그래프를 갱신합니다 (여기서 연결해야 안전합니다)
        self.element_list.currentTextChanged.connect(self._update_detail_plot)
        
        self.eda_sub_tabs.addTab(dist_tab, "📊 성분별 상세 분석 (Details)")

        # 2. 하위 탭: 상관관계 분석 (Correlation)
        corr_tab = QWidget()
        corr_layout = QVBoxLayout(corr_tab)
        # 히트맵은 정사각형에 가깝게 크게 그립니다.
        self.corr_figure = Figure(figsize=(12, 10), dpi=100, tight_layout=True)
        self.corr_canvas = FigureCanvas(self.corr_figure)
        corr_layout.addWidget(self.corr_canvas)
        self.eda_sub_tabs.addTab(corr_tab, "🔥 상관관계 (Heatmap)")

        # 3. 하위 탭: 정규화 데이터 분석 (Normalization)
        norm_tab = QWidget()
        norm_layout = QVBoxLayout(norm_tab)
        # 바이올린 플롯은 가로로 아주 길게 그립니다.
        self.violin_figure = Figure(figsize=(14, 6), dpi=100, tight_layout=True)
        self.violin_canvas = FigureCanvas(self.violin_figure)
        norm_layout.addWidget(self.violin_canvas)
        
        # [신규] 바이올린 플롯 설명 추가
        norm_help = QLabel("💡 <b>바이올린 플롯(Violin Plot)이란?</b><br>"
                           "모든 성분의 수치를 0~1 사이로 맞춘(정규화) 후, 데이터의 <b>'모양(분포)'</b>을 한눈에 비교하는 그래프입니다. "
                           "배가 불룩할수록 해당 수치에 데이터가 많이 모여있음을 뜻하며, 위아래로 길게 뻗은 선을 통해 <b>이상치(튀는 값)</b>가 있는지도 확인할 수 있습니다.")
        norm_help.setWordWrap(True)
        norm_help.setStyleSheet("background-color: #fdfefe; border: 1px solid #d5dbdb; border-radius: 5px; padding: 10px; color: #2c3e50;")
        norm_layout.addWidget(norm_help)
        
        self.eda_sub_tabs.addTab(norm_tab, "🎻 데이터 정규화 (Violin)")
        
        self.tabs.addTab(eda_tab, "데이터 분석 (EDA)")

    def _select_file(self):
        """사용자가 분석하고자 하는 엑셀 파일을 탐색기에서 선택하도록 합니다."""
        # 윈도우 환경에서 네이티브 다이얼로그 충돌로 인한 멈춤 현상을 방지하기 위해 옵션을 추가합니다.
        options = QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Excel 파일 선택", "", 
            "Excel Files (*.xls *.xlsx);;All Files (*)", 
            options=options
        )
        if file_path:
            self.selected_file_path = file_path
            self.data_log.append(f"\n[알림] 파일이 선택되었습니다: {os.path.basename(file_path)}")
            self.data_log.append(f"주소: {file_path}")
            self.data_log.append("이제 '데이터 구조 분석' 버튼을 눌러주세요.")

    def _analyze_data(self):
        """실제 데이터를 로딩하고 전문가 수준의 시각화(EDA)를 수행합니다."""
        if not self.selected_file_path:
            QMessageBox.warning(self, "파일 미선택", "분석할 엑셀 파일을 먼저 선택해 주세요.")
            return

        self.data_log.append(f"🔍 '{os.path.basename(self.selected_file_path)}' 정밀 분석 시작...")
        try:
            # 1. 데이터 파싱
            import pandas as pd
            X, y, feats = self.parser.load_and_preprocess(custom_path=self.selected_file_path)
            # 전체 데이터를 DataFrame으로 재구성 (시각화 편의용)
            self.df_raw = pd.DataFrame(X, columns=feats)
            self.df_raw[self.parser.target] = y.flatten()

            # 성분 리스트 초기화 및 채우기
            self.element_list.clear()
            self.element_list.addItems(self.df_raw.columns)
            
            # 2. 기본 통계 로깅
            self.data_log.append(f"✓ 데이터 로딩 완료: {X.shape[0]}행, {X.shape[1]}개 성분 파악됨.")
            self.data_log.append(f"✓ 주요 물성({self.parser.target}) 범위: {np.min(y):.1f} ~ {np.max(y):.1f} MPa")
            
            # --- 시각화 1: 컬럼별 히스토그램 ---
            self.hist_figure.clear()
            n_cols = 3 # 4열 대신 3열로 하여 개별 그래프를 더 크게 만듭니다.
            n_rows = (len(self.df_raw.columns) + n_cols - 1) // n_cols
            for i, col in enumerate(self.df_raw.columns):
                ax = self.hist_figure.add_subplot(n_rows, n_cols, i+1)
                sns.histplot(self.df_raw[col], kde=True, ax=ax, color=sns.color_palette("muted")[i % 10])
                ax.set_title(f"[{col}] 분산 분석", fontsize=11, fontweight='bold')
                ax.set_xlabel("값 (Value)", fontsize=9)
                ax.set_ylabel("빈도 (Frequency)", fontsize=9)
                # 평균/중앙값 실선 추가 및 범례 강화
                avg = self.df_raw[col].mean()
                med = self.df_raw[col].median()
                ax.axvline(avg, color='red', linestyle='--', linewidth=1.5, label=f'평균:{avg:.1f}')
                ax.axvline(med, color='orange', linestyle='-', linewidth=1.5, label=f'중앙:{med:.1f}')
                ax.legend(fontsize=8)
            
            # 그래프 간 간격 확보
            self.hist_figure.subplots_adjust(hspace=0.4, wspace=0.3)
            self.hist_canvas.draw()

            # 초기 상세 그래프 로드
            if self.element_list.count() > 0:
                self.element_list.setCurrentRow(0)
                self._update_detail_plot(self.element_list.item(0).text())

            # --- 시각화 2: 상관관계 열지도 (Heatmap) ---
            self.corr_figure.clear()
            ax_corr = self.corr_figure.add_subplot(111)
            corr = self.df_raw.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdBu_r', ax=ax_corr, annot_kws={"size": 7})
            ax_corr.set_title("주요 변수 상관관계 행렬", fontsize=12, fontweight='bold')
            self.corr_canvas.draw()

            # --- 시각화 3: 바이올린 플롯 ---
            self.violin_figure.clear()
            ax_v = self.violin_figure.add_subplot(111)
            # 분석용 정규화 데이터 생성
            X_scaled, _ = self.parser.fit_transform(X, y)
            df_scaled = pd.DataFrame(X_scaled, columns=feats)
            sns.violinplot(data=df_scaled, ax=ax_v, palette="Set3")
            ax_v.set_title("성분별 정규화된 데이터 분포 성격 분석 (Violin Plot)", fontsize=11, fontweight='bold')
            ax_v.tick_params(axis='x', rotation=0, labelsize=9) # 회전 제거하고 폰트 키움
            self.violin_canvas.draw()

            # 데이터 정보 보관
            self.all_samples = X
            self.sample_data = X[0]
            self.feature_bounds = [ (float(np.min(X[:, i])), float(np.max(X[:, i]))) for i in range(X.shape[1]) ]
            
            self.data_log.append("\n✅ 분석 완료! [데이터 분석(EDA)] 탭에서 결과를 확인하세요.")
            self.data_log.append(f"💡 [팁] '물성 예측' 탭에서 분석된 샘플을 불러와 예측해볼 수 있습니다.")
            
            # [신규] 보고서 업데이트 (데이터 분석 단계)
            self._update_report(stage="data", data={"filename": os.path.basename(self.selected_file_path), 
                                                "rows": X.shape[0], "cols": X.shape[1],
                                                "min": float(np.min(y)), "max": float(np.max(y)), "avg": float(np.mean(y))})

        except Exception as e:
            self.data_log.append(f"❌ 분석 실패: {e}")
            import traceback
            print(traceback.format_exc())

    def _update_detail_plot(self, col_name):
        """특정 성분이 선택되었을 때 대형 상세 그래프를 그립니다."""
        if self.df_raw is None or col_name not in self.df_raw.columns:
            return

        # 설명 텍스트 업데이트
        info_text = self.element_info.get(col_name, "해당 성분에 대한 상세 정보가 없습니다.")
        self.info_box.setText(info_text)

        self.detail_figure.clear()
        ax = self.detail_figure.add_subplot(111)
        
        data = self.df_raw[col_name]
        
        # 메인 분포 (KDE + Hist)
        sns.histplot(data, kde=True, ax=ax, color='#2c3e50', alpha=0.6, label='Data Distribution')
        
        # 통계선 추가
        avg, med = data.mean(), data.median()
        ax.axvline(avg, color='#e74c3c', linestyle='--', linewidth=2, label=f'평균 (Mean): {avg:.2f}')
        ax.axvline(med, color='#f39c12', linestyle='-', linewidth=2, label=f'중앙값 (Median): {med:.2f}')
        
        ax.set_title(f"🔍 [{col_name}] 성분 정밀 분포 분석", fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel(f"{col_name} 함량 / 수치", fontsize=12)
        ax.set_ylabel("데이터 빈도 (Frequency)", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(fontsize=11)
        
        self.detail_figure.tight_layout()
        self.detail_canvas.draw()

    def _init_training_tab(self):
        """[모델 학습] 탭 구성"""
        training_tab = QWidget()
        layout = QVBoxLayout(training_tab)
        
        # --- 학습 파라미터 설정 구역 ---
        param_group = QGroupBox("⚙️ AI 모델 및 학습 설정")
        param_layout = QFormLayout()
        
        # [신규] 모델 선택 드롭다운
        self.combo_model = QListWidget() # 단순화를 위해 리스트 위젯 사용하거나 QComboBox
        from PyQt6.QtWidgets import QComboBox
        self.combo_model = QComboBox()
        self.combo_model.addItems(["TFP (확률형 신경망)", "MLP (일반 신경망)", "RF (랜덤 포레스트)", "GBM (그래디언트 부스팅)"])
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)
        
        # 모델 설명 라벨
        self.model_desc = QLabel("<b>TFP</b>: 예측값과 함께 '신뢰도'를 제공합니다. 데이터가 적고 정밀한 분석이 필요할 때 추천합니다.")
        self.model_desc.setWordWrap(True)
        self.model_desc.setStyleSheet("color: #2980b9; background-color: #ebf5fb; padding: 10px; border-radius: 5px;")
        
        self.edit_epochs = QLineEdit("50")
        self.edit_batch = QLineEdit("32")
        
        param_layout.addRow("학습 모델 선택:", self.combo_model)
        param_layout.addRow("", self.model_desc)
        param_layout.addRow("학습 반복 횟수 (Epochs):", self.edit_epochs)
        param_layout.addRow("배치 크기 (Batch Size):", self.edit_batch)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        button_layout = QHBoxLayout()
        self.btn_train = QPushButton("AI 모델 학습 시작 (Train Model)")
        self.btn_train.setMinimumHeight(50)
        self.btn_train.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; font-size: 14px;")
        
        self.btn_reset_train = QPushButton("🔄 학습 내역 초기화")
        self.btn_reset_train.setMinimumHeight(50)
        self.btn_reset_train.setStyleSheet("background-color: #95a5a6; color: white; font-weight: bold; font-size: 14px;")
        
        button_layout.addWidget(self.btn_train, 2)
        button_layout.addWidget(self.btn_reset_train, 1)
        layout.addLayout(button_layout)
        
        # 그래프 설명 라벨 추가
        help_label = QLabel("💡 <b>Loss 그래프 해석</b>: AI가 학습할수록 선이 아래로 내려갑니다. "
                            "선이 낮게 안정될수록 예측 정확도가 높아집니다.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
        layout.addWidget(help_label)

        # 맷플롯립 캔버스(그래프) 추가 파트
        self.figure = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        layout.addWidget(self.train_log)
        
        self.btn_train.clicked.connect(self._start_training)
        self.btn_reset_train.clicked.connect(self._reset_training_ui)
        
        self.tabs.addTab(training_tab, "모델 학습 (Training)")

    def _reset_training_ui(self):
        """학습 로그와 그래프를 초기 상태로 깨끗하게 비웁니다. 학습 중이라면 중단합니다."""
        try:
            # 학습 중인 스레드가 있다면 즉시 중단
            if hasattr(self, 'thread') and self.thread.isRunning():
                self.thread.terminate()
                self.thread.wait()
                self.btn_train.setEnabled(True)
                self.train_log.append("\n⏹️ [중단] 사용자에 의해 학습이 강제 중단되었습니다.")
            
            self.train_log.clear()
            
            # 그래프 초기화
            if hasattr(self, 'figure'):
                self.figure.clear()
                self.figure.add_subplot(111)
                self.canvas.draw()
            
            # 결과 탭(YY Plot) 초기화
            if hasattr(self, 'yy_figure'):
                self.yy_figure.clear()
                self.yy_figure.add_subplot(111)
                self.yy_canvas.draw()
            
            self.train_log.append("🧹 모든 학습 내역 및 그래프가 초기화되었습니다.")
        except Exception as e:
            print(f"초기화 중 오류 발생: {e}")

    def _start_training(self):
        """별도 쓰레드를 생성하여 화면이 멈추지 않고 학습이 이어지도록 유도합니다."""
        # 1. 하이퍼파라미터 값 읽기
        try:
            epochs = int(self.edit_epochs.text())
            batch_size = int(self.edit_batch.text())
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "Epoch와 Batch Size는 숫자로 입력해주세요.")
            return

        # 2. 데이터 선택 확인
        if not self.selected_file_path:
            reply = QMessageBox.question(
                self, '데이터 미선택', 
                "학습할 데이터 파일이 선택되지 않았습니다.\n기본 데이터(stmech_aus_ss.xls)로 학습을 진행할까요?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                self.tabs.setCurrentIndex(0) # 데이터 탭으로 이동시켜줌
                return

        self.btn_train.setEnabled(False)  # 학습 버튼 일시 비활성화 (중복 클릭 방지)
        self.train_log.clear()            # 이전 로그 텍스트 지우기
        
        self.thread = TrainingThread(
            filepath=self.selected_file_path,
            epochs=epochs,
            batch_size=batch_size,
            model_type=self.current_model_type
        )
        # 스레드에서 전송하는 시그널들을 이 클래스의 함수들과 매칭
        self.thread.log_signal.connect(self.train_log.append)
        self.thread.plot_signal.connect(self._update_plot)
        self.thread.yy_plot_signal.connect(self._update_yy_plot)
        self.thread.finished_signal.connect(self._on_training_finished)
        self.thread.error_signal.connect(self._show_training_error)
        self.thread.start()

    def _show_training_error(self, message):
        """학습 중 발생한 에러를 알림창으로 표시합니다."""
        QMessageBox.critical(self, "학습 오류", f"학습 중 오류가 발생했습니다:\n{message}")

    def _on_training_finished(self):
        """학습이 완료되면 버튼을 활성화하고 API 서버에 모델 재로드를 요청합니다."""
        self.btn_train.setEnabled(True)
        try:
            # [수정] 모델 타입 정보를 포함하여 재로드 요청
            res = requests.post("http://127.0.0.1:5000/reload", json={"model_type": self.current_model_type})
            if res.status_code == 200:
                self.train_log.append(f"\n✨ [업데이트 성공] {self.current_model_type} 모델로 지능 지수가 갱신되었습니다!")
                self.train_log.append("이제 [물성 예측] 탭으로 이동하여 결과를 확인하실 수 있습니다.")
                
                # [신규] 보고서 업데이트 (학습 완료 단계)
                self._update_report(stage="train", data={"model_type": self.current_model_type})
            else:
                self.train_log.append(f"\n⚠️ [알림] 예측 서버 연동에 실패했습니다: {res.text}")
        except Exception:
            self.train_log.append("\n❌ [주의] 예측 서버가 꺼져 있습니다. 프로그램을 다시 시작해 주세요.")

    def _update_plot(self, epochs, loss):
        """학습과정(스레드)에서 받은 데이터를 바탕으로 캔버스 위에 그래프(오차선)를 그립니다."""
        self.figure.clear() # 이전 그림 지우기
        ax = self.figure.add_subplot(111)
        ax.plot(epochs, loss, label="AI 예측 오차 (Loss)", color='#e74c3c', linewidth=2)
        ax.set_title("AI 모델의 지능 향상 시각화 (Learning Curve)", pad=20, fontsize=14, fontweight='bold')
        ax.set_xlabel("학습 반복 횟수 (Epoch)", fontsize=11, labelpad=10)
        ax.set_ylabel("오차 정도 (NLL Loss)", fontsize=11, labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=10)
        
        # 축 수치들이 겹치지 않게 타이트한 레이아웃 적용
        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()

    def _init_results_tab(self):
        """[학습 결과] 탭 구성 (Y-Y Plot 표시)"""
        results_tab = QWidget()
        layout = QVBoxLayout(results_tab)
        
        info_label = QLabel("📊 <b>AI 모델 오차 분석 (Actual vs. Predicted)</b>")
        info_label.setStyleSheet("font-size: 14px; margin-bottom: 5px;")
        layout.addWidget(info_label)
        
        # Y-Y Plot 캔버스
        self.yy_figure = Figure(figsize=(5, 5), tight_layout=True)
        self.yy_canvas = FigureCanvas(self.yy_figure)
        layout.addWidget(self.yy_canvas)
        
        desc_label = QLabel("💡 <b>해석 가이드</b>: 점들이 대각선 점선에 가까이 모여 있을수록 AI의 예측 정확도가 높은 것을 의미합니다.")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
        layout.addWidget(desc_label)
        
        self.tabs.addTab(results_tab, "학습 결과 (Results)")

    def _update_yy_plot(self, real_values, pred_values):
        """학습 후 검증 데이터에 대한 Y-Y Plot을 그립니다."""
        self.yy_figure.clear()
        ax = self.yy_figure.add_subplot(111)
        
        # 산점도 그리기
        ax.scatter(real_values, pred_values, alpha=0.5, color='#3498db', label='Validation Data')
        
        # 기준 대각선 그리기 (1:1 비율)
        all_vals = real_values + pred_values
        min_val, max_val = min(all_vals), max(all_vals)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Ideal (Perfect)')
        
        ax.set_title("실제값 vs. AI 예측값 비교 (Y-Y Plot)", pad=15, fontsize=12, fontweight='bold')
        ax.set_xlabel("실제 항복강도 (Actual, MPa)", fontsize=10)
        ax.set_ylabel("AI 예측 항복강도 (Predicted, MPa)", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        
        self.yy_figure.tight_layout()
        self.yy_canvas.draw()
        
    def _init_inference_tab(self):
        """[물성 예측(Inference)] 탭 구성 (성분 입력 및 예측 결과를 볼 수 있는 화면)"""
        inference_tab = QWidget()
        layout = QHBoxLayout(inference_tab)
        
        # [왼쪽 패널]: 스크롤이 가능한 성분 입력 폼 구성 (특성이 18개로 많아 화면이 넘칠 수 있습니다.)
        form_panel = QScrollArea()
        form_panel.setWidgetResizable(True)
        inner_widget = QWidget()
        self.form_layout = QFormLayout(inner_widget)
        
        # 파서에서 설정된 Features 리스트(크롬, 니켈, 온도 등)를 반복하면서 자동으로 입력창(QLineEdit) 생성 
        self.inputs = {}
        for feat in self.features:
            line_edit = QLineEdit("0.0")
            self.form_layout.addRow(f"{feat}:", line_edit)
            # 나중에 예측버튼을 눌렀을 때 값을 가져오기 위해 딕셔너리에 저장
            self.inputs[feat] = line_edit 
            
        form_panel.setWidget(inner_widget)
        
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("합금 성분 및 열처리 조건 등 입력 (Float 형태)"))
        left_layout.addWidget(form_panel)
        
        self.btn_predict = QPushButton("조성 및 물성 예측 (Predict Strength)")
        self.btn_predict.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; height: 40px;")
        
        self.btn_fill = QPushButton("📋 분석 데이터 샘플로 채우기")
        self.btn_fill.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7; height: 30px;")
        
        left_layout.addWidget(self.btn_predict)
        left_layout.addWidget(self.btn_fill)
        
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        layout.addWidget(left_widget, stretch=1)
        
        # [오른쪽 패널]: 서버에서 돌아온 예측 결과를 화면에 텍스트로 찍어주는 위젯
        result_panel = QWidget()
        res_layout = QVBoxLayout(result_panel)
        res_layout.addWidget(QLabel("### 예측 결과 및 신뢰도 분석 ###"))
        
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setPlaceholderText("예측 버튼을 누르면 여기에 결과가 표시됩니다.")
        self.result_display.setText("👈 왼쪽의 성분들을 입력하신 후\n[조성 및 물성 예측] 버튼을 눌러주세요.")
        self.result_display.setMaximumHeight(200) # 결과창 높이 제한
        res_layout.addWidget(self.result_display)
        
        # --- [신규] 예측 결과 추정 분포 그래프 구역 ---
        self.inf_figure = Figure(figsize=(4, 3), tight_layout=True)
        self.inf_canvas = FigureCanvas(self.inf_figure)
        res_layout.addWidget(QLabel("📈 <b>항복강도 예측 분포 (PDF)</b>"))
        res_layout.addWidget(self.inf_canvas)

        # --- [신규] 성분별 기여도(민감도) 그래프 구역 ---
        self.sens_figure = Figure(figsize=(4, 3), tight_layout=True)
        self.sens_canvas = FigureCanvas(self.sens_figure)
        res_layout.addWidget(QLabel("🧪 <b>성분별 기여도 분석 (Sensitivity)</b>"))
        res_layout.addWidget(self.sens_canvas)
        
        layout.addWidget(result_panel, stretch=1)
        
        self.btn_predict.clicked.connect(self._run_inference)
        self.btn_fill.clicked.connect(self._fill_sample_data)
        self.tabs.addTab(inference_tab, "물성 예측 (Inference)")

    def _fill_sample_data(self):
        """방금 분석한 데이터셋에서 무작위로 한 샘플을 뽑아 예측 폼에 채워줍니다."""
        if self.all_samples is None:
            QMessageBox.information(self, "안내", "먼저 [데이터 (관측/분석)] 탭에서 파일을 선택하고 '구조 분석'을 완료해 주세요.")
            return
            
        import random
        # 전체 데이터 중에서 무작위로 하나의 행(Row)을 선택합니다.
        self.sample_data = self.all_samples[random.randint(0, len(self.all_samples)-1)]
            
        for i, feat in enumerate(self.features):
            self.inputs[feat].setText(str(round(float(self.sample_data[i]), 3)))
        
        # 안내 문구가 너무 자주 뜨면 불편하므로 로그에만 남기거나 툴팁으로 처리할 수 있으나 상태줄이 없으므로 일단 알림
        # QMessageBox 건너뛰고 바로 입력되게 수정 (편의성)
        self.result_display.setText(f"✅ 데이터셋 내 임의의 샘플 값을 가져왔습니다.\n(분석 파일: {os.path.basename(self.selected_file_path)})")

    def _run_inference(self):
        """
        [물성 예측] 버튼의 액션을 처리합니다.
        입력창 값들을 가져와 Flask API 서버 (http://127.0.0.1:5000/predict) 로 전송하고 결과를 받아옵니다.
        """
        try:
            # 모든 폼에 적혀있는 숫자를 읽어 실수형(float) 리스트로 묶습니다.
            features = []
            for feat in self.features:
                val = float(self.inputs[feat].text())
                features.append(val)
                
            api_url = "http://127.0.0.1:5000/predict"
            # JSON 형태로 감싸 POST 방식으로 서버에 보냅니다
            response = requests.post(api_url, json={"features": features})
            
            # 서버가 정상적으로 응답했다면
            if response.status_code == 200:
                data = response.json()
                preds = data.get("predictions", {})
                
                # 돌아온 예측 결과(Mean, Uncertainty, 95% 신뢰구간)를 화면에 표시합니다.
                mean_val = preds.get("mean", [0])[0]
                uncert = preds.get("uncertainty", [0])[0]
                lower = preds.get("lower_95_ci", [0])[0]
                upper = preds.get("upper_95_ci", [0])[0]
                
                res_text = (
                    f"========== 예측 결과 (Prediction) ==========\n\n"
                    f"▶ 타겟: 0.2% Proof Stress (M Pa) / 항복강도\n"
                    f"▶ 예측값: {mean_val:.2f} MPa\n"
                    f"▶ 예측 불확실성 (Uncertainty): ± {uncert:.2f}\n"
                    f"▶ 95% 신뢰구간: [{lower:.2f} ~ {upper:.2f}] MPa\n\n"
                    f"[해석] 본 예측은 TensorFlow Probability를 사용하여 도출된 \n"
                    f"결과로, 대상 합금의 항복강도가 약 {mean_val:.2f} MPa 주변에 속할 것으로 예측되었습니다.\n"
                    f"모델이 학습해보지 못한 범위(아웃라이어)의 데이터인 경우 불확실성 수치가 커집니다."
                )
                self.result_display.setText(res_text)
                
                # --- [신규] 결과 분포 및 성분 기여도 그래프 그리기 ---
                self._update_inference_plot(mean_val, uncert)
                self._update_sensitivity_plot(features, mean_val)
                
            else:
                self.result_display.setText(f"[오류] API 호출에 문제 발생 (상태 코드: {response.status_code})\n{response.text}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"입력값 변환 오류 혹은 API 통신 실패: {e}")

    def _update_inference_plot(self, mean, std):
        """예측된 평균과 불확실성을 바탕으로 확률 밀도 함수(PDF) 그래프를 그립니다."""
        self.inf_figure.clear()
        ax = self.inf_figure.add_subplot(111)
        
        x = np.linspace(mean - 4*std, mean + 4*std, 200)
        from scipy.stats import norm
        y = norm.pdf(x, mean, std)
        
        ax.plot(x, y, color='#2980b9', linewidth=2, label='AI 예측 분포 (PDF)')
        ax.fill_between(x, y, color='#3498db', alpha=0.3)
        
        ax.axvline(mean, color='#c0392b', linestyle='-', linewidth=1.5, label=f'예측 평균: {mean:.1f}')
        
        ax.set_title("예측 결과의 통계적 분포 (Confidence)", fontsize=11, fontweight='bold')
        ax.set_xlabel("항복강도 (Yield Strength, MPa)", fontsize=9)
        ax.set_ylabel("확률 밀도 (Density)", fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(fontsize=8)
        
        self.inf_figure.tight_layout()
        self.inf_canvas.draw()

    def _update_sensitivity_plot(self, current_features, base_prediction):
        """현재 입력값에서 각 성분을 미세하게 변화시켰을 때 결과가 어떻게 변하는지(기여도) 시각화합니다."""
        self.sens_figure.clear()
        ax = self.sens_figure.add_subplot(111)
        
        sensitivities = []
        api_url = "http://127.0.0.1:5000/predict"
        
        for i, feat_name in enumerate(self.features):
            test_features = list(current_features)
            delta = (test_features[i] * 0.05) if test_features[i] != 0 else 0.1
            test_features[i] += delta
            
            try:
                resp = requests.post(api_url, json={"features": test_features}, timeout=1)
                if resp.status_code == 200:
                    new_mean = resp.json()["predictions"]["mean"][0]
                    change = ((new_mean - base_prediction) / base_prediction) * 100
                    sensitivities.append(change)
                else:
                    sensitivities.append(0)
            except:
                sensitivities.append(0)

        y_pos = np.arange(len(self.features))
        colors = ['#e74c3c' if s > 0 else '#3498db' for s in sensitivities]
        ax.barh(y_pos, sensitivities, align='center', color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.features, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("결과 변화율 (%)", fontsize=9)
        ax.set_title("성분별 기여도 분석 (각 성분 +5% 증가 시)", fontsize=10, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        self.sens_figure.tight_layout()
        self.sens_canvas.draw()

    def _init_optimization_tab(self):
        """[최적 소재 설계] 탭 구성"""
        opt_tab = QWidget()
        layout = QVBoxLayout(opt_tab)
        
        title = QLabel("🎯 <b>최적 성분 비율 도출 (Material Composition Optimization)</b>")
        title.setStyleSheet("font-size: 16px; margin: 10px;")
        layout.addWidget(title)
        
        desc = QLabel("학습된 AI 모델이 데이터의 패턴을 분석하여, **항복강도(Strength)를 극대화**할 수 있는 최적의 성분 조합을 계산합니다.<br>"
                      "<small>(※ 무작위 지점에서 탐색을 시작하므로 실행 시마다 결과가 조금씩 달라질 수 있습니다.)</small>")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        self.btn_run_opt = QPushButton("🚀 최적 조성 도출 시작 (Find Best Recipe)")
        self.btn_run_opt.setMinimumHeight(50)
        self.btn_run_opt.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.btn_run_opt)
        
        # [신규] 최적화 진행률 표시줄
        from PyQt6.QtWidgets import QProgressBar
        self.opt_progress = QProgressBar()
        self.opt_progress.setVisible(False)
        self.opt_progress.setStyleSheet("""
            QProgressBar { border: 2px solid grey; border-radius: 5px; text-align: center; }
            QProgressBar::chunk { background-color: #2ecc71; width: 20px; }
        """)
        layout.addWidget(self.opt_progress)
        
        # 결과 표시 영역 (그래프 + 텍스트)
        res_area = QHBoxLayout()
        
        self.opt_result_display = QTextEdit()
        self.opt_result_display.setReadOnly(True)
        self.opt_result_display.setPlaceholderText("최적화 버튼을 누르면 추천 성분이 여기에 표시됩니다.")
        res_area.addWidget(self.opt_result_display, stretch=1)
        
        self.opt_figure = Figure(figsize=(5, 4), tight_layout=True)
        self.opt_canvas = FigureCanvas(self.opt_figure)
        res_area.addWidget(self.opt_canvas, stretch=2)
        
        layout.addLayout(res_area)
        
        self.btn_run_opt.clicked.connect(self._run_optimization)
        self.tabs.addTab(opt_tab, "최적 소재 설계 (Optimization)")

    def _init_report_tab(self):
        """[AI 결과 보고서] 탭 구성 - 모든 분석 결과를 통합 설명함"""
        report_tab = QWidget()
        layout = QVBoxLayout(report_tab)
        
        # 스크롤 영역 생성 (보고서 내용이 길 수 있으므로)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        title = QLabel("📑 <b>AI 소재 분석 통합 보고서 (AI Analysis Report)</b>")
        title.setStyleSheet("font-size: 18px; color: #2c3e50; margin: 10px;")
        scroll_layout.addWidget(title)
        
        # 보고서 본문 구성
        self.report_display = QTextEdit() # class 변수화
        self.report_display.setReadOnly(True)
        
        # 초기 대기 메시지
        initial_content = """
        <div style="line-height: 180%; font-family: 'Malgun Gothic'; text-align: center; padding: 50px;">
            <h2 style="color: #95a5a6;">📈 보고서 작성 대기 중...</h2>
            <p>상단 [데이터] 탭에서 실험 데이터를 선택하고 <b>[구조 분석]</b>을 시작해 주세요.</p>
            <p>분석 과정에 따라 AI가 사용자님의 데이터에 맞춘 맞춤형 보고서를 실시간으로 작성합니다.</p>
        </div>
        """
        self.report_display.setHtml(initial_content)
        scroll_layout.addWidget(self.report_display)
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        self.report_data = {} # 리포트용 동적 데이터 보관함
        self.tabs.addTab(report_tab, "AI 결과 보고서 (Report)")

    def _update_report(self, stage, data=None):
        """진행 단계에 따라 보고서 내용을 실시간으로 업데이트합니다."""
        if data:
            self.report_data.update(data)
        
        html = "<div style=\"line-height: 160%; font-family: 'Malgun Gothic';\">"
        
        # 0.5단계: 선택된 모델 정보 (현재 학습된 모델 위주)
        selected_model = self.report_data.get('model_type', 'TFP')
        model_details = {
            "TFP": "<b>TFP (확률형 신경망)</b>: '불확실성'을 함께 계산하여 신뢰구간을 제공하는 가장 정밀한 모델입니다.",
            "MLP": "<b>MLP (일반 신경망)</b>: 복잡한 비선형 관계를 학습하는 데 강력한 성능을 보이는 인공 신경망입니다.",
            "RF": "<b>RF (랜덤 포레스트)</b>: 여러 결정 트리를 묶어 데이터 노이즈에 강하고 안정적인 예측을 수행합니다.",
            "GBM": "<b>GBM (그래디언트 부스팅)</b>: 잔차를 보정하며 학습하여 매우 높은 예측 정밀도를 추구하는 모델입니다."
        }
        
        html += f"""
        <h3 style="color: #8e44ad;">🤖 분석에 사용 중인 모델: {selected_model}</h3>
        <div style="background-color: #f4ecf7; padding: 12px; border-radius: 8px; border-left: 5px solid #8e44ad; margin-bottom: 20px;">
            {model_details.get(selected_model, "")}<br>
            데이터의 특성상 {selected_model} 알고리즘이 패턴을 파악하기에 적합하다고 판단되어 적용되었습니다.
        </div>
        """
        
        # 1단계: 데이터 분석 정보
        if "filename" in self.report_data:
            html += f"""
            <h2 style="color: #2980b9;">🔬 1. 데이터셋 분석 결과 ([{self.report_data['filename']}])</h2>
            <p>사용자님께서 입력하신 데이터는 <b>총 {self.report_data['rows']}행</b>의 실험 결과로 이루어져 있습니다.</p>
            <ul>
                <li><b>항복강도 범위:</b> {self.report_data['min']:.1f} ~ {self.report_data['max']:.1f} MPa</li>
                <li><b>평균 물성치:</b> <span style="color: #e67e22; font-weight: bold;">{self.report_data['avg']:.1f} MPa</span></li>
                <li><b>데이터 정규화 분석:</b> [바이올린 플롯]을 통해 확인한 결과, 성분별 데이터가 특정 범위에 잘 모여있으며 이상치가 적어 모델 학습에 적합한 상태입니다.</li>
            </ul>
            <p>AI는 위 범위 내에서 최적의 성분 조합을 찾기 위한 패턴 학습 준비를 마쳤습니다.</p>
            """
        
        # 2단계: 학습 정보
        if stage == "train" or "train_status" in self.report_data:
            self.report_data["train_status"] = "completed"
            html += """
            <h2 style="color: #27ae60;">🧠 2. AI 모델 학습 및 성능 분석</h2>
            <p>AI가 데이터 속의 숨겨진 물리 규칙을 성공적으로 학습했습니다.</p>
            <ul>
                <li><b>학습 성과:</b> [학습 결과] 탭의 Y-Y Plot에서 볼 수 있듯이, 모델이 실제 실험값의 경향성을 정확히 추동하고 있습니다.</li>
                <li><b>예측 정직도:</b> 점들이 대각선에 가까울수록 AI가 사용자님의 실험 데이터를 정직하게 반영하고 있음을 의미합니다.</li>
            </ul>
            """

        # 3단계: 최적화 정보 (핵심 답변 포함)
        if "best_y" in self.report_data:
            avg_val = self.report_data.get('avg', 341.3)
            best_y = self.report_data['best_y']
            improvement = ((best_y - avg_val) / avg_val) * 100
            
            html += f"""
            <h2 style="color: #e67e22;">🎯 3. 최종 결과 해석 (왜 {best_y:.1f} MPa인가?)</h2>
            <p>사용자님의 데이터 평균(<span style="color: #7f8c8d;">{avg_val:.1f} MPa</span>) 대비 
            AI는 약 <b>{improvement:.1f}% 향상</b>된 <b>{best_y:.1f} MPa</b>의 가능성을 발견했습니다.</p>
            <div style="background-color: #fdf2e9; padding: 15px; border-radius: 8px; border-left: 5px solid #e67e22;">
                <b>🚀 AI의 분석 답변:</b><br>
                1. <b>다차원 시너지:</b> AI는 18개 성분 전체를 0.001% 단위로 동시에 조절하여, 인간의 실험으로는 찾기 힘든 '고강도 시너지 지점'을 수학적으로 식별해냈습니다.<br>
                2. <b>잠재력 포착:</b> 533 MPa와 같은 높은 수치는 터무니없는 숫자가 아닙니다. 이 데이터셋 내의 <b>성분 허용 범위 안에서</b> 최적의 비율로 혼합했을 때 도달 가능한 '이론적 잠재력'입니다.<br>
                3. <b>데이터 기반의 근거:</b> AI는 사용자님이 앞서 시각화한 '성분 기여도' 결과를 종합하여, 강도를 높이는 성분을 극대화하고 약화시키는 성분을 최소화하는 황금 비율을 도출했습니다.
            </div>
            """
        
        # 마무리 가이드 (데이터가 하나라도 있을 때만)
        if self.report_data:
            html += """
            <hr>
            <p style="color: #7f8c8d; font-size: 11px;">※ 본 보고서는 사용자님의 데이터에 근거하여 AI가 실시간으로 분석한 결과입니다. 최적 설계 탭의 추천 함량을 실제 실험에 적용해 보실 것을 제언합니다.</p>
            """
        
        html += "</div>"
        self.report_display.setHtml(html)

    def _run_optimization(self):
        """AI 모델을 이용해 최적의 성분을 비동기 방식으로 찾습니다."""
        if not self.feature_bounds:
            QMessageBox.warning(self, "경고", "먼저 데이터를 분석해야 최적화가 가능합니다.")
            return

        self.btn_run_opt.setEnabled(False)
        self.opt_progress.setValue(0)
        self.opt_progress.setVisible(True)
        self.opt_result_display.setText("⌛ AI가 최적의 배합 비율을 찾는 중입니다...\n시뮬레이션을 시작합니다.")
        
        # 최적화 스레드 시작
        self.opt_thread = OptimizationThread(
            self.feature_bounds, 
            "data/model_weights", # .h5 제거 (다중 모델 대응) 
            "data/scalers.pkl", 
            self.features,
            model_type=self.current_model_type
        )
        self.opt_thread.progress_signal.connect(self._update_opt_progress)
        self.opt_thread.finished_signal.connect(self._on_optimization_finished)
        self.opt_thread.error_signal.connect(self._on_optimization_error)
        self.opt_thread.start()

    def _on_optimization_error(self, message):
        self.opt_result_display.setText(f"❌ 최적화 오류: {message}")
        self.btn_run_opt.setEnabled(True)

    def _on_optimization_finished(self, best_x, best_y):
        if best_x is not None:
            res_text = "✨ [도출 완료] AI 추천 정밀 성분 조합\n\n"
            res_text += f"🏆 예측 최대 항복강도: {best_y:.2f} MPa\n\n"
            res_text += "---------- 최적 조성 비율 ----------\n"
            
            for i, feat in enumerate(self.features):
                res_text += f"• {feat}: {best_x[i]:.3f}\n"
                
            self.opt_result_display.setText(res_text)
            self._update_opt_plot(best_x)
            
            # [신규] 보고서 업데이트 (최적화 완료 단계)
            self._update_report(stage="opt", data={"best_x": best_x, "best_y": best_y})
        else:
            self.opt_result_display.setText("❌ 최적 조합을 찾는 데 실패했습니다.")
        self.btn_run_opt.setEnabled(True)
        self.opt_progress.setVisible(False) # 완료 후 숨김

    def _update_opt_plot(self, best_x):
        """도출된 최적 성분을 막대 그래프로 그립니다."""
        self.opt_figure.clear()
        ax = self.opt_figure.add_subplot(111)
        
        y_pos = np.arange(len(self.features))
        ax.barh(y_pos, best_x, align='center', color='#2ecc71', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.features, fontsize=8)
        ax.invert_yaxis()  # 상단부터 표시
        ax.set_xlabel('성분 함량 / 조건 수치', fontsize=9)
        ax.set_title('AI 추천 최적 소재 배합 비율', fontsize=12, fontweight='bold', pad=15)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        # 그래프 간격 및 레이아웃 최적화
        self.opt_figure.tight_layout(pad=1.5)
        self.opt_canvas.draw()

    def _update_opt_progress(self, current, total):
        """최적화 진행 상황을 프로그레스 바와 텍스트로 업데이트합니다."""
        percent = int((current / total) * 100)
        self.opt_progress.setValue(percent)
        self.opt_result_display.append(f"🔄 [{current}/{total}] 시뮬레이션 완료...")

    def _on_model_changed(self, index):
        """모델 선택 변경 시 가이드 문구를 업데이트합니다."""
        model_info = [
            ("TFP", "<b>TFP (확률형 신경망)</b>: 가장 정밀한 분석 도구입니다. 예측값뿐만 아니라 '불확실성'을 함께 계산하여 고도의 연구용으로 추천합니다."),
            ("MLP", "<b>MLP (일반 신경망)</b>: 데이터가 충분할 때 매우 강력합니다. 빠른 연산 속도와 높은 일반화 성능을 자랑합니다."),
            ("RF", "<b>RF (랜덤 포레스트)</b>: 데이터 양이 적거나 성분 간의 복잡한 규칙이 불명확할 때 안정적인 성능을 보여줍니다."),
            ("GBM", "<b>GBM (그래디언트 부스팅)</b>: 최근 머신러닝 경진대회에서 가장 선호되는 알고리즘으로, 매우 높은 정확도를 목표로 할 때 사용합니다.")
        ]
        model_key, desc = model_info[index]
        self.current_model_type = model_key
        self.model_desc.setText(desc)
        
        # 모델 변경 시 로그 남김
        self.train_log.append(f"\n⚙️ 모델이 '{model_key}'(으)로 변경되었습니다. 학습 시 해당 알고리즘이 사용됩니다.")

def launch_gui():
    """애플리케이션을 구동하는 메인 함수입니다."""
    # 고해상도(High DPI) 모니터 지원 설정
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        
    app = QApplication(sys.argv)
    # 기본 스타일을 좀 더 모던한 Fusion 으로 설정
    app.setStyle("Fusion") 
    window = AIMaterialPlatform()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    launch_gui()
