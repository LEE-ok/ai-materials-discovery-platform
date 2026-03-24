import sys
import os
import pandas as pd
import numpy as np
import joblib
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QTabWidget, QFileDialog, 
                             QFormLayout, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QSpinBox, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.pyplot as plt

# Handle Korean font for Matplotlib on Windows
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

from src.engine.data_engine import DataEngine
from src.engine.model_engine import ModelEngine

class TrainingThread(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(self, data_engine, model_type='RF', max_iter=2000):
        super().__init__()
        self.data_engine = data_engine
        self.model_type = model_type
        self.max_iter = max_iter

    def run(self):
        try:
            self.progress.emit("데이터 로딩 및 전처리 중...")
            try:
                self.data_engine.load_data()
            except Exception as e:
                self.finished.emit(f"엑셀 파일을 읽는 중 오류 발생: {str(e)}")
                return

            validation = self.data_engine.validation_summary or {}
            if validation:
                self.progress.emit(
                    f"정합성 검증 결과 - pass:{validation.get('rows_pass', 0)} "
                    f"hold:{validation.get('rows_hold', 0)} reject:{validation.get('rows_reject', 0)}"
                )
                
            X_train, X_test, y_train, y_test, X_raw_test, y_raw_test = self.data_engine.preprocess_data()
            
            if len(X_train) == 0:
                self.finished.emit("오류: 정제된 데이터가 없습니다.")
                return

            self.progress.emit(f"모델 초기화 중 (선택: {self.model_type}, Iter: {self.max_iter})...")
            model_engine = ModelEngine(model_type=self.model_type, 
                                      output_dim=y_train.shape[1],
                                      max_iter=self.max_iter)
            
            self.progress.emit(f"모델 학습 중 ({self.model_type})...")
            history = model_engine.train(X_train, y_train)
            
            self.progress.emit("모델 저장 중...")
            if not os.path.exists("models"): os.makedirs("models")
            model_engine.save("models/material_model.pkl")
            joblib.dump(self.data_engine, "models/data_engine.pkl")
            
            self.progress.emit("성능 검증 중...")
            mean_scaled, std_scaled = model_engine.predict(X_test)
            y_pred = self.data_engine.inverse_transform_y(mean_scaled)
            
            from sklearn.metrics import r2_score, mean_absolute_error
            r2 = r2_score(y_raw_test, y_pred, multioutput='raw_values')
            mae = mean_absolute_error(y_raw_test, y_pred, multioutput='raw_values')
            
            results = {
                "model": model_engine,
                "metrics": {"r2": r2, "mae": mae},
                "y_test": y_raw_test,
                "y_pred": y_pred,
                "validation_summary": validation
            }
            self.finished.emit(results)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.finished.emit(f"Unexpected error: {str(e)}")

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Materials Discovery Platform")
        self.resize(1100, 800)
        # Initializing with NO default file to ensure user chooses one
        self.data_engine = DataEngine(None)
        self.model_engine = None
        self.model_type = "RF" # Default
        
        self.last_corr = None # For dynamic resizing of annotations
        self.init_ui()
        # Removed load_existing_model() to prevent data appearing without user action

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Header
        header = QLabel("AI Materials Discovery Platform")
        header.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #2c3e50; margin: 10px;")
        layout.addWidget(header)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        self.setup_training_tab()
        self.setup_performance_tab() # New performance tab
        self.setup_inference_tab()
        self.setup_validation_tab()
        self.setup_analysis_tab()

    def setup_training_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left Panel: Controls
        left_panel = QVBoxLayout()
        
        info_group = QGroupBox("Model Training")
        info_layout = QVBoxLayout(info_group)
        
        self.file_path_label = QLabel("파일: 선택되지 않음")
        self.file_path_label.setStyleSheet("font-size: 10px; color: #7f8c8d;")
        info_layout.addWidget(self.file_path_label)
        
        self.select_file_btn = QPushButton("Select Data File (.xls/.xlsx)")
        self.select_file_btn.clicked.connect(self.on_select_file_clicked)
        info_layout.addWidget(self.select_file_btn)
        
        self.status_label = QLabel("상태: 데이터를 선택해 주세요")
        info_layout.addWidget(self.status_label)
        
        # New: Model Selection & Parameters
        model_selection_group = QGroupBox("AI 모델 및 학습 설정")
        ms_layout = QFormLayout(model_selection_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Random Forest (안정적)", "Gradient Boosting (정확함)", "Neural Network (MLP)", "TFP (확률론적 모델)"])
        ms_layout.addRow("학습 모델:", self.model_combo)
        
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(100, 10000)
        self.iter_spin.setValue(2000)
        self.iter_spin.setSingleStep(500)
        ms_layout.addRow("최대 반복 횟수:", self.iter_spin)
        
        help_label = QLabel("* 분석 성능에 진전이 없으면 조기 종료(Early Stopping)되어 모델을 보호합니다.\n"
                            "* 주의: 반복 횟수를 무조건 높인다고 해서 정확도가 반드시 향상되는 것은 아닙니다.")
        help_label.setStyleSheet("font-family: 'Malgun Gothic'; font-size: 10px; color: #3498db; font-style: italic; line-height: 1.2;")
        ms_layout.verticalSpacing = 10
        info_layout.addWidget(model_selection_group)
        info_layout.addWidget(help_label)

        self.train_btn = QPushButton("모델 학습 시작 (Start)")
        self.train_btn.setFixedHeight(45)
        self.train_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; font-size: 13px;")
        self.train_btn.clicked.connect(self.on_train_clicked)
        info_layout.addWidget(self.train_btn)
        
        # Simple Metrics display
        self.metrics_label = QLabel("<b>모델 성능 요약:</b><br>- 예측 정확도: N/A<br>- 평균 오차: N/A")
        self.metrics_label.setStyleSheet("background-color: #f8f9fa; padding: 10px; border-radius: 5px;")
        info_layout.addWidget(self.metrics_label)
        
        help_metrics = QLabel("<i>* 정확도(R2): 100%에 가까울수록 완벽함<br>* 오차(MAE): 실제값과 차이(낮을수록 좋음)</i>")
        help_metrics.setStyleSheet("font-size: 9px; color: #95a5a6;")
        info_layout.addWidget(help_metrics)
        
        left_panel.addWidget(info_group)
        left_panel.addStretch()
        
        # Right Panel: Plots
        right_panel = QVBoxLayout()
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        right_panel.addWidget(self.canvas)
        
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 2)
        self.tabs.addTab(tab, "모델 학습 및 최적화")

    def setup_inference_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left: Inputs (Scrollable)
        from PyQt6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        left_widget = QWidget()
        left_panel = QVBoxLayout(left_widget)
        scroll.setWidget(left_widget)
        
        self.inputs = {}
        
        # New: Active Model Info
        self.active_model_info = QLabel("사용 중인 모델: 학습된 모델 없음")
        self.active_model_info.setStyleSheet("background-color: #fcf3cf; padding: 10px; border: 1px solid #f39c12; border-radius: 5px; font-weight: bold; margin-bottom: 10px;")
        left_panel.addWidget(self.active_model_info)
        
        # Group 1: Composition
        comp_group = QGroupBox("Chemical Composition (wt%)")
        comp_layout = QFormLayout(comp_group)
        comp_list = ['Cr', 'Ni', 'Mo', 'Mn', 'Si', 'Nb', 'Ti', 'Zr', 'Ta', 'V', 'W', 'Cu', 'N', 'C', 'B', 'P', 'S', 'Co', 'Al', 'Sn', 'Pb']
        for col in comp_list:
            le = QLineEdit()
            # Default for 18-8 stainless steel like
            default_map = {'Cr': '18.0', 'Ni': '8.0', 'Mn': '2.0', 'Si': '1.0', 'C': '0.08'}
            le.setText(default_map.get(col, '0.0'))
            comp_layout.addRow(QLabel(col), le)
            self.inputs[col] = le
        left_panel.addWidget(comp_group)
        
        # Group 2: Process & Structure
        proc_group = QGroupBox("Process & Structure")
        proc_layout = QFormLayout(proc_group)
        proc_list = [
            'Solution_treatment_temperature', 'Solution_treatment_time(s)', 
            'Water_Quenched_after_s.t.', 'Air_Quenched_after_s.t.', 
            'Grains mm-2', 'Type of melting', 'Size of ingot', 
            'Product form', 'Temperature (K)'
        ]
        for col in proc_list:
            le = QLineEdit()
            defaults = {
                'Solution_treatment_temperature': '1050',
                'Solution_treatment_time(s)': '3600',
                'Water_Quenched_after_s.t.': '1',
                'Air_Quenched_after_s.t.': '0',
                'Grains mm-2': '500',
                'Type of melting': '2',
                'Size of ingot': '50',
                'Product form': '3',
                'Temperature (K)': '300'
            }
            le.setText(defaults.get(col, '0'))
            proc_layout.addRow(QLabel(col), le)
            self.inputs[col] = le
        left_panel.addWidget(proc_group)
            
        predict_btn = QPushButton("Predict Properties")
        predict_btn.setFixedHeight(45)
        predict_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; margin-top: 10px;")
        predict_btn.clicked.connect(self.on_predict_clicked)
        left_panel.addWidget(predict_btn)
        left_panel.addStretch()
        
        # Right: Results
        right_panel = QVBoxLayout()
        result_group = QGroupBox("Predictions")
        self.result_layout = QVBoxLayout(result_group)
        
        self.result_display = QLabel("Enter parameters and click Predict.")
        self.result_display.setFont(QFont("Arial", 12))
        self.result_layout.addWidget(self.result_display)
        
        self.prediction_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.result_layout.addWidget(self.prediction_canvas)
        
        right_panel.addWidget(result_group)
        
        layout.addWidget(scroll, 1)
        layout.addLayout(right_panel, 1)
        self.tabs.addTab(tab, "물성 예측 및 분석")

    def setup_performance_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        header = QLabel("상세 모델 성능 분석 (Predicted vs Actual)")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)
        
        self.perf_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        layout.addWidget(self.perf_canvas)
        
        desc = QLabel("* 점들이 대각성(y=x)에 가까울수록 모델의 예측이 실제값과 일치함을 의미합니다.")
        desc.setStyleSheet("color: #7f8c8d; font-style: italic;")
        layout.addWidget(desc)
        
        self.tabs.addTab(tab, "상세 성능 분석")

    def setup_analysis_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        header = QLabel("데이터 간 상관관계 분석 (Data Relationship)")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)
        
        self.analysis_btn = QPushButton("상관관계 그래프 생성 (Generate Relationship Plot)")
        self.analysis_btn.setFixedHeight(40)
        self.analysis_btn.clicked.connect(self.on_analysis_clicked)
        layout.addWidget(self.analysis_btn)
        
        # Container for plot to help with layout
        self.plot_container = QGroupBox("분석 결과 시각화")
        plot_layout = QVBoxLayout(self.plot_container)
        
        self.analysis_canvas = MplCanvas(self)
        self.analysis_canvas.setVisible(False) # Hide initially
        plot_layout.addWidget(self.analysis_canvas)
        
        # Placeholder message when hidden
        self.analysis_placeholder = QLabel("상단 버튼을 눌러 상관관계 분석을 시작하세요.\n데이터 양에 따라 수 초가 소요될 수 있습니다.")
        self.analysis_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analysis_placeholder.setStyleSheet("color: #95a5a6; font-size: 14px; border: 2px dashed #bdc3c7; padding: 100px; background-color: #f8f9fa;")
        plot_layout.addWidget(self.analysis_placeholder)
        
        layout.addWidget(self.plot_container, 1)
        
        desc = QLabel("* 상관관계 계수(숫자)가 1에 가까울수록 정비례, -1에 가까울수록 반비례를 의미합니다.")
        desc.setStyleSheet("color: #7f8c8d; font-style: italic; margin-top: 10px;")
        layout.addWidget(desc)
        
        self.tabs.addTab(tab, "데이터 관계 분석")

    def setup_validation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        header = QLabel("데이터 정합성 검증 결과")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        self.validation_summary_label = QLabel(
            "학습 실행 후 정합성 결과가 표시됩니다.\n"
            "pass/hold/reject 건수와 hold/reject 상세 행을 확인할 수 있습니다."
        )
        self.validation_summary_label.setStyleSheet(
            "background-color: #f8f9fa; border: 1px solid #dfe6e9; padding: 10px; border-radius: 5px;"
        )
        layout.addWidget(self.validation_summary_label)

        self.validation_table = QTableWidget()
        self.validation_table.setColumnCount(6)
        self.validation_table.setHorizontalHeaderLabels(
            ["Status", "Cr", "Ni", "Temp(K)", "Reason Codes", "Reason Detail"]
        )
        self.validation_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.validation_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.validation_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.validation_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.validation_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.validation_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.validation_table, 1)

        self.tabs.addTab(tab, "정합성 검증")

    def refresh_validation_tab(self):
        report = getattr(self.data_engine, "validation_report_df", None)
        summary = getattr(self.data_engine, "validation_summary", None)

        if summary is None:
            self.validation_summary_label.setText(
                "학습 실행 후 정합성 결과가 표시됩니다.\n"
                "pass/hold/reject 건수와 hold/reject 상세 행을 확인할 수 있습니다."
            )
            self.validation_table.setRowCount(0)
            return

        self.validation_summary_label.setText(
            f"총 {summary.get('rows_total', 0)}건 | "
            f"pass {summary.get('rows_pass', 0)}건 | "
            f"hold {summary.get('rows_hold', 0)}건 | "
            f"reject {summary.get('rows_reject', 0)}건"
        )

        if report is None or report.empty:
            self.validation_table.setRowCount(0)
            return

        show_df = report[report["quality_status"].isin(["hold", "reject"])].copy()
        if show_df.empty:
            self.validation_table.setRowCount(0)
            return

        cols = ["quality_status", "Cr", "Ni", "Temperature (K)", "quality_reason_codes", "quality_reason_detail"]
        for c in cols:
            if c not in show_df.columns:
                show_df[c] = ""
        show_df = show_df[cols].head(300)

        self.validation_table.setRowCount(len(show_df))
        for r, (_, row) in enumerate(show_df.iterrows()):
            status_item = QTableWidgetItem(str(row["quality_status"]))
            if row["quality_status"] == "reject":
                status_item.setBackground(QColor("#ffb3b3"))
            elif row["quality_status"] == "hold":
                status_item.setBackground(QColor("#fff3b0"))
            self.validation_table.setItem(r, 0, status_item)
            self.validation_table.setItem(r, 1, QTableWidgetItem(str(row["Cr"])))
            self.validation_table.setItem(r, 2, QTableWidgetItem(str(row["Ni"])))
            self.validation_table.setItem(r, 3, QTableWidgetItem(str(row["Temperature (K)"])))
            self.validation_table.setItem(r, 4, QTableWidgetItem(str(row["quality_reason_codes"])))
            self.validation_table.setItem(r, 5, QTableWidgetItem(str(row["quality_reason_detail"])))

    def on_analysis_clicked(self):
        try:
            if not self.data_engine.file_path or not os.path.exists(self.data_engine.file_path):
                self.status_label.setText("상태: 분석할 데이터 파일이 없습니다. 파일을 선택해 주세요.")
                return
                
            self.status_label.setText("상태: 상관관계 계산 중...")
            
            # Include more features for a complete analysis
            cols = self.data_engine.feature_cols + self.data_engine.target_cols
            df_filtered = self.data_engine.df[cols].select_dtypes(include=[np.number])
            corr = df_filtered.corr()
            
            self.last_corr = corr
            
            # Show canvas and hide placeholder
            self.analysis_placeholder.setVisible(False)
            self.analysis_canvas.setVisible(True)
            
            self.draw_analysis_heatmap()
            
            # Hide the button after first generation as requested
            self.analysis_btn.setVisible(False)
            self.status_label.setText("상태: 상관관계 분석 완료 (데이터 시각화)")
        except Exception as e:
            self.status_label.setText(f"상태: 분석 오류 - {str(e)}")

    def draw_analysis_heatmap(self):
        if self.last_corr is None:
            return
            
        # Decide whether to show numbers based on window width
        show_annot = True if self.width() > 1000 else False
        
        self.analysis_canvas.axes.clear()
        sns.heatmap(self.last_corr, annot=show_annot, cmap='RdBu_r', ax=self.analysis_canvas.axes, 
                    cbar=True, center=0, fmt=".1f", annot_kws={"size": 7})
        
        title_size = 14 if self.width() > 1000 else 10
        self.analysis_canvas.axes.set_title("재료 데이터 변수 간 상관성 분석", pad=20, fontsize=title_size, fontweight='bold')
        self.analysis_canvas.axes.set_xticklabels(self.analysis_canvas.axes.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        self.analysis_canvas.axes.set_yticklabels(self.analysis_canvas.axes.get_yticklabels(), fontsize=8)
        
        self.analysis_canvas.fig.tight_layout(pad=2.0)
        self.analysis_canvas.draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-draw heatmap with/without numbers only when tab is visible or to be responsive
        if hasattr(self, 'last_corr') and self.last_corr is not None:
            # We use a simple debounce/check if needed, but for now just re-draw
            # Optimization: only redraw if the 'show_annot' state would change
            current_state = True if self.width() > 1000 else False
            if not hasattr(self, '_last_annot_state') or self._last_annot_state != current_state:
                self._last_annot_state = current_state
                self.draw_analysis_heatmap()

    def on_select_file_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Data File", "", "Excel Files (*.xls *.xlsx)")
        if file_path:
            self.data_engine.set_file_path(file_path)
            self.file_path_label.setText(f"File: {os.path.basename(file_path)}")
            self.status_label.setText("Status: New file selected. Ready to train.")
            self.refresh_validation_tab()

    def load_existing_model(self):
        if os.path.exists("models/data_engine.pkl") and os.path.exists("models/material_model.pkl"):
            try:
                self.data_engine = joblib.load("models/data_engine.pkl")
                self.model_engine = ModelEngine()
                self.model_engine.load("models/material_model.pkl")
                self.model_type = self.model_engine.model_type
                self.status_label.setText(f"상태: 기존 모델 로드 완료 ({self.model_type})")
                self.update_active_model_display()
            except Exception as e:
                self.status_label.setText(f"Status: Model Load Error - {str(e)}")

    def on_train_clicked(self):
        if not self.data_engine.file_path or not os.path.exists(self.data_engine.file_path):
            self.status_label.setText("상태: 오류 - 데이터를 먼저 선택해 주세요!")
            return
            
        self.train_btn.setEnabled(False)
        self.status_label.setText("상태: 학습 준비 중...")
        # Clear previous metrics to show that new training is happening
        self.metrics_label.setText("<b>모델 성능 요약:</b><br>- 계산 중...")
        
        model_map = {0: 'RF', 1: 'GBM', 2: 'MLP', 3: 'TFP'}
        self.model_type = model_map.get(self.model_combo.currentIndex(), 'RF')
        max_iter = self.iter_spin.value()
        
        self.thread = TrainingThread(self.data_engine, model_type=self.model_type, max_iter=max_iter)
        self.thread.progress.connect(lambda s: self.status_label.setText(f"상태: {s}"))
        self.thread.finished.connect(self.on_training_finished)
        self.thread.start()

    def on_training_finished(self, results):
        self.train_btn.setEnabled(True)
        if isinstance(results, str):
            self.status_label.setText(f"상태: 오류 발생 - {results}")
            self.refresh_validation_tab()
            return
            
        self.model_engine = results["model"]
        validation = results.get("validation_summary", {})
        self.status_label.setText(f"상태: {self.model_type} 학습 완료 및 저장됨")
        self.update_active_model_display()
        self.refresh_validation_tab()

        if validation and validation.get("rows_hold", 0) > 0:
            QMessageBox.warning(
                self,
                "데이터 정합성 경고",
                (
                    f"정합성 검증 경고 데이터가 있습니다.\n"
                    f"pass: {validation.get('rows_pass', 0)}\n"
                    f"hold: {validation.get('rows_hold', 0)}\n"
                    f"reject: {validation.get('rows_reject', 0)}\n\n"
                    f"hold 데이터는 리뷰 후 학습 반영 여부를 결정하는 것을 권장합니다."
                )
            )
        
        metrics = results["metrics"]
        # Simplify metrics for user
        r2_avg = np.mean(metrics['r2'])
        mae_avg = np.mean(metrics['mae'])
        
        acc_text = "매우 높음" if r2_avg > 0.9 else "높음" if r2_avg > 0.8 else "보통"
        
        self.metrics_label.setText(
            f"<b>종합 모델 성능 요약:</b><br>"
            f"- 평균 예측 정확도(R²): <b>{r2_avg*100:.1f}% ({acc_text})</b><br>"
            f"- 평균 오차(MAE): <b>{mae_avg:.2f} (물성별 상이)</b><br>"
            f"<font color='#7f8c8d' size='2'>* 강도(MPa) 및 연성(%)을 포함한 4개 지표 합산 결과입니다.</font>"
        )
        
        # Plot R2 scores for all 4 targets
        self.canvas.axes.clear()
        target_names = ['Yield Stress', 'UTS', 'Elongation', 'Area Red.']
        r2_scores = metrics['r2']
        
        # Color bars based on performance
        colors = ['#3498db' if r > 0.8 else '#f1c40f' if r > 0.6 else '#e74c3c' for r in r2_scores]
        
        bars = self.canvas.axes.bar(target_names, r2_scores, color=colors)
        self.canvas.axes.set_ylim(0, 1.1)
        self.canvas.axes.set_ylabel("신뢰도 (R² Score)")
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            self.canvas.axes.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                                
        name_map = {'RF': 'Random Forest', 'GBM': 'Gradient Boosting', 'MLP': 'Neural Network', 'TFP': 'Probabilistic Model'}
        display_name = name_map.get(self.model_type, self.model_type)
        self.canvas.axes.set_title(f"모델별 특성 예측 정확도 ({display_name})")
        self.canvas.draw()
        
        # New Performance Tab Visualization: Parity Plots
        self.perf_canvas.figure.clear()
        axes = self.perf_canvas.figure.subplots(2, 2)
        y_test = results["y_test"].values
        y_pred = results["y_pred"]
        target_names = ['Yield Stress (MPa)', 'UTS (MPa)', 'Elongation (%)', 'Area Reduction (%)']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, ax in enumerate(axes.flatten()):
            ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.5, color=colors[i], s=10)
            # Identity line
            all_data = np.concatenate([y_test[:, i], y_pred[:, i]])
            min_val, max_val = all_data.min(), all_data.max()
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, lw=1)
            
            ax.set_title(target_names[i], fontsize=10, fontweight='bold')
            ax.set_xlabel("실제값 (Actual)", fontsize=8)
            ax.set_ylabel("예측값 (Predicted)", fontsize=8)
            ax.grid(True, linestyle=':', alpha=0.6)
            
        self.perf_canvas.figure.tight_layout()
        self.perf_canvas.draw()

    def on_predict_clicked(self):
        if not self.model_engine:
            self.result_display.setText("Please train or load the model first!")
            return
            
        try:
            input_dict = {k: v.text() for k, v in self.inputs.items()}
            scaled_input = self.data_engine.get_inference_data(input_dict)
            
            # Predict
            mean_scaled, std_scaled = self.model_engine.predict(scaled_input.astype(np.float32))
            
            mean = self.data_engine.scaler_y.inverse_transform(mean_scaled)[0]
            # Uncertainty scaling using scaler's std
            std = std_scaled[0] * self.data_engine.scaler_y.scale_
            
            res_text = (f"<b>[강도 예측 결과]</b><br>"
                       f"0.2% Yield Stress: <b>{mean[0]:.1f} ± {std[0]:.1f} MPa</b><br>"
                       f"UTS: <b>{mean[1]:.1f} ± {std[1]:.1f} MPa</b><br><br>"
                       f"<b>[연성 예측 결과]</b><br>"
                       f"Elongation: <b>{mean[2]:.1f} ± {std[2]:.1f} %</b><br>"
                       f"Area Reduction: <b>{mean[3]:.1f} ± {std[3]:.1f} %</b><br><br>"
                       f"<i>* 신뢰도: 95% 신뢰구간 정보 포함.</i>")
            self.result_display.setText(res_text)
            
            # Plot with dual Y-axes
            self.prediction_canvas.axes.clear()
            labels = ['Yield', 'UTS', 'Elong.', 'Area Red.']
            x = np.arange(len(labels))
            
            # Stress bars (primary y)
            stress_vals = [mean[0], mean[1], 0, 0]
            stress_errs = [1.96 * std[0], 1.96 * std[1], 0, 0]
            self.prediction_canvas.axes.bar(x[:2], stress_vals[:2], yerr=stress_errs[:2], 
                                          capsize=10, color=['#3498db', '#e74c3c'], label='Stress (MPa)')
            self.prediction_canvas.axes.set_ylabel("Stress (MPa)", color='#3498db')
            
            # Ductility bars (secondary y)
            ax2 = self.prediction_canvas.axes.twinx()
            ax2.clear()
            duct_vals = [0, 0, mean[2], mean[3]]
            duct_errs = [0, 0, 1.96 * std[2], 1.96 * std[3]]
            ax2.bar(x[2:], duct_vals[2:], yerr=duct_errs[2:], 
                   capsize=10, color=['#2ecc71', '#f39c12'], label='Ductility (%)')
            ax2.set_ylabel("Percentage (%)", color='#2ecc71')
            
            self.prediction_canvas.axes.set_xticks(x)
            self.prediction_canvas.axes.set_xticklabels(labels)
            
            name_map = {'RF': 'Random Forest', 'GBM': 'Gradient Boosting', 'MLP': 'Neural Network', 'TFP': 'Probabilistic (Ensemble)'}
            display_name = name_map.get(self.model_type, self.model_type)
            self.prediction_canvas.axes.set_title(f"Predicted Properties: {display_name}")
            self.prediction_canvas.draw()
            
        except Exception as e:
            self.result_display.setText(f"Error during prediction: {str(e)}")

    def update_active_model_display(self):
        name_map = {'RF': 'Random Forest (안정적)', 'GBM': 'Gradient Boosting (정확함)', 'MLP': 'Neural Network (일반)', 'TFP': 'Bootstrap Ensemble (확률론적)'}
        display_name = name_map.get(self.model_type, self.model_type)
        self.active_model_info.setText(f"현재 예측 모델: {display_name}")
        self.active_model_info.setStyleSheet("background-color: #d4efdf; padding: 10px; border: 1px solid #27ae60; border-radius: 5px; font-weight: bold; margin-bottom: 10px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
