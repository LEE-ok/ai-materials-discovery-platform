import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.engine.data_engine import DataEngine
from src.engine.model_engine import ModelEngine


try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


class TrainingThread(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(self, data_engine, model_type="RF", max_iter=2000):
        super().__init__()
        self.data_engine = data_engine
        self.model_type = model_type
        self.max_iter = max_iter

    def run(self):
        try:
            self.progress.emit("데이터를 다시 불러오고 전처리하는 중입니다.")
            self.data_engine.load_data()
            self.progress.emit(self.data_engine.format_quality_report())

            X_train, X_test, y_train, _, _, y_raw_test = self.data_engine.preprocess_data()
            if len(X_train) == 0:
                self.finished.emit("전처리 후 학습 가능한 데이터가 없습니다.")
                return

            self.progress.emit(f"{self.model_type} 모델을 초기화하는 중입니다.")
            model_engine = ModelEngine(
                model_type=self.model_type,
                output_dim=y_train.shape[1],
                max_iter=self.max_iter,
            )

            self.progress.emit(f"{self.model_type} 모델을 학습하는 중입니다.")
            model_engine.train(X_train, y_train)

            if not os.path.exists("models"):
                os.makedirs("models")
            model_engine.save("models/material_model.pkl")
            joblib.dump(self.data_engine, "models/data_engine.pkl")

            self.progress.emit("학습 결과를 평가하는 중입니다.")
            mean_scaled, _ = model_engine.predict(X_test)
            y_pred = self.data_engine.inverse_transform_y(mean_scaled)

            from sklearn.metrics import mean_absolute_error, r2_score

            r2 = r2_score(y_raw_test, y_pred, multioutput="raw_values")
            mae = mean_absolute_error(y_raw_test, y_pred, multioutput="raw_values")

            self.finished.emit(
                {
                    "model": model_engine,
                    "metrics": {"r2": r2, "mae": mae},
                    "y_test": y_raw_test,
                    "y_pred": y_pred,
                    "quality_report": self.data_engine.last_quality_report,
                }
            )
        except Exception as exc:
            self.finished.emit(f"학습 중 오류가 발생했습니다: {exc}")


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Materials Discovery Platform")
        self.resize(1200, 820)

        self.data_engine = DataEngine(None)
        self.model_engine = None
        self.model_type = "RF"
        self.preprocessing_ready = False

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)

        header = QLabel("AI Materials Discovery Platform")
        header.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #2c3e50; margin: 10px;")
        root_layout.addWidget(header)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        self.setup_preprocessing_tab()
        self.setup_training_tab()
        self.setup_performance_tab()
        self.setup_inference_tab()

    def setup_preprocessing_tab(self):
        tab = QWidget()
        outer_layout = QVBoxLayout(tab)

        top_row = QHBoxLayout()
        top_row.addStretch()
        quality_help_btn = QPushButton("전처리 도움말")
        quality_help_btn.setFixedWidth(120)
        quality_help_btn.clicked.connect(self.show_quality_help)
        top_row.addWidget(quality_help_btn)
        outer_layout.addLayout(top_row)

        content_layout = QHBoxLayout()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(420)
        left_scroll.setMaximumWidth(500)

        left_widget = QWidget()
        left_panel = QVBoxLayout(left_widget)
        left_panel.setSpacing(12)

        file_group = QGroupBox("1. 데이터 파일 선택")
        file_layout = QVBoxLayout(file_group)

        self.file_path_label = QLabel("파일: 선택되지 않음")
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setStyleSheet("font-size: 11px; color: #7f8c8d;")
        file_layout.addWidget(self.file_path_label)

        self.select_file_btn = QPushButton("데이터 파일 선택 (.xls/.xlsx)")
        self.select_file_btn.clicked.connect(self.on_select_file_clicked)
        file_layout.addWidget(self.select_file_btn)

        self.status_label = QLabel("상태: 학습용 데이터를 선택해 주세요.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #34495e; padding: 2px 0 6px 0;")
        file_layout.addWidget(self.status_label)
        left_panel.addWidget(file_group)

        domain_group = QGroupBox("2. 도메인 검증 기준")
        domain_layout = QVBoxLayout(domain_group)
        self.domain_rule_label = QLabel(
            "도메인 검증은 '오스테나이트 조성 기준'과 '고온 특성 기준' 두 부류로 나누어 범위를 확인합니다."
        )
        self.domain_rule_label.setWordWrap(True)
        self.domain_rule_label.setStyleSheet(
            "background-color: #fff8e8; padding: 10px; border-radius: 8px; color: #6b4f00;"
        )
        domain_layout.addWidget(self.domain_rule_label)

        domain_button_row = QHBoxLayout()
        self.austenite_domain_btn = QPushButton("오스테나이트 조성 기준")
        self.austenite_domain_btn.setFixedHeight(36)
        self.austenite_domain_btn.clicked.connect(self.show_austenite_domain_dialog)
        domain_button_row.addWidget(self.austenite_domain_btn)
        self.high_temp_domain_btn = QPushButton("고온 특성 기준")
        self.high_temp_domain_btn.setFixedHeight(36)
        self.high_temp_domain_btn.clicked.connect(self.show_high_temp_domain_dialog)
        domain_button_row.addWidget(self.high_temp_domain_btn)
        domain_button_row.addStretch()
        domain_layout.addLayout(domain_button_row)

        self.domain_range_status_label = QLabel("")
        self.domain_range_status_label.setWordWrap(True)
        self.domain_range_status_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        domain_layout.addWidget(self.domain_range_status_label)
        left_panel.addWidget(domain_group)
        self.refresh_domain_range_status()

        quality_group = QGroupBox("3. 데이터 품질 처리 설정")
        quality_layout = QFormLayout(quality_group)

        self.missing_combo = QComboBox()
        self.missing_combo.addItems(
            ["평균값으로 채우기", "중앙값으로 채우기", "주변 값으로 예측(KNN)", "해당 행 제거"]
        )
        quality_layout.addRow("누락값 처리:", self.missing_combo)

        self.outlier_combo = QComboBox()
        self.outlier_combo.addItems(["감지 범위로 보정", "이상치 행 제거", "표시만 하고 유지"])
        quality_layout.addRow("이상치 처리:", self.outlier_combo)

        self.invalid_type_combo = QComboBox()
        self.invalid_type_combo.addItems(["잘못된 값을 NaN으로 변환", "잘못된 값이 있는 행 제거"])
        quality_layout.addRow("형식 검증:", self.invalid_type_combo)

        self.iqr_spin = QDoubleSpinBox()
        self.iqr_spin.setRange(0.5, 5.0)
        self.iqr_spin.setSingleStep(0.1)
        self.iqr_spin.setValue(1.5)
        quality_layout.addRow("이상치 민감도:", self.iqr_spin)
        left_panel.addWidget(quality_group)

        feature_group = QGroupBox("4. 합금 지표 생성 전처리")
        feature_layout = QVBoxLayout(feature_group)
        self.feature_engineering_check = QCheckBox("합금 지표 생성 사용")
        self.feature_engineering_check.setChecked(True)
        self.feature_engineering_check.setVisible(False)
        feature_layout.addWidget(self.feature_engineering_check)

        self.feature_engineering_label = QLabel(
            "1차 전처리 후 2차 전처리로 Cr/Ni, C+N, Ni_eq, Cr_eq를 생성합니다."
        )
        self.feature_engineering_label.setWordWrap(True)
        self.feature_engineering_label.setStyleSheet(
            "background-color: #eef6ff; padding: 10px; border-radius: 8px; color: #355c7d;"
        )
        feature_layout.addWidget(self.feature_engineering_label)
        left_panel.addWidget(feature_group)

        self.preprocess_btn = QPushButton("전처리 실행")
        self.preprocess_btn.setFixedHeight(45)
        self.preprocess_btn.setStyleSheet(
            "background-color: #2d8cff; color: white; font-weight: bold; font-size: 13px; border-radius: 8px;"
        )
        self.preprocess_btn.clicked.connect(self.on_preprocess_clicked)
        left_panel.addWidget(self.preprocess_btn)

        self.generate_features_btn = QPushButton("합금 지표 생성")
        self.generate_features_btn.setFixedHeight(42)
        self.generate_features_btn.setEnabled(False)
        self.generate_features_btn.setStyleSheet(
            "background-color: #16a085; color: white; font-weight: bold; font-size: 12px; border-radius: 8px;"
        )
        self.generate_features_btn.clicked.connect(self.on_generate_features_clicked)
        left_panel.addWidget(self.generate_features_btn)

        self.reset_preprocess_btn = QPushButton("전처리 결과 초기화")
        self.reset_preprocess_btn.setFixedHeight(40)
        self.reset_preprocess_btn.setStyleSheet(
            "background-color: #ecf0f1; color: #2c3e50; font-weight: bold; font-size: 12px; border-radius: 8px;"
        )
        self.reset_preprocess_btn.clicked.connect(self.on_reset_preprocessing_clicked)
        left_panel.addWidget(self.reset_preprocess_btn)

        self.go_to_training_btn = QPushButton("2번 탭에서 모델 학습하기")
        self.go_to_training_btn.setFixedHeight(42)
        self.go_to_training_btn.setEnabled(False)
        self.go_to_training_btn.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        left_panel.addWidget(self.go_to_training_btn)

        self.quality_summary_label = QLabel("전처리 결과 요약이 아직 없습니다.")
        self.quality_summary_label.setWordWrap(True)
        self.quality_summary_label.setStyleSheet(
            "background-color: #f6f8fa; padding: 12px; border-radius: 8px; color: #34495e; border: 1px solid #e2e8f0;"
        )
        left_panel.addWidget(self.quality_summary_label)
        left_panel.addStretch()
        left_scroll.setWidget(left_widget)

        right_panel = QVBoxLayout()
        preview_group = QGroupBox("전처리 완료 데이터 확인")
        preview_layout = QVBoxLayout(preview_group)

        self.processed_preview_info_label = QLabel(
            "전처리를 실행하면 처리 완료된 전체 데이터를 아래 표에서 확인할 수 있습니다."
        )
        self.processed_preview_info_label.setWordWrap(True)
        self.processed_preview_info_label.setStyleSheet("color: #5d6d7e; padding-bottom: 6px;")
        preview_layout.addWidget(self.processed_preview_info_label)

        self.processed_result_tabs = QTabWidget()

        self.processed_preview_table = QTableWidget()
        self.processed_preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.processed_preview_table.setAlternatingRowColors(True)
        self.processed_preview_table.verticalHeader().setVisible(False)
        self.processed_preview_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.processed_result_tabs.addTab(self.processed_preview_table, "데이터 전처리 결과")

        self.engineered_preview_table = QTableWidget()
        self.engineered_preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.engineered_preview_table.setAlternatingRowColors(True)
        self.engineered_preview_table.verticalHeader().setVisible(False)
        self.engineered_preview_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.processed_result_tabs.addTab(self.engineered_preview_table, "합금 지표 생성 결과")

        preview_layout.addWidget(self.processed_result_tabs)

        preview_note = QLabel("전처리 완료 후 전체 결과가 표에 표시됩니다.")
        preview_note.setStyleSheet("font-size: 11px; color: #7f8c8d;")
        preview_layout.addWidget(preview_note)
        right_panel.addWidget(preview_group)

        content_layout.addWidget(left_scroll, 1)
        content_layout.addLayout(right_panel, 2)
        outer_layout.addLayout(content_layout)
        self.tabs.addTab(tab, "데이터 전처리")

        self.missing_combo.currentIndexChanged.connect(self.mark_preprocessing_dirty)
        self.outlier_combo.currentIndexChanged.connect(self.mark_preprocessing_dirty)
        self.invalid_type_combo.currentIndexChanged.connect(self.mark_preprocessing_dirty)
        self.iqr_spin.valueChanged.connect(self.mark_preprocessing_dirty)
        self.feature_engineering_check.stateChanged.connect(self.mark_preprocessing_dirty)

    def setup_training_tab(self):
        tab = QWidget()
        outer_layout = QVBoxLayout(tab)

        top_row = QHBoxLayout()
        top_row.addStretch()
        model_help_btn = QPushButton("모델 학습 도움말")
        model_help_btn.setFixedWidth(140)
        model_help_btn.clicked.connect(self.show_model_training_help)
        top_row.addWidget(model_help_btn)
        outer_layout.addLayout(top_row)

        content_layout = QHBoxLayout()
        left_panel = QVBoxLayout()

        info_group = QGroupBox("2. 모델 학습")
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(10)

        self.training_data_status_label = QLabel("")
        self.training_data_status_label.setWordWrap(True)
        self.training_data_status_label.setStyleSheet(
            "background-color: #fff7d6; padding: 10px; border-radius: 8px; color: #7a5d00;"
        )
        info_layout.addWidget(self.training_data_status_label)

        model_selection_group = QGroupBox("AI 모델 학습 설정")
        model_form = QFormLayout(model_selection_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems(
            ["Random Forest", "Gradient Boosting", "Neural Network", "TFP"]
        )
        model_form.addRow("학습 모델:", self.model_combo)

        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(100, 10000)
        self.iter_spin.setValue(2000)
        self.iter_spin.setSingleStep(500)
        model_form.addRow("최대 반복 횟수:", self.iter_spin)
        self.training_input_combo = QComboBox()
        self.training_input_combo.addItems(
            ["데이터 정제 + 합금 지표", "데이터 정제만", "합금 지표만"]
        )
        model_form.addRow("학습 데이터 선택:", self.training_input_combo)
        info_layout.addWidget(model_selection_group)

        help_label = QLabel(
            "반복 횟수를 크게 늘린다고 항상 성능이 좋아지지는 않습니다.\n기본값으로 먼저 학습한 뒤 필요할 때만 조정하는 것을 권장합니다."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet(
            "font-family: 'Malgun Gothic'; font-size: 10px; color: #3498db; font-style: italic; line-height: 1.3; background-color: #eef6ff; padding: 8px; border-radius: 6px;"
        )
        info_layout.addWidget(help_label)

        self.training_status_label = QLabel("")
        self.training_status_label.setWordWrap(True)
        self.training_status_label.setStyleSheet("color: #34495e; padding: 2px 0 6px 0;")
        info_layout.addWidget(self.training_status_label)

        self.train_btn = QPushButton("모델 학습 시작")
        self.train_btn.setFixedHeight(45)
        self.train_btn.setEnabled(False)
        self.train_btn.setStyleSheet(
            "background-color: #3498db; color: white; font-weight: bold; font-size: 13px; border-radius: 8px;"
        )
        self.train_btn.clicked.connect(self.on_train_clicked)
        info_layout.addWidget(self.train_btn)

        self.metrics_label = QLabel("<b>모델 성능 요약:</b><br>- 예측 정확도: N/A<br>- 평균 오차: N/A")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet(
            "background-color: #f8f9fa; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;"
        )
        info_layout.addWidget(self.metrics_label)

        left_panel.addWidget(info_group)
        left_panel.addStretch()

        right_panel = QVBoxLayout()
        self.canvas = MplCanvas(self, width=6.6, height=5.2, dpi=100)
        right_panel.addWidget(self.canvas)
        self.render_training_placeholder()

        content_layout.addLayout(left_panel, 1)
        content_layout.addLayout(right_panel, 2)
        outer_layout.addLayout(content_layout)
        self.tabs.addTab(tab, "모델 학습")

    def setup_performance_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        header = QLabel("상세 성능 분석 (Predicted vs Actual)")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        self.perf_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        layout.addWidget(self.perf_canvas)

        desc = QLabel("* 학습이 끝나면 실제값과 예측값 비교 그래프가 여기에 표시됩니다.")
        desc.setStyleSheet("color: #7f8c8d; font-style: italic;")
        layout.addWidget(desc)

        self.tabs.addTab(tab, "상세 성능 분석")
        self.render_performance_placeholder()

    def setup_inference_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        left_widget = QWidget()
        left_panel = QVBoxLayout(left_widget)
        scroll.setWidget(left_widget)

        self.inputs = {}
        self.active_model_info = QLabel("사용 중인 모델: 학습된 모델 없음")
        self.active_model_info.setStyleSheet(
            "background-color: #fcf3cf; padding: 10px; border: 1px solid #f39c12; border-radius: 5px; font-weight: bold; margin-bottom: 10px;"
        )
        left_panel.addWidget(self.active_model_info)

        comp_group = QGroupBox("Chemical Composition (wt%)")
        comp_layout = QFormLayout(comp_group)
        comp_list = [
            "Cr", "Ni", "Mo", "Mn", "Si", "Nb", "Ti", "Zr", "Ta", "V", "W", "Cu", "N", "C", "B", "P", "S", "Co", "Al", "Sn", "Pb",
        ]
        default_map = {"Cr": "18.0", "Ni": "8.0", "Mn": "2.0", "Si": "1.0", "C": "0.08"}
        for col in comp_list:
            line_edit = QLineEdit()
            line_edit.setText(default_map.get(col, "0.0"))
            comp_layout.addRow(QLabel(col), line_edit)
            self.inputs[col] = line_edit
        left_panel.addWidget(comp_group)

        proc_group = QGroupBox("Process & Structure")
        proc_layout = QFormLayout(proc_group)
        proc_defaults = {
            "Solution_treatment_temperature": "1050",
            "Solution_treatment_time(s)": "3600",
            "Water_Quenched_after_s.t.": "1",
            "Air_Quenched_after_s.t.": "0",
            "Grains mm-2": "500",
            "Type of melting": "2",
            "Size of ingot": "50",
            "Product form": "3",
            "Temperature (K)": "300",
        }
        for col, value in proc_defaults.items():
            line_edit = QLineEdit()
            line_edit.setText(value)
            proc_layout.addRow(QLabel(col), line_edit)
            self.inputs[col] = line_edit
        left_panel.addWidget(proc_group)

        predict_btn = QPushButton("Predict Properties")
        predict_btn.setFixedHeight(45)
        predict_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; margin-top: 10px;")
        predict_btn.clicked.connect(self.on_predict_clicked)
        left_panel.addWidget(predict_btn)
        left_panel.addStretch()

        right_panel = QVBoxLayout()
        result_group = QGroupBox("Predictions")
        result_layout = QVBoxLayout(result_group)

        self.result_display = QLabel("Enter parameters and click Predict.")
        self.result_display.setFont(QFont("Arial", 12))
        result_layout.addWidget(self.result_display)

        self.prediction_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        result_layout.addWidget(self.prediction_canvas)
        right_panel.addWidget(result_group)

        layout.addWidget(scroll, 1)
        layout.addLayout(right_panel, 1)
        self.tabs.addTab(tab, "물성 예측")

    def refresh_domain_range_status(self):
        custom_count = len(getattr(self.data_engine, "custom_ranges", {}))
        total_count = len(self.data_engine.get_domain_ranges())
        if custom_count:
            self.domain_range_status_label.setText(
                f"현재 사용자 지정 기준 {custom_count}개가 적용되어 있습니다. 검증 대상 컬럼은 총 {total_count}개입니다."
            )
        else:
            self.domain_range_status_label.setText(
                f"현재 기본 도메인 기준으로 검증합니다. 검증 대상 컬럼은 총 {total_count}개입니다."
            )

    def mark_preprocessing_dirty(self, *_args):
        self.preprocessing_ready = False
        self.train_btn.setEnabled(False)
        self.go_to_training_btn.setEnabled(False)
        if hasattr(self, "generate_features_btn"):
            self.generate_features_btn.setEnabled(False)
        if self.data_engine.df is not None and not self.data_engine.df.empty:
            self.training_data_status_label.setText("현재 표시 중인 결과는 이전 설정 기준입니다. 새 설정으로 다시 전처리를 실행해 주세요.")
            self.training_status_label.setText("상태: 설정이 변경되어 학습이 잠시 비활성화되었습니다. 전처리를 다시 실행해 주세요.")
            self.quality_summary_label.setText("전처리 설정이 변경되었습니다. 현재 표는 이전 설정 기준 결과입니다.")
            self.processed_preview_info_label.setText("현재 표는 이전 전처리 결과입니다. 새 설정을 반영하려면 전처리를 다시 실행해 주세요.")
            return
        self.training_data_status_label.setText("")
        self.training_status_label.setText("")
        self.quality_summary_label.setText("전처리 설정이 변경되었습니다. 전처리를 다시 실행해 주세요.")
        self.processed_preview_info_label.setText("설정이 변경되었고 아직 전처리 결과가 없습니다. 전처리를 다시 실행해 주세요.")

    def apply_quality_settings_from_ui(self):
        missing_map = {0: "mean", 1: "median", 2: "knn", 3: "drop"}
        outlier_map = {0: "clip", 1: "remove", 2: "flag"}
        invalid_type_map = {0: "coerce", 1: "drop"}
        input_feature_mode_map = {
            0: "combined",
            1: "clean_only",
            2: "engineered_only",
        }
        self.data_engine.configure_quality_rules(
            missing_strategy=missing_map.get(self.missing_combo.currentIndex(), "mean"),
            outlier_strategy=outlier_map.get(self.outlier_combo.currentIndex(), "clip"),
            invalid_type_strategy=invalid_type_map.get(self.invalid_type_combo.currentIndex(), "coerce"),
            iqr_factor=self.iqr_spin.value(),
            input_feature_mode=input_feature_mode_map.get(self.training_input_combo.currentIndex(), "combined"),
        )

    def update_quality_summary_from_report(self, report):
        if not report:
            self.quality_summary_label.setText("전처리 결과 요약이 아직 없습니다.")
            self.domain_rule_label.setText("도메인 기준 검증 결과가 아직 없습니다.")
            return

        self.quality_summary_label.setText(
            "데이터 품질 처리 결과: "
            f"행 {report.get('rows_before', 0)} -> {report.get('rows_after', 0)}, "
            f"형식 오류 {report.get('invalid_type_cells', 0)}개, "
            f"누락값 {report.get('missing_cells_before', 0)} -> {report.get('missing_cells_after', 0)}, "
            f"이상치 감지 {report.get('outlier_cells', 0)}개, "
            f"합금 지표 생성 {len(report.get('engineered_features_added', []))}개"
        )
        self.domain_rule_label.setText(
            "도메인 기준 검증 결과: "
            f"{report.get('domain_range_cells', 0)}개의 값이 설정 범위를 벗어났습니다."
        )

    def populate_processed_preview(self, df):
        base_df = self.data_engine.get_preprocessed_display_df()
        engineered_df = self.data_engine.get_engineered_display_df()
        self.processed_preview_table.clear()
        self.processed_preview_table.setRowCount(len(base_df))
        self.processed_preview_table.setColumnCount(len(base_df.columns))
        self.processed_preview_table.setHorizontalHeaderLabels([str(col) for col in base_df.columns])

        for row_index, (_, row) in enumerate(base_df.iterrows()):
            for col_index, value in enumerate(row):
                if pd.isna(value):
                    text = ""
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    text = f"{float(value):.4g}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.processed_preview_table.setItem(row_index, col_index, item)

        self.processed_preview_table.resizeColumnsToContents()
        self.engineered_preview_table.clear()
        self.engineered_preview_table.setRowCount(len(engineered_df))
        self.engineered_preview_table.setColumnCount(len(engineered_df.columns))
        self.engineered_preview_table.setHorizontalHeaderLabels([str(col) for col in engineered_df.columns])

        for row_index, (_, row) in enumerate(engineered_df.iterrows()):
            for col_index, value in enumerate(row):
                if pd.isna(value):
                    text = ""
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    text = f"{float(value):.4g}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.engineered_preview_table.setItem(row_index, col_index, item)

        self.engineered_preview_table.resizeColumnsToContents()
        self.processed_preview_info_label.setText(
            f"전처리 완료 데이터 전체 결과: 총 {len(df)}행 표시"
        )

    def render_training_placeholder(self):
        self.canvas.axes.clear()
        self.canvas.axes.axis("off")
        self.canvas.axes.text(0.5, 0.58, "2번 탭에서 모델 학습 결과가 여기에 표시됩니다.", ha="center", va="center", fontsize=14, color="#5d6d7e", transform=self.canvas.axes.transAxes)
        self.canvas.axes.text(0.5, 0.42, "먼저 1번 탭에서 전처리를 실행해 주세요.", ha="center", va="center", fontsize=11, color="#85929e", transform=self.canvas.axes.transAxes)
        self.canvas.draw()

    def render_performance_placeholder(self):
        self.perf_canvas.figure.clear()
        ax = self.perf_canvas.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.56, "모델 학습이 끝나면 상세 성능 분석 그래프가 여기에 표시됩니다.", ha="center", va="center", fontsize=14, color="#5d6d7e", transform=ax.transAxes)
        ax.text(0.5, 0.40, "실제값과 예측값이 얼마나 비슷한지 특성별로 확인할 수 있습니다.", ha="center", va="center", fontsize=11, color="#85929e", transform=ax.transAxes)
        self.perf_canvas.figure.tight_layout()
        self.perf_canvas.draw()

    def reset_preprocessing_state(self, keep_file=True):
        current_file = self.data_engine.file_path if keep_file else None
        self.preprocessing_ready = False
        self.data_engine.df = None
        self.data_engine.last_quality_report = {}
        if not keep_file:
            self.data_engine.set_file_path(None)

        self.train_btn.setEnabled(False)
        self.go_to_training_btn.setEnabled(False)
        if hasattr(self, "generate_features_btn"):
            self.generate_features_btn.setEnabled(False)
        self.metrics_label.setText("<b>모델 성능 요약:</b><br>- 예측 정확도: N/A<br>- 평균 오차: N/A")
        self.training_data_status_label.setText("")
        self.training_status_label.setText("")
        self.quality_summary_label.setText("전처리 결과 요약이 아직 없습니다.")
        self.domain_rule_label.setText("도메인 검증은 '오스테나이트 조성 기준'과 '고온 특성 기준' 두 부류로 나누어 범위를 확인합니다.")
        self.processed_preview_info_label.setText("전처리를 실행하면 처리 완료된 전체 데이터를 아래 표에서 확인할 수 있습니다.")
        self.processed_preview_table.clear()
        self.processed_preview_table.setRowCount(0)
        self.processed_preview_table.setColumnCount(0)
        self.engineered_preview_table.clear()
        self.engineered_preview_table.setRowCount(0)
        self.engineered_preview_table.setColumnCount(0)
        self.render_training_placeholder()
        self.render_performance_placeholder()

        if keep_file and current_file:
            self.file_path_label.setText(f"파일: {os.path.basename(current_file)}")
            self.status_label.setText("상태: 전처리 설정을 초기화했습니다. 다시 전처리를 실행해 주세요.")
        else:
            self.file_path_label.setText("파일: 선택되지 않음")
            self.status_label.setText("상태: 학습용 데이터를 선택해 주세요.")

    def on_reset_preprocessing_clicked(self):
        self.missing_combo.setCurrentIndex(0)
        self.outlier_combo.setCurrentIndex(0)
        self.invalid_type_combo.setCurrentIndex(0)
        self.iqr_spin.setValue(1.5)
        self.feature_engineering_check.blockSignals(True)
        self.feature_engineering_check.setChecked(False)
        self.feature_engineering_check.blockSignals(False)
        if hasattr(self, "training_input_combo"):
            self.training_input_combo.setCurrentIndex(0)
        self.reset_preprocessing_state(keep_file=True)

    def on_select_file_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "데이터 파일 열기", "", "Excel Files (*.xls *.xlsx)")
        if not file_path:
            return

        self.data_engine.set_file_path(file_path)
        self.data_engine.df = None
        self.data_engine.last_quality_report = {}
        self.file_path_label.setText(f"파일: {os.path.basename(file_path)}")
        self.status_label.setText("상태: 새 데이터 파일이 선택되었습니다. 1번 탭에서 전처리를 실행해 주세요.")
        self.quality_summary_label.setText("전처리 결과 요약이 아직 없습니다.")
        self.domain_rule_label.setText("조성, 열처리 온도, 시간, 이진값 등에 대해 '말이 되는 범위'를 확인합니다.")
        self.processed_preview_info_label.setText("전처리를 실행하면 처리 완료된 전체 데이터를 아래 표에서 확인할 수 있습니다.")
        self.processed_preview_table.clear()
        self.processed_preview_table.setRowCount(0)
        self.processed_preview_table.setColumnCount(0)
        self.metrics_label.setText("<b>모델 성능 요약:</b><br>- 예측 정확도: N/A<br>- 평균 오차: N/A")
        self.reset_preprocessing_state(keep_file=True)
        self.status_label.setText("상태: 새 데이터 파일이 선택되었습니다. 1번 탭에서 전처리를 실행해 주세요.")

    def on_preprocess_clicked(self):
        if not self.data_engine.file_path or not os.path.exists(self.data_engine.file_path):
            self.status_label.setText("상태: 오류 - 먼저 데이터 파일을 선택해 주세요.")
            return

        self.apply_quality_settings_from_ui()
        self.data_engine.quality_options["feature_engineering"] = False
        self.feature_engineering_check.blockSignals(True)
        self.feature_engineering_check.setChecked(False)
        self.feature_engineering_check.blockSignals(False)
        self.status_label.setText("상태: 1차 데이터 정제 전처리를 실행 중입니다.")
        self.preprocess_btn.setEnabled(False)
        self.processed_preview_info_label.setText("새 설정으로 전처리 중입니다. 완료되면 표가 갱신됩니다.")
        self.processed_preview_table.clear()
        self.processed_preview_table.setRowCount(0)
        self.processed_preview_table.setColumnCount(0)
        self.engineered_preview_table.clear()
        self.engineered_preview_table.setRowCount(0)
        self.engineered_preview_table.setColumnCount(0)

        try:
            processed_df = self.data_engine.load_data(include_engineered=False)
            self.update_quality_summary_from_report(self.data_engine.last_quality_report)
            self.populate_processed_preview(processed_df)
            self.preprocessing_ready = True
            self.status_label.setText("상태: 1차 데이터 정제 전처리가 완료되었습니다.")
            self.training_data_status_label.setText(f"전처리 완료 데이터 {len(processed_df)}행이 준비되었습니다. 이제 모델 학습을 시작할 수 있습니다.")
            self.training_status_label.setText("상태: 전처리 완료. 2번 탭에서 모델을 학습할 수 있습니다.")
            self.generate_features_btn.setEnabled(True)
            self.train_btn.setEnabled(True)
            self.go_to_training_btn.setEnabled(True)
        except Exception as exc:
            self.preprocessing_ready = False
            self.train_btn.setEnabled(False)
            self.go_to_training_btn.setEnabled(False)
            self.generate_features_btn.setEnabled(False)
            self.status_label.setText(f"상태: 전처리 오류 - {exc}")
        finally:
            self.preprocess_btn.setEnabled(True)

    def on_generate_features_clicked(self):
        try:
            if self.data_engine.df is None or self.data_engine.df.empty:
                self.status_label.setText("상태: 먼저 데이터 전처리를 실행해 주세요.")
                return

            self.status_label.setText("상태: 2차 합금 지표 생성 전처리를 실행 중입니다.")
            self.generate_features_btn.setEnabled(False)
            self.data_engine.generate_engineered_features_on_current_df()
            self.feature_engineering_check.blockSignals(True)
            self.feature_engineering_check.setChecked(True)
            self.feature_engineering_check.blockSignals(False)
            self.update_quality_summary_from_report(self.data_engine.last_quality_report)
            self.populate_processed_preview(self.data_engine.df)
            self.status_label.setText("상태: 합금 지표 생성이 완료되었습니다.")
            self.training_status_label.setText("상태: 합금 지표 생성 완료. 선택한 학습 데이터 방식으로 모델을 학습할 수 있습니다.")
        except Exception as exc:
            self.status_label.setText(f"상태: 합금 지표 생성 오류 - {exc}")
        finally:
            self.generate_features_btn.setEnabled(True)

    def show_austenite_domain_dialog(self):
        self.show_domain_range_dialog("오스테나이트 조성 기준")

    def show_high_temp_domain_dialog(self):
        self.show_domain_range_dialog("고온 특성 기준")

    def show_domain_range_dialog(self, group_filter=None):
        dialog = QDialog(self)
        dialog.setWindowTitle("도메인 기준 설정")
        dialog.resize(760, 620)

        layout = QVBoxLayout(dialog)
        intro = QLabel("각 컬럼의 최소값과 최대값을 직접 설정할 수 있습니다. SSINA 기준을 반영해 오스테나이트 조성 기준과 고온 특성 기준 두 부류로 나누어 표시합니다.")
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 13px; color: #34495e; padding-bottom: 4px;")
        layout.addWidget(intro)

        table = QTableWidget(dialog)
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["부류", "컬럼", "최소값", "최대값", "근거"])
        domain_ranges = self.data_engine.get_domain_ranges()
        sorted_columns = sorted(
            [
                col
                for col in domain_ranges.keys()
                if group_filter is None or self.data_engine.get_domain_group(col) == group_filter
            ],
            key=lambda col: (self.data_engine.get_domain_group(col), col),
        )
        table.setRowCount(len(sorted_columns))
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        for row, column in enumerate(sorted_columns):
            lower_bound, upper_bound = domain_ranges[column]
            group = self.data_engine.get_domain_group(column)
            basis = self.data_engine.get_domain_basis(column)
            source = "사용자 지정" if column in self.data_engine.custom_ranges else basis
            group_item = QTableWidgetItem(group)
            group_item.setFlags(group_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, group_item)
            name_item = QTableWidgetItem(column)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 1, name_item)
            table.setItem(row, 2, QTableWidgetItem("" if lower_bound is None else str(lower_bound)))
            table.setItem(row, 3, QTableWidgetItem("" if upper_bound is None else str(upper_bound)))
            source_item = QTableWidgetItem(source)
            source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 4, source_item)

        layout.addWidget(table, 1)

        button_row = QHBoxLayout()
        reset_btn = QPushButton("기본값으로 되돌리기")
        apply_btn = QPushButton("적용")
        close_btn = QPushButton("닫기")
        button_row.addWidget(reset_btn)
        button_row.addStretch()
        button_row.addWidget(apply_btn)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        def apply_ranges(reset_to_default=False):
            if reset_to_default:
                self.data_engine.reset_custom_domain_ranges()
                self.refresh_domain_range_status()
                self.domain_rule_label.setText("도메인 검증은 '오스테나이트 조성 기준'과 '고온 특성 기준' 두 부류로 나누어 범위를 확인합니다.")
                self.mark_preprocessing_dirty()
                dialog.accept()
                return

            custom_ranges = {}
            for row, column in enumerate(sorted_columns):
                min_item = table.item(row, 2)
                max_item = table.item(row, 3)
                min_text = min_item.text().strip() if min_item and min_item.text() else ""
                max_text = max_item.text().strip() if max_item and max_item.text() else ""

                try:
                    lower_bound = None if min_text == "" else float(min_text)
                    upper_bound = None if max_text == "" else float(max_text)
                except ValueError:
                    QMessageBox.warning(dialog, "입력 오류", f"{column} 값의 최소값 또는 최대값이 숫자가 아닙니다.")
                    return

                if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
                    QMessageBox.warning(dialog, "입력 오류", f"{column} 값에서 최소값이 최대값보다 큽니다.")
                    return

                default_bounds = self.data_engine.default_domain_ranges.get(column)
                current_bounds = (lower_bound, upper_bound)
                if default_bounds is None or current_bounds != default_bounds:
                    custom_ranges[column] = current_bounds

            self.data_engine.set_custom_domain_ranges(custom_ranges)
            self.refresh_domain_range_status()
            self.domain_rule_label.setText("도메인 검증은 '오스테나이트 조성 기준'과 '고온 특성 기준' 두 부류의 범위를 기준으로 비정상 값을 확인합니다.")
            self.mark_preprocessing_dirty()
            QMessageBox.information(dialog, "적용 완료", "도메인 기준이 저장되었습니다. 전처리를 다시 실행하면 반영됩니다.")
            dialog.accept()

        reset_btn.clicked.connect(lambda: apply_ranges(True))
        apply_btn.clicked.connect(apply_ranges)
        close_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def on_train_clicked(self):
        if not self.data_engine.file_path or not os.path.exists(self.data_engine.file_path):
            self.training_status_label.setText("상태: 오류 - 먼저 데이터 파일을 선택해 주세요.")
            return

        if not self.preprocessing_ready:
            self.training_status_label.setText("상태: 1번 탭에서 전처리를 먼저 실행해 주세요.")
            return

        self.apply_quality_settings_from_ui()
        self.train_btn.setEnabled(False)
        self.training_status_label.setText("상태: 학습 준비 중입니다.")
        self.metrics_label.setText("<b>모델 성능 요약:</b><br>- 계산 중...")

        model_map = {0: "RF", 1: "GBM", 2: "MLP", 3: "TFP"}
        self.model_type = model_map.get(self.model_combo.currentIndex(), "RF")

        self.thread = TrainingThread(self.data_engine, model_type=self.model_type, max_iter=self.iter_spin.value())
        self.thread.progress.connect(lambda text: self.training_status_label.setText(f"상태: {text}"))
        self.thread.finished.connect(self.on_training_finished)
        self.thread.start()

    def on_training_finished(self, results):
        self.train_btn.setEnabled(True)
        if isinstance(results, str):
            self.training_status_label.setText(f"상태: 오류 발생 - {results}")
            return

        self.model_engine = results["model"]
        self.training_status_label.setText(f"상태: {self.model_type} 모델 학습이 완료되었습니다.")
        self.update_active_model_display()
        self.update_quality_summary_from_report(results.get("quality_report", {}))

        metrics = results["metrics"]
        r2_avg = float(np.mean(metrics["r2"]))
        mae_avg = float(np.mean(metrics["mae"]))
        acc_text = "매우 높음" if r2_avg > 0.9 else "높음" if r2_avg > 0.8 else "보통"

        self.metrics_label.setText(
            f"<b>종합 모델 성능 요약:</b><br>- 평균 예측 정확도(R2): <b>{r2_avg * 100:.1f}% ({acc_text})</b><br>- 평균 오차(MAE): <b>{mae_avg:.2f}</b>"
        )

        self.canvas.axes.clear()
        target_names = ["Yield Stress", "UTS", "Elongation", "Area Red."]
        r2_scores = metrics["r2"]
        colors = ["#3498db" if score > 0.8 else "#f1c40f" if score > 0.6 else "#e74c3c" for score in r2_scores]
        bars = self.canvas.axes.bar(target_names, r2_scores, color=colors)
        self.canvas.axes.set_ylim(0, 1.1)
        self.canvas.axes.set_ylabel("정확도 (R2 Score)")
        for bar in bars:
            height = bar.get_height()
            self.canvas.axes.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{height:.2f}", ha="center", va="bottom", fontsize=9)
        name_map = {"RF": "Random Forest", "GBM": "Gradient Boosting", "MLP": "Neural Network", "TFP": "TFP"}
        self.canvas.axes.set_title(f"모델별 특성 예측 정확도 ({name_map.get(self.model_type, self.model_type)})")
        self.canvas.draw()
        self.render_performance_results(results)

    def render_performance_results(self, results):
        self.perf_canvas.figure.clear()
        axes = self.perf_canvas.figure.subplots(2, 2)
        y_test = results["y_test"].values
        y_pred = results["y_pred"]
        target_names = ["Yield Stress (MPa)", "UTS (MPa)", "Elongation (%)", "Area Reduction (%)"]
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

        for index, ax in enumerate(axes.flatten()):
            ax.scatter(y_test[:, index], y_pred[:, index], alpha=0.55, color=colors[index], s=18)
            all_data = np.concatenate([y_test[:, index], y_pred[:, index]])
            min_val, max_val = all_data.min(), all_data.max()
            ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7, lw=1)
            ax.set_title(target_names[index], fontsize=10, fontweight="bold")
            ax.set_xlabel("실제값", fontsize=9)
            ax.set_ylabel("예측값", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.6)

        self.perf_canvas.figure.tight_layout()
        self.perf_canvas.draw()

    def on_predict_clicked(self):
        if not self.model_engine:
            self.result_display.setText("Please train or load the model first!")
            return

        try:
            input_dict = {key: widget.text() for key, widget in self.inputs.items()}
            scaled_input = self.data_engine.get_inference_data(input_dict)
            mean_scaled, std_scaled = self.model_engine.predict(scaled_input.astype(np.float32))

            mean = self.data_engine.scaler_y.inverse_transform(mean_scaled)[0]
            std = std_scaled[0] * self.data_engine.scaler_y.scale_

            result_text = (
                f"<b>[강도 예측 결과]</b><br>"
                f"0.2% Yield Stress: <b>{mean[0]:.1f} ± {std[0]:.1f} MPa</b><br>"
                f"UTS: <b>{mean[1]:.1f} ± {std[1]:.1f} MPa</b><br><br>"
                f"<b>[연성 예측 결과]</b><br>"
                f"Elongation: <b>{mean[2]:.1f} ± {std[2]:.1f} %</b><br>"
                f"Area Reduction: <b>{mean[3]:.1f} ± {std[3]:.1f} %</b>"
            )
            self.result_display.setText(result_text)

            self.prediction_canvas.axes.clear()
            labels = ["Yield", "UTS", "Elong.", "Area Red."]
            x = np.arange(len(labels))

            stress_vals = [mean[0], mean[1], 0, 0]
            stress_errs = [1.96 * std[0], 1.96 * std[1], 0, 0]
            self.prediction_canvas.axes.bar(x[:2], stress_vals[:2], yerr=stress_errs[:2], capsize=10, color=["#3498db", "#e74c3c"])
            self.prediction_canvas.axes.set_ylabel("Stress (MPa)", color="#3498db")

            ax2 = self.prediction_canvas.axes.twinx()
            ax2.clear()
            duct_vals = [0, 0, mean[2], mean[3]]
            duct_errs = [0, 0, 1.96 * std[2], 1.96 * std[3]]
            ax2.bar(x[2:], duct_vals[2:], yerr=duct_errs[2:], capsize=10, color=["#2ecc71", "#f39c12"])
            ax2.set_ylabel("Percentage (%)", color="#2ecc71")

            self.prediction_canvas.axes.set_xticks(x)
            self.prediction_canvas.axes.set_xticklabels(labels)
            self.prediction_canvas.axes.set_title("Predicted Properties")
            self.prediction_canvas.draw()
        except Exception as exc:
            self.result_display.setText(f"Error during prediction: {exc}")

    def update_active_model_display(self):
        name_map = {"RF": "Random Forest", "GBM": "Gradient Boosting", "MLP": "Neural Network", "TFP": "TFP"}
        self.active_model_info.setText(f"현재 예측 모델: {name_map.get(self.model_type, self.model_type)}")
        self.active_model_info.setStyleSheet(
            "background-color: #d4efdf; padding: 10px; border: 1px solid #27ae60; border-radius: 5px; font-weight: bold; margin-bottom: 10px;"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)


    def show_quality_help(self):
        help_text = """
        <h2>데이터 전처리 도움말</h2>
        <p>전처리 탭에서는 누락값, 이상치, 형식 오류, 도메인 범위를 학습 전에 먼저 정리합니다.</p>
        <h3>권장 순서</h3>
        <p>파일 선택 → 도메인 기준 설정 → 데이터 품질 처리 설정 → 전처리 실행 → 합금 지표 생성 → 결과 확인</p>
        <h3>추천 시작값</h3>
        <p>누락값 처리: <b>중앙값으로 채우기</b><br>이상치 처리: <b>감지 범위로 보정</b><br>형식 검증: <b>잘못된 값을 NaN으로 변환</b><br>이상치 민감도: <b>1.5</b></p>
        <h3>합금 지표 설명</h3>
        <p><b>Ni 당량 (Ni_eq)</b>: 오스테나이트 안정성을 보는 지표입니다.<br><b>Cr 당량 (Cr_eq)</b>: 페라이트 형성 경향을 보는 지표입니다.<br><b>Cr/Ni 비율</b>: 조직 균형을 보는 지표입니다.<br><b>침입형 원소 합 (C+N)</b>: 고용강화와 고온 강도에 영향을 주는 지표입니다.</p>
        <h3>결과 확인</h3>
        <p>전처리 또는 합금 지표 생성이 끝나면 오른쪽 표에서 전체 데이터를 바로 확인할 수 있습니다.</p>
        """
        self.show_help_dialog("전처리 도움말", help_text)

    def show_model_training_help(self):
        help_text = """
        <h2>모델 학습 도움말</h2>
        <p>모델 학습 탭에서는 전처리 완료 데이터를 이용해 모델을 학습합니다.</p>
        <h3>모델 선택 가이드</h3>
        <p><b>Random Forest</b>: 처음 시작할 때 가장 무난합니다.<br><b>Gradient Boosting</b>: 비교용으로 좋습니다.<br><b>Neural Network</b>: 데이터가 충분할 때 시도해 볼 수 있습니다.<br><b>TFP</b>: 불확실성까지 함께 보고 싶을 때 사용합니다.</p>
        <h3>학습 데이터 선택</h3>
        <p><b>데이터 정제 + 합금 지표</b>: 가장 기본 추천 방식입니다.<br><b>데이터 정제만</b>: 원본 정제 변수만 사용합니다.<br><b>합금 지표만</b>: 생성된 합금 지표만 사용합니다.</p>
        <h3>주의 사항</h3>
        <p>전처리 설정을 바꿨다면 최신 설정을 반영하기 위해 전처리를 다시 실행한 뒤 학습해 주세요.</p>
        """
        self.show_help_dialog("모델 학습 도움말", help_text)

    def show_help_dialog(self, title, html_text):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(820, 640)

        layout = QVBoxLayout(dialog)
        browser = QTextBrowser(dialog)
        browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        browser.setLineWrapMode(QTextBrowser.LineWrapMode.WidgetWidth)
        browser.setStyleSheet("QTextBrowser { font-size: 15px; line-height: 1.7; padding: 16px; }")
        browser.setHtml(html_text)
        layout.addWidget(browser)

        button_row = QHBoxLayout()
        button_row.addStretch()
        close_btn = QPushButton("닫기")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(dialog.accept)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
