import os

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataEngine:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.df = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        self.raw_feature_cols = [
            "Cr",
            "Ni",
            "Mo",
            "Mn",
            "Si",
            "Nb",
            "Ti",
            "Zr",
            "Ta",
            "V",
            "W",
            "Cu",
            "N",
            "C",
            "B",
            "P",
            "S",
            "Co",
            "Al",
            "Sn",
            "Pb",
            "Solution_treatment_temperature",
            "Solution_treatment_time(s)",
            "Water_Quenched_after_s.t.",
            "Air_Quenched_after_s.t.",
            "Grains mm-2",
            "Type of melting",
            "Size of ingot",
            "Product form",
            "Temperature (K)",
        ]
        self.engineered_feature_cols = [
            "Cr_Ni_ratio",
            "C_plus_N",
            "Ni_eq",
            "Cr_eq",
        ]
        self.feature_cols = self.raw_feature_cols + self.engineered_feature_cols

        self.target_cols = [
            "0.2%proof_stress (M Pa)",
            "UTS (M Pa)",
            "Elongation (%)",
            "Area_reduction (%)",
        ]
        self.binary_cols = [
            "Water_Quenched_after_s.t.",
            "Air_Quenched_after_s.t.",
        ]
        self.nonnegative_cols = [
            "Cr",
            "Ni",
            "Mo",
            "Mn",
            "Si",
            "Nb",
            "Ti",
            "Zr",
            "Ta",
            "V",
            "W",
            "Cu",
            "N",
            "C",
            "B",
            "P",
            "S",
            "Co",
            "Al",
            "Sn",
            "Pb",
            "Solution_treatment_temperature",
            "Solution_treatment_time(s)",
            "Grains mm-2",
            "Type of melting",
            "Size of ingot",
            "Product form",
            "Temperature (K)",
            "0.2%proof_stress (M Pa)",
            "UTS (M Pa)",
        ]
        self.quality_options = {
            "missing_strategy": "mean",
            "outlier_strategy": "clip",
            "invalid_type_strategy": "coerce",
            "iqr_factor": 1.5,
            "feature_engineering": True,
            "input_feature_mode": "combined",
        }
        self.custom_ranges = {}
        self.domain_range_groups = {
            "Cr": "오스테나이트 조성 기준",
            "Ni": "오스테나이트 조성 기준",
            "Mo": "오스테나이트 조성 기준",
            "Mn": "오스테나이트 조성 기준",
            "Cu": "오스테나이트 조성 기준",
            "N": "오스테나이트 조성 기준",
            "C": "오스테나이트 조성 기준",
            "Si": "오스테나이트 조성 기준",
            "P": "오스테나이트 조성 기준",
            "S": "오스테나이트 조성 기준",
            "Nb": "오스테나이트 조성 기준",
            "Ti": "오스테나이트 조성 기준",
            "Zr": "오스테나이트 조성 기준",
            "Ta": "오스테나이트 조성 기준",
            "V": "오스테나이트 조성 기준",
            "W": "오스테나이트 조성 기준",
            "B": "오스테나이트 조성 기준",
            "Co": "오스테나이트 조성 기준",
            "Al": "오스테나이트 조성 기준",
            "Sn": "오스테나이트 조성 기준",
            "Pb": "오스테나이트 조성 기준",
            "Solution_treatment_temperature": "고온 특성 기준",
            "Solution_treatment_time(s)": "고온 특성 기준",
            "Water_Quenched_after_s.t.": "고온 특성 기준",
            "Air_Quenched_after_s.t.": "고온 특성 기준",
            "Grains mm-2": "고온 특성 기준",
            "Type of melting": "고온 특성 기준",
            "Size of ingot": "고온 특성 기준",
            "Product form": "고온 특성 기준",
            "Temperature (K)": "고온 특성 기준",
            "0.2%proof_stress (M Pa)": "고온 특성 기준",
            "UTS (M Pa)": "고온 특성 기준",
            "Elongation (%)": "고온 특성 기준",
            "Area_reduction (%)": "고온 특성 기준",
        }
        self.domain_range_basis = {
            "Cr": "SSINA 오스테나이트 표",
            "Ni": "SSINA 오스테나이트 표",
            "Mo": "SSINA 오스테나이트 표",
            "Mn": "SSINA 오스테나이트 표",
            "Cu": "SSINA 오스테나이트 표",
            "N": "SSINA 오스테나이트 표",
            "C": "SSINA 오스테나이트 표",
            "Temperature (K)": "SSINA 고온 특성",
        }
        self.default_domain_ranges = {
            "Cr": (16, 26),
            "Ni": (3.5, 37),
            "Mo": (0, 5),
            "Mn": (0, 9),
            "Si": (0, 2.5),
            "Nb": (0, 1),
            "Ti": (0, 1),
            "Zr": (0, 2),
            "Ta": (0, 5),
            "V": (0, 5),
            "W": (0, 20),
            "Cu": (0, 4),
            "N": (0, 0.4),
            "C": (0, 0.15),
            "B": (0, 0.2),
            "P": (0, 0.2),
            "S": (0, 0.2),
            "Co": (0, 30),
            "Al": (0, 10),
            "Sn": (0, 2),
            "Pb": (0, 2),
            "Solution_treatment_temperature": (900, 1500),
            "Solution_treatment_time(s)": (0, 172800),
            "Water_Quenched_after_s.t.": (0, 1),
            "Air_Quenched_after_s.t.": (0, 1),
            "Grains mm-2": (0, 1000000),
            "Type of melting": (0, 10),
            "Size of ingot": (0, 10000),
            "Product form": (0, 20),
            "Temperature (K)": (273, 1422),
            "0.2%proof_stress (M Pa)": (0, 3000),
            "UTS (M Pa)": (0, 4000),
            "Elongation (%)": (0, 100),
            "Area_reduction (%)": (0, 100),
        }
        self.last_quality_report = {}

    def set_file_path(self, path):
        self.file_path = path

    def configure_quality_rules(
        self,
        missing_strategy=None,
        outlier_strategy=None,
        invalid_type_strategy=None,
        iqr_factor=None,
        custom_ranges=None,
        feature_engineering=None,
        input_feature_mode=None,
    ):
        if missing_strategy:
            self.quality_options["missing_strategy"] = missing_strategy
        if outlier_strategy:
            self.quality_options["outlier_strategy"] = outlier_strategy
        if invalid_type_strategy:
            self.quality_options["invalid_type_strategy"] = invalid_type_strategy
        if iqr_factor is not None:
            self.quality_options["iqr_factor"] = float(iqr_factor)
        if feature_engineering is not None:
            self.quality_options["feature_engineering"] = bool(feature_engineering)
        if input_feature_mode:
            self.quality_options["input_feature_mode"] = input_feature_mode
        if custom_ranges:
            self.custom_ranges = dict(custom_ranges)

    def get_domain_ranges(self):
        merged = dict(self.default_domain_ranges)
        merged.update(self.custom_ranges)
        return merged

    def get_domain_group(self, column):
        return self.domain_range_groups.get(column, "기타 기준")

    def get_domain_basis(self, column):
        return self.domain_range_basis.get(column, "현재 데이터셋 기준 추론")

    def set_custom_domain_ranges(self, custom_ranges):
        normalized = {}
        for column, bounds in custom_ranges.items():
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                continue
            lower_bound, upper_bound = bounds
            normalized[column] = (
                None if lower_bound is None else float(lower_bound),
                None if upper_bound is None else float(upper_bound),
            )
        self.custom_ranges = normalized

    def reset_custom_domain_ranges(self):
        self.custom_ranges = {}

    def get_preprocessed_display_df(self):
        if self.df is None:
            return pd.DataFrame()
        display_cols = [col for col in self.raw_feature_cols + self.target_cols if col in self.df.columns]
        return self.df[display_cols].copy()

    def get_engineered_display_df(self):
        if self.df is None:
            return pd.DataFrame()
        display_cols = [col for col in self.engineered_feature_cols if col in self.df.columns]
        if not display_cols:
            return pd.DataFrame(index=self.df.index)
        return self.df[display_cols].copy()

    def load_data(self, include_engineered=None):
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found at {self.file_path}")

        df = pd.read_excel(self.file_path, header=5)
        self.df = self.apply_quality_routine(df, include_engineered=include_engineered)
        return self.df

    def apply_quality_routine(self, df, include_engineered=None):
        numeric_cols = self._existing_columns(self.raw_feature_cols + self.target_cols, df)
        if not numeric_cols:
            raise ValueError("No expected feature/target columns were found in the dataset.")

        report = {
            "rows_before": int(len(df)),
            "rows_after": 0,
            "invalid_type_cells": 0,
            "missing_cells_before": 0,
            "missing_cells_after": 0,
            "outlier_cells": 0,
            "rows_removed_by_missing": 0,
            "rows_removed_by_outlier": 0,
            "domain_range_cells": 0,
            "missing_strategy": self.quality_options["missing_strategy"],
            "outlier_strategy": self.quality_options["outlier_strategy"],
            "invalid_type_strategy": self.quality_options["invalid_type_strategy"],
            "feature_engineering_enabled": bool(self.quality_options["feature_engineering"]),
            "engineered_features_added": [],
            "columns_checked": len(numeric_cols),
        }

        cleaned_df = df.copy()
        cleaned_df, invalid_type_cells = self._coerce_numeric_columns(cleaned_df, numeric_cols)
        report["invalid_type_cells"] = invalid_type_cells
        report["missing_cells_before"] = int(cleaned_df[numeric_cols].isna().sum().sum())

        if self.quality_options["invalid_type_strategy"] == "drop":
            before_rows = len(cleaned_df)
            cleaned_df = cleaned_df.dropna(subset=numeric_cols)
            report["rows_removed_by_missing"] += before_rows - len(cleaned_df)

        cleaned_df, missing_removed = self._handle_missing_values(cleaned_df, numeric_cols)
        report["rows_removed_by_missing"] += missing_removed

        outlier_masks, clip_bounds = self._detect_outliers(cleaned_df, numeric_cols)
        report["outlier_cells"] = int(sum(int(mask.sum()) for mask in outlier_masks.values()))
        report["domain_range_cells"] = int(
            sum(int(mask.sum()) for col, mask in outlier_masks.items() if col in self._active_domain_range_columns())
        )

        cleaned_df, outlier_removed = self._handle_outliers(cleaned_df, outlier_masks, clip_bounds)
        report["rows_removed_by_outlier"] = outlier_removed
        report["missing_cells_after"] = int(cleaned_df[numeric_cols].isna().sum().sum())

        cleaned_df = cleaned_df.reset_index(drop=True)

        use_engineered = self.quality_options["feature_engineering"] if include_engineered is None else bool(include_engineered)
        if use_engineered:
            cleaned_df = self._add_engineered_features(cleaned_df)
            report["engineered_features_added"] = [
                col for col in self.engineered_feature_cols if col in cleaned_df.columns
            ]

        report["rows_after"] = int(len(cleaned_df))
        self.last_quality_report = report
        return cleaned_df

    def generate_engineered_features_on_current_df(self):
        if self.df is None or self.df.empty:
            raise ValueError("정제된 데이터가 없습니다. 먼저 데이터 전처리를 실행해 주세요.")

        base_df = self.df.copy()
        for col in self.engineered_feature_cols:
            if col in base_df.columns:
                base_df = base_df.drop(columns=[col])

        self.df = self._add_engineered_features(base_df).reset_index(drop=True)
        self.quality_options["feature_engineering"] = True
        self.last_quality_report["feature_engineering_enabled"] = True
        self.last_quality_report["engineered_features_added"] = list(self.engineered_feature_cols)
        self.last_quality_report["rows_after"] = int(len(self.df))
        return self.df

    def preprocess_data(self, test_size=0.2):
        if self.df is None or self.df.empty:
            raise ValueError("Dataset is empty after quality processing.")

        available_features = self._get_selected_feature_columns(self.df)
        available_targets = self._existing_columns(self.target_cols, self.df)

        X = self.df[available_features].copy()
        y = self.df[available_targets].copy()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        X_train_scaled = self.scaler_x.fit_transform(X_train)
        X_test_scaled = self.scaler_x.transform(X_test)

        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_test, y_test

    def get_inference_data(self, input_dict):
        input_row = {}
        for col in self.raw_feature_cols:
            raw_value = input_dict.get(col, 0)
            input_row[col] = float(raw_value or 0)

        input_df = pd.DataFrame([input_row], columns=self.raw_feature_cols)
        if self.quality_options.get("feature_engineering", True):
            input_df = self._add_engineered_features(input_df)

        available_features = self._get_selected_feature_columns(input_df)
        return self.scaler_x.transform(input_df[available_features])

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled)

    def format_quality_report(self):
        if not self.last_quality_report:
            return "No data quality report available."

        report = self.last_quality_report
        engineered_count = len(report.get("engineered_features_added", []))
        return (
            f"Rows {report['rows_before']} -> {report['rows_after']} | "
            f"invalid types: {report['invalid_type_cells']} | "
            f"missing: {report['missing_cells_before']} -> {report['missing_cells_after']} | "
            f"outliers: {report['outlier_cells']} | "
            f"domain range flags: {report.get('domain_range_cells', 0)} | "
            f"engineered features: {engineered_count}"
        )

    def _existing_columns(self, columns, df):
        return [col for col in columns if col in df.columns]

    def _get_selected_feature_columns(self, df):
        mode = self.quality_options.get("input_feature_mode", "combined")
        if mode == "clean_only":
            candidate_cols = self.raw_feature_cols
        elif mode == "engineered_only":
            candidate_cols = self.engineered_feature_cols
        else:
            candidate_cols = self.feature_cols
        return self._existing_columns(candidate_cols, df)

    def _coerce_numeric_columns(self, df, numeric_cols):
        invalid_type_cells = 0
        for col in numeric_cols:
            original = df[col]
            converted = pd.to_numeric(original, errors="coerce")
            invalid_mask = original.notna() & converted.isna()
            invalid_type_cells += int(invalid_mask.sum())
            df[col] = converted
        return df, invalid_type_cells

    def _handle_missing_values(self, df, numeric_cols):
        strategy = self.quality_options["missing_strategy"]
        before_rows = len(df)

        if strategy == "drop":
            dropped_df = df.dropna(subset=numeric_cols)
            return dropped_df, before_rows - len(dropped_df)

        if df.empty:
            return df, 0

        if strategy == "mean":
            fill_values = df[numeric_cols].mean()
            df[numeric_cols] = df[numeric_cols].fillna(fill_values)
        elif strategy == "median":
            fill_values = df[numeric_cols].median()
            df[numeric_cols] = df[numeric_cols].fillna(fill_values)
        elif strategy == "knn":
            imputer = KNNImputer(n_neighbors=min(5, max(1, len(df) - 1)))
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = df[numeric_cols].fillna(0)

        return df, 0

    def _detect_outliers(self, df, numeric_cols):
        outlier_masks = {}
        clip_bounds = {}

        for col in numeric_cols:
            series = df[col]
            if series.dropna().empty:
                outlier_masks[col] = pd.Series(False, index=df.index)
                continue

            lower_bound, upper_bound = self._expected_range_for_column(col)
            if lower_bound is None or upper_bound is None:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if pd.isna(iqr) or iqr == 0:
                    stat_lower = series.min()
                    stat_upper = series.max()
                else:
                    factor = self.quality_options["iqr_factor"]
                    stat_lower = q1 - (iqr * factor)
                    stat_upper = q3 + (iqr * factor)

                if lower_bound is None:
                    lower_bound = stat_lower
                if upper_bound is None:
                    upper_bound = stat_upper

            outlier_masks[col] = (series < lower_bound) | (series > upper_bound)
            clip_bounds[col] = (lower_bound, upper_bound)

        return outlier_masks, clip_bounds

    def _handle_outliers(self, df, outlier_masks, clip_bounds):
        strategy = self.quality_options["outlier_strategy"]
        if strategy == "flag":
            return df, 0

        if strategy == "remove":
            combined_mask = pd.Series(False, index=df.index)
            for mask in outlier_masks.values():
                combined_mask = combined_mask | mask
            removed_rows = int(combined_mask.sum())
            return df.loc[~combined_mask].copy(), removed_rows

        for col, (lower_bound, upper_bound) in clip_bounds.items():
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return df, 0

    def _add_engineered_features(self, df):
        df = df.copy()
        eps = 1e-8

        cr = df.get("Cr", pd.Series(0.0, index=df.index)).fillna(0.0)
        ni = df.get("Ni", pd.Series(0.0, index=df.index)).fillna(0.0)
        c = df.get("C", pd.Series(0.0, index=df.index)).fillna(0.0)
        n = df.get("N", pd.Series(0.0, index=df.index)).fillna(0.0)
        mn = df.get("Mn", pd.Series(0.0, index=df.index)).fillna(0.0)
        mo = df.get("Mo", pd.Series(0.0, index=df.index)).fillna(0.0)
        si = df.get("Si", pd.Series(0.0, index=df.index)).fillna(0.0)
        nb = df.get("Nb", pd.Series(0.0, index=df.index)).fillna(0.0)
        cu = df.get("Cu", pd.Series(0.0, index=df.index)).fillna(0.0)

        df["Cr_Ni_ratio"] = np.where(np.abs(ni) > eps, cr / ni, 0.0)
        df["C_plus_N"] = c + n
        df["Ni_eq"] = ni + (30.0 * c) + (0.5 * mn) + (30.0 * n) + (0.3 * cu)
        df["Cr_eq"] = cr + mo + (1.5 * si) + (0.5 * nb)
        return df

    def _expected_range_for_column(self, column):
        if column in self.custom_ranges:
            return self.custom_ranges[column]
        if column in self.default_domain_ranges:
            return self.default_domain_ranges[column]
        if column in self.binary_cols:
            return 0, 1
        if column in {"Elongation (%)", "Area_reduction (%)"}:
            return 0, 100
        if column in self.nonnegative_cols:
            return 0, None
        return None, None

    def _active_domain_range_columns(self):
        return set(self.default_domain_ranges.keys()) | set(self.custom_ranges.keys())
