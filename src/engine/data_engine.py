import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json

class DataEngine:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.df = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_cols = [
            'Cr', 'Ni', 'Mo', 'Mn', 'Si', 'Nb', 'Ti', 'Zr', 'Ta', 'V', 'W', 'Cu', 'N', 'C', 'B', 'P', 'S', 'Co', 'Al', 'Sn', 'Pb',
            'Solution_treatment_temperature', 
            'Solution_treatment_time(s)', 
            'Water_Quenched_after_s.t.', 
            'Air_Quenched_after_s.t.', 
            'Grains mm-2', 
            'Type of melting', 
            'Size of ingot', 
            'Product form', 
            'Temperature (K)'
        ]
        self.target_cols = [
            '0.2%proof_stress (M Pa)', 
            'UTS (M Pa)', 
            'Elongation (%)', 
            'Area_reduction (%)'
        ]
        self.validation_report_df = None
        self.validation_summary = None
        self.validation_out_dir = "outputs/validation"
        
    def set_file_path(self, path):
        self.file_path = path

    def set_validation_output_dir(self, out_dir):
        self.validation_out_dir = out_dir

    def load_data(self):
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found at {self.file_path}")
        
        # Load XLS file with correct header row
        self.df = pd.read_excel(self.file_path, header=5)
        
        # Robust cleaning: Convert columns to numeric, non-convertible strings become NaN
        for col in self.feature_cols + self.target_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Run integrity validation before dropping/filling rows.
        self.validation_report_df, self.validation_summary = self.validate_data_integrity(self.df.copy())
        self._persist_validation_outputs()
        if self.validation_summary["rows_reject"] > 0:
            raise ValueError(
                f"데이터 정합성 검증 실패: reject={self.validation_summary['rows_reject']}건, "
                f"hold={self.validation_summary['rows_hold']}건. 학습을 중단합니다."
            )
        
        # Drop rows where targets are missing
        self.df = self.df.dropna(subset=self.target_cols)
        
        # Fill NaN in features with 0
        self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0)
        
        return self.df

    def validate_data_integrity(self, df):
        """
        Validate rows and classify into pass/hold/reject.
        - reject: physically impossible values or invalid binary columns.
        - hold: suspicious but potentially usable values (e.g., Cr/Ni family-range outliers).
        """
        wt_cols = [c for c in ['Cr', 'Ni', 'Mo', 'Mn', 'Si', 'Nb', 'Ti', 'Zr', 'Ta', 'V', 'W', 'Cu', 'N', 'C', 'B', 'P', 'S', 'Co', 'Al', 'Sn', 'Pb'] if c in df.columns]
        binary_cols = [c for c in ['Water_Quenched_after_s.t.', 'Air_Quenched_after_s.t.'] if c in df.columns]
        min_zero_cols = [c for c in self.feature_cols + self.target_cols if c in df.columns]

        statuses = []
        reason_codes = []
        reason_details = []

        for _, row in df.iterrows():
            hard = []
            soft = []
            details = []

            # Hard fail: any negative numeric value in core columns.
            for c in min_zero_cols:
                v = row.get(c)
                if pd.notna(v) and v < 0:
                    hard.append("NEGATIVE_VALUE")
                    details.append(f"{c}={v} < 0")
                    break

            # Hard fail: invalid binary values.
            for c in binary_cols:
                v = row.get(c)
                if pd.notna(v) and v not in (0, 1):
                    hard.append("BINARY_INVALID")
                    details.append(f"{c}={v} not in {{0,1}}")
                    break

            # Hard fail: listed element wt% sum unrealistically above 100.
            if wt_cols:
                wt_sum = row[wt_cols].sum(skipna=True)
                if pd.notna(wt_sum) and wt_sum > 100.5:
                    hard.append("WT_SUM_GT_100")
                    details.append(f"sum_wt={wt_sum:.3f} > 100.5")

            # Soft fail: family range outliers for austenitic stainless.
            cr = row.get('Cr')
            ni = row.get('Ni')
            if pd.notna(cr) and not (16.0 <= cr <= 28.0):
                soft.append("CR_OUT_OF_RANGE")
                details.append(f"Cr={cr:.3f} outside [16,28]")
            if pd.notna(ni) and not (3.5 <= ni <= 32.0):
                soft.append("NI_OUT_OF_RANGE")
                details.append(f"Ni={ni:.3f} outside [3.5,32]")

            if hard:
                statuses.append("reject")
                reason_codes.append(";".join(hard + soft))
            elif soft:
                statuses.append("hold")
                reason_codes.append(";".join(soft))
            else:
                statuses.append("pass")
                reason_codes.append("")
            reason_details.append(";".join(details))

        report = df.copy()
        report["quality_status"] = statuses
        report["quality_reason_codes"] = reason_codes
        report["quality_reason_detail"] = reason_details

        summary = {
            "rows_total": int(len(report)),
            "rows_pass": int((report["quality_status"] == "pass").sum()),
            "rows_hold": int((report["quality_status"] == "hold").sum()),
            "rows_reject": int((report["quality_status"] == "reject").sum()),
        }
        return report, summary

    def _persist_validation_outputs(self):
        if self.validation_report_df is None or self.validation_summary is None:
            return
        out_dir = self.validation_out_dir
        if not out_dir:
            return
        os.makedirs(out_dir, exist_ok=True)
        self.validation_report_df.to_csv(os.path.join(out_dir, "gui_validation_report.csv"), index=False)
        with open(os.path.join(out_dir, "gui_validation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(self.validation_summary, f, ensure_ascii=False, indent=2)

    def preprocess_data(self, test_size=0.2):
        X = self.df[self.feature_cols].copy()
        y = self.df[self.target_cols].copy()
        
        # Split data without fixed seed to see variation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        # Scale data
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        X_test_scaled = self.scaler_x.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_test, y_test

    def get_inference_data(self, input_dict):
        """Convert GUI input dict to scaled numpy array for prediction"""
        # Ensure all features are present
        input_data = []
        for col in self.feature_cols:
            input_data.append(float(input_dict.get(col, 0)))
            
        input_arr = np.array([input_data])
        # Convert to DataFrame to provide feature names and avoid scikit-learn warnings
        input_df = pd.DataFrame(input_arr, columns=self.feature_cols)
        return self.scaler_x.transform(input_df)

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled)
