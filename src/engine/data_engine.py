import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

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
        
    def set_file_path(self, path):
        self.file_path = path

    def load_data(self):
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found at {self.file_path}")
        
        # Load XLS file with correct header row
        self.df = pd.read_excel(self.file_path, header=5)
        
        # Robust cleaning: Convert columns to numeric, non-convertible strings become NaN
        for col in self.feature_cols + self.target_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Drop rows where targets are missing
        self.df = self.df.dropna(subset=self.target_cols)
        
        # Fill NaN in features with 0
        self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0)
        
        return self.df

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
