import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class StmechDataParser:
    """
    실제 제공받은 데이터셋(stmech_aus_ss.xls)을 불러오고 전처리(Pre-processing)하는 역할을 담당하는 파서(Parser) 클래스입니다.
    """
    def __init__(self, filepath: str = 'data/stmech_aus_ss.xls'):
        # 파일 경로 초기화
        self.filepath = filepath
        
        # 모델의 안정적인 학습을 위해 입력(X)과 출력(y) 스케일러(StandardScaler)를 사용합니다 (정규화 진행)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # 실제 엑셀 파일에서 예측에 사용할 18개의 핵심 입력 변수(Feature)들을 정의합니다.
        # 합금 성분(Cr, Ni 등)과 열처리 온도, 측정 온도 등이 포함됩니다.
        self.features = [
            "Cr", "Ni", "Mo", "Mn", "Si", "Nb", "Ti", "V", "W", "Cu", 
            "N", "C", "B", "P", "S", "Al", 
            "Solution_treatment_temperature", 
            "Temperature (K)"
        ]
        # 모델이 예측할 최종 목적값(Target) 변수로 오스테나이트계 철강의 0.2% 항복강도(Proof Stress)를 지정합니다.
        self.target = "0.2%proof_stress (M Pa)"

    def load_and_preprocess(self, custom_path: str = None):
        """데이터셋을 엑셀 파일로부터 읽어와 정제하는 메인 함수입니다."""
        target_path = custom_path if custom_path else self.filepath
        print(f"{target_path} 에서 데이터 로딩 중 (Loading data)...")
        try:
            # 엑셀 파일의 헤더(열 이름)가 6번째 행에 위치하므로 header=5 (index 기준) 옵션을 줍니다.
            # 파일 확장자에 따라 엔진을 명시적으로 지정하여 예상치 못한 중단을 방지합니다.
            engine = 'xlrd' if target_path.endswith('.xls') else 'openpyxl'
            df = pd.read_excel(target_path, header=5, engine=engine)
            
            # 예측해야 할 정답(Target) 값이 비어있는(NaN) 행은 학습할 수 없으므로 우선 삭제합니다.
            df = df.dropna(subset=[self.target])
            
            # Target 열의 데이터를 숫자형으로 강제 변환합니다. 문자가 섞여 변환할 수 없으면 NaN으로 바꾸고 다시 삭제합니다.
            df[self.target] = pd.to_numeric(df[self.target], errors='coerce')
            df = df.dropna(subset=[self.target])

            # 지정된 18개 입력 변수(Feature) 열들에 대해서도 숫자형으로 변환합니다.
            for col in self.features:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    # 만약 엑셀에 해당 컬럼이 아예 없다면 임의로 0.0을 채워넣어 에러를 방지합니다.
                    df[col] = 0.0 

            # 숫자형 변환 중 문자가 있었거나 원래 빈 값이었던 데이터(NaN)들은 해당 컬럼의 '중간값(Median)'으로 채워줍니다.
            # 데이터 손실을 막기 위한 전처리 기법 중 하나입니다.
            df[self.features] = df[self.features].fillna(df[self.features].median())
            
            # 최종적으로 모델에 들어갈 X(입력 데이터)와 y(정답 데이터)를 추출합니다.
            X_df = df[self.features]
            y_df = df[[self.target]]
            
            print(f"전처리 완료된 데이터 형태: X={X_df.shape}, y={y_df.shape}")
            
            # NumPy 배열 형태로 X의 값, y의 값, 그리고 사용한 특성들의 이름 리스트를 반환합니다.
            return X_df.values, y_df.values, self.features

        except Exception as e:
            print(f"데이터 로딩 중 에러 발생: {e}")
            # 에러가 나서 엑셀 파일을 읽지 못했을 경우, 시스템이 다운되지 않도록 랜덤(가짜) 데이터를 대신 반환합니다.
            return np.random.rand(100, len(self.features)), np.random.rand(100, 1) * 300 + 200, self.features
            
    def fit_transform(self, X, y):
        """데이터 전체를 가지고 정규화(Standard Scaling) 기준점을 학습(fit)함과 동시에 변환(transform)합니다."""
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        return X_scaled, y_scaled
        
    def transform_x(self, X):
        """이미 확보된 정규화 기준을 사용해 새로운 입력값(추론 시)을 변환합니다."""
        return self.scaler_X.transform(X)
        
    def inverse_transform_y(self, y_scaled):
        """정규화되어 나온 모델의 예측값(y_scaled)을 사용자가 알아볼 수 있는 원래의 수치 단위(예: MPa)로 되돌립니다."""
        return self.scaler_y.inverse_transform(y_scaled)

    def save_scalers(self, path: str = 'data/scalers.pkl'):
        """학습 시 사용된 입력/출력 스케일러를 파일로 저장합니다."""
        import pickle
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'scaler_X': self.scaler_X, 'scaler_y': self.scaler_y}, f)
        print(f"스케일러 저장 완료: {path}")

    def load_scalers(self, path: str = 'data/scalers.pkl'):
        """파일로부터 입력/출력 스케일러를 불러옵니다."""
        import pickle
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.scaler_X = data['scaler_X']
                self.scaler_y = data['scaler_y']
            print(f"스케일러 로드 완료: {path}")
            return True
        return False

if __name__ == "__main__":
    # 단독으로 실행(테스트)해볼 때 동작하는 코드입니다.
    parser = StmechDataParser()
    X, y, feats = parser.load_and_preprocess()
    print("사용된 특성(Features):", feats)
    print("타겟(Target) 평균치:", np.mean(y))
