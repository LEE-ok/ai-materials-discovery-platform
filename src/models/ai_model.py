import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pickle

# 확률 분포를 쉽게 다루기 위해 tensorflow_probability 내부의 distributions 모듈을 가져옵니다.
tfd = tfp.distributions

class MaterialPredictionModel:
    """
    TensorFlow(TFP) 및 Scikit-learn 모델을 통합 관리하며, 
    예측 및 다차원 최적화를 수행하는 통합 AI 클래스입니다.
    """
    def __init__(self, input_shape: int, model_type: str = 'TFP'):
        self.input_shape = input_shape
        self.model_type = model_type
        self.output_shape = 1
        
        # 모델 타입에 따라 내부 모델 초기화
        self.model = None
        if model_type == 'TFP':
            self.model = self._build_tfp_model()
        elif model_type == 'MLP':
            self.model = self._build_mlp_model()
        # RF, GBM 등은 Scikit-learn 객체이므로 학습 시점에 초기화하거나 필요시 미리 생성
        
    def _build_tfp_model(self) -> tf.keras.Model:
        """TFP 모델: 불확실성 측정이 가능한 확률적 신경망"""
        inputs = tf.keras.layers.Input(shape=(self.input_shape,))
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dense(2)(x) 
        outputs = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., 0:1], scale=1e-3 + tf.math.softplus(t[..., 1:2]))
        )(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        def nll(y_true, y_pred_dist):
            return -y_pred_dist.log_prob(y_true)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss=nll)
        return model

    def _build_mlp_model(self) -> tf.keras.Model:
        """일반 MLP 모델: 결정론적 고성능 신경망"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.input_shape,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, validation_data: tuple = None, epochs: int = 50, batch_size: int = 32, callbacks: list = None):
        """선택된 모델 타입에 맞춰 학습을 진행합니다."""
        if self.model_type in ['TFP', 'MLP']:
            return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                validation_data=validation_data, verbose=1, callbacks=callbacks)
        
        elif self.model_type == 'RF':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train.flatten())
            return None # SKLearn은 history 객체 없음
            
        elif self.model_type == 'GBM':
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train.flatten())
            return None

    def predict_with_confidence(self, X_test: np.ndarray):
        """모델별 예측 수행 (TFP인 경우에만 신뢰구간 포함)"""
        if self.model_type == 'TFP':
            prediction_dist = self.model(X_test)
            mean_pred = prediction_dist.mean().numpy()
            stddev_pred = prediction_dist.stddev().numpy()
            lower_bound = mean_pred - 1.96 * stddev_pred
            upper_bound = mean_pred + 1.96 * stddev_pred
            return {
                "mean": mean_pred.flatten().tolist(),
                "uncertainty": stddev_pred.flatten().tolist(),
                "lower_95_ci": lower_bound.flatten().tolist(),
                "upper_95_ci": upper_bound.flatten().tolist()
            }
        else:
            # MLP, RF, GBM 등은 일반 예측만 수행
            if self.model_type in ['RF', 'GBM']:
                mean_pred = self.model.predict(X_test).reshape(-1, 1)
            else:
                mean_pred = self.model.predict(X_test)
            return {
                "mean": mean_pred.flatten().tolist(),
                "uncertainty": [0.0] * len(mean_pred),
                "lower_95_ci": mean_pred.flatten().tolist(),
                "upper_95_ci": mean_pred.flatten().tolist()
            }

    def save_model(self, filepath: str):
        """가중치 또는 객체 상태 저장"""
        if self.model_type in ['TFP', 'MLP']:
            self.model.save_weights(filepath)
        else:
            with open(filepath + ".pkl", "wb") as f:
                pickle.dump(self.model, f)

    def load_model(self, filepath: str, model_type: str = 'TFP'):
        """저장된 모델 불러오기"""
        self.model_type = model_type
        if model_type in ['TFP', 'MLP']:
            # TFP/MLP 뼈대 먼저 빌드
            self.model = self._build_tfp_model() if model_type == 'TFP' else self._build_mlp_model()
            if os.path.exists(filepath + ".index") or os.path.exists(filepath):
                self.model.load_weights(filepath)
                return True
        else:
            if os.path.exists(filepath + ".pkl"):
                with open(filepath + ".pkl", "rb") as f:
                    self.model = pickle.load(f)
                return True
        return False

    def optimize_composition(self, bounds: list, scaler_X=None, scaler_y=None, n_restarts: int = 10, iteration_callback=None):
        """시뮬레이션 진행 상황을 iteration_callback으로 알리며 최적화 수행"""
        from scipy.optimize import minimize
        
        def objective(x):
            x_input = np.array([x], dtype=np.float32)
            if scaler_X: x_input = scaler_X.transform(x_input)
            
            if self.model_type == 'TFP':
                mean_val = self.model(x_input).mean().numpy().flatten()[0]
            elif self.model_type in ['RF', 'GBM']:
                mean_val = self.model.predict(x_input)[0]
            else: # MLP
                mean_val = self.model.predict(x_input, verbose=0).flatten()[0]
                
            if scaler_y:
                mean_val = scaler_y.inverse_transform(np.array([[mean_val]]))[0, 0]
            return -float(mean_val)

        best_res_x, best_res_y = None, float('inf')

        for i in range(n_restarts):
            if iteration_callback:
                iteration_callback(i + 1, n_restarts) # 현재 단계 알림
            
            x0 = [ np.random.uniform(b[0], b[1]) for b in bounds ]
            res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            
            if res.success and res.fun < best_res_y:
                best_res_y = res.fun
                best_res_x = res.x
        
        return (best_res_x, -best_res_y) if best_res_x is not None else (None, None)

if __name__ == "__main__":
    # 이 파일만 단독 실행 시, 18개의 입력 특성을 가진 모델의 뼈대(Summary)를 출력합니다.
    ai_system = MaterialPredictionModel(input_shape=18)
    ai_system.model.summary()
