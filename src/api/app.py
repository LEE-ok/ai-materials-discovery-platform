from flask import Flask, request, jsonify
import numpy as np
import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.ai_model import MaterialPredictionModel
from src.data_pipeline.parser import StmechDataParser

app = Flask(__name__)

# System Globals
data_parser = StmechDataParser()
# We must fit the scaler on the data first so we can inverse_transform predictions properly
temp_X, temp_y, features_list = data_parser.load_and_preprocess()
data_parser.fit_transform(temp_X, temp_y)

INPUT_SHAPE = len(features_list)
model = MaterialPredictionModel(input_shape=INPUT_SHAPE)
# Try loading weights if they exist
model_path = os.path.join(os.path.dirname(__file__), "../../data/model_weights.h5")
if model.load_model(model_path):
    print("Loaded trained model weights.")
else:
    print("No trained weights found. Will return mock/randomized predictions until trained.")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running"}), 200

# 전역 모델 상태 관리용
MODEL_TYPE = "TFP" # 기본값
MODEL_BASE_PATH = os.path.join(os.path.dirname(__file__), "../../data/model_weights")

@app.route('/reload', methods=['POST'])
def reload_model():
    """Reload the model weights and scalers from disk."""
    global MODEL_TYPE
    data = request.json or {}
    new_model_type = data.get('model_type', 'TFP')
    
    scalers_path = os.path.join(os.path.dirname(__file__), "../../data/scalers.pkl")
    # 모델 타입 변경 및 로드
    success = model.load_model(MODEL_BASE_PATH, model_type=new_model_type)
    scaler_success = data_parser.load_scalers(scalers_path)
    
    if success:
        MODEL_TYPE = new_model_type
    
    if success and scaler_success:
        return jsonify({"status": "success", "message": f"Model ({new_model_type}) reloaded."}), 200
    else:
        return jsonify({"status": "warning", "message": "Reloaded partially (weights: {}, scalers: {})".format(success, scaler_success)}), 200

@app.route('/features', methods=['GET'])
def get_features():
    """Return the list of features the model expects."""
    return jsonify({"features": features_list})

@app.route('/predict', methods=['POST'])
def predict_properties():
    try:
        data = request.json
        input_features = data.get('features')
        
        if not input_features or len(input_features) != INPUT_SHAPE:
            return jsonify({"error": f"Features must be a list of length {INPUT_SHAPE}"}), 400

        # Create numpy array and scale it
        X_test = np.array([input_features])
        X_test_scaled = data_parser.transform_x(X_test)
        
        # Predict
        res_scaled = model.predict_with_confidence(X_test_scaled)
        
        # Inverse transform the mean
        mean_scaled = np.array([[res_scaled['mean'][0]]])
        mean_unscaled = data_parser.inverse_transform_y(mean_scaled)[0][0]
        
        # For uncertainty, we scale approximately using the scaler variance
        scale_factor = data_parser.scaler_y.scale_[0]
        uncert_unscaled = res_scaled['uncertainty'][0] * scale_factor
        lower_unscaled = mean_unscaled - 1.96 * uncert_unscaled
        upper_unscaled = mean_unscaled + 1.96 * uncert_unscaled

        res = {
            "mean": [float(mean_unscaled)],
            "uncertainty": [float(uncert_unscaled)],
            "lower_95_ci": [float(lower_unscaled)],
            "upper_95_ci": [float(upper_unscaled)],
            "unit": "MPa"
        }
        
        return jsonify({
            "status": "success",
            "predictions": res
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
