from flask import Flask, request, jsonify
from src.engine.data_engine import DataEngine
from src.engine.model_engine import ModelEngine
import os
import joblib
import numpy as np

app = Flask(__name__)

# Global instances (will be loaded on start if models exist)
data_engine = None
model_engine = None

def get_model_paths():
    model_dir = os.getenv("AIMD_MODEL_DIR", "models")
    model_path = os.path.join(model_dir, "material_model.pkl")
    engine_path = os.path.join(model_dir, "data_engine.pkl")
    return model_path, engine_path

def load_resources():
    global data_engine, model_engine
    model_path, engine_path = get_model_paths()
    
    if os.path.exists(engine_path):
        data_engine = joblib.load(engine_path)
    
    if os.path.exists(model_path):
        # Initialize with dummy values, load() will overwrite them
        model_engine = ModelEngine(model_type='RF', output_dim=4)
        model_engine.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_engine or not data_engine:
        return jsonify({"error": "Model not trained or loaded. Please train the model via GUI first."}), 400
    
    data = request.json
    try:
        # Scale input
        scaled_input = data_engine.get_inference_data(data)
        
        # Predict (returns mean and std/uncertainty)
        mean_scaled, std_scaled = model_engine.predict(scaled_input)
        
        # Inverse transform to get original units (MPa, %)
        mean = data_engine.scaler_y.inverse_transform(mean_scaled)
        
        # Scale std to original units
        std = std_scaled * data_engine.scaler_y.scale_
        
        target_names = ["yield_stress_mpa", "uts_mpa", "elongation_pct", "area_reduction_pct"]
        results = {}
        
        for i, name in enumerate(target_names):
            results[name] = {
                "value": round(float(mean[0, i]), 2),
                "uncertainty": round(float(std[0, i]), 2)
            }
            
        return jsonify({
            "status": "success",
            "model_type": model_engine.model_type,
            "predictions": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_resources()
    print("AI 소재 발굴 API 서버가 5000번 포트에서 시작되었습니다.")
    app.run(host='0.0.0.0', port=5000)
