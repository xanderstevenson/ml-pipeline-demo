import pickle
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load model at startup
model = None

def load_model():
    global model
    model_path = 'models/model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    else:
        print("Model not found. Training new model...")
        import sys
        sys.path.append('.')
        from train import train_model
        train_model()
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'features' not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()
        
        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0', port=8080, debug=False)
