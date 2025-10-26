# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback

app = Flask(__name__)

MODEL_PATH = 'linear_regression_model.pkl'

# Load the pre-trained model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Fallback if model fails to load


@app.route('/')
def home():
    """Check if the API is running."""
    return "ML Model Prediction API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint - expects JSON with 'features' key."""
    if model is None:
        return jsonify({
            "status": "error",
            "message": f"Model '{MODEL_PATH}' not loaded. Cannot make predictions."
        }), 500

    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Request must be JSON"
        }), 400

    data = request.get_json()
    print(f"Received prediction request with data: {data}")

    if 'features' not in data:
        return jsonify({
            "status": "error",
            "message": "Missing 'features' key in JSON payload."
        }), 400

    features = data['features']

    if not isinstance(features, list) or len(features) != 5:
        return jsonify({
            "status": "error",
            "message": "Features must be a list of 5 numeric values."
        }), 400

    try:
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        return jsonify({
            "status": "success",
            "prediction": float(prediction)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": "An error occurred during prediction.",
            "details": str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
