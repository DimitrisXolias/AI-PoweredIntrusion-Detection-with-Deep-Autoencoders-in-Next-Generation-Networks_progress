
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from joblib import load

# Load model and scaler
model = tf.keras.models.load_model('models/1dcnn_model.h5')
scaler = load('models/scaler_raw83.pkl')

print("Model input shape:", model.input_shape)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'IDS API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("🔹 Received raw input:", data)

        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features).reshape(1, 84, 1)

        print("🔹 Scaled shape:", features_scaled.shape)

        prediction = model.predict(features_scaled)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        print("✅ Predicted class:", predicted_class)
        print("✅ Confidence:", confidence)

        return jsonify({'predicted_class': predicted_class, 'confidence': confidence})
    except Exception as e:
        print("❌ Error occurred:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
