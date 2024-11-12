from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Load the saved model
model_path = r"C:\Users\abhin\OneDrive\Desktop\model\best_crop_recommendation_model.pkl"
model = joblib.load(model_path)

# Initialize Flask app
app = Flask(__name__)  # Fixed the name here

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # Optional HTML interface

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from POST request
    data = request.form if request.form else request.get_json()

    # Extract features as needed for the model
    try:
        # Ensure all inputs are in numeric format (float or int)
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        # Arrange features in the correct order as expected by the model
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict with the model
        prediction = model.predict(features)
        crop_recommendation = prediction[0]

        # Return JSON response
        return jsonify({
            'success': True,
            'crop_recommendation': crop_recommendation
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Run app
if __name__ == "__main__":
    app.run(debug=True)
