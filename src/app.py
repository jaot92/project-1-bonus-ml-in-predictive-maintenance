from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app with correct template folder
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Load the model and scaler
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl'))
scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'best_scaler.pkl'))

# Check if model and scaler exist
if os.path.exists(model_path) and os.path.exists(scaler_path):
    print("Loading best performing model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model loaded successfully!")
else:
    raise Exception("Best model or scaler not found. Please run model comparison first.")

def prepare_input_data(data):
    """
    Prepare input data for prediction
    """
    # Create DataFrame with the required features
    features = pd.DataFrame([data])
    
    # Add engineered features
    features['Temperature_diff'] = features['Process temperature [K]'] - features['Air temperature [K]']
    features['Power'] = (features['Rotational speed [rpm]'] * features['Torque [Nm]']) / 9550
    
    # Ensure correct order of features
    feature_order = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'Temperature_diff',
        'Power'
    ]
    
    return features[feature_order]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Prepare input data
        features = prepare_input_data(data)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': int(prediction),
            'probability': {
                'no_failure': float(prediction_proba[0]),
                'failure': float(prediction_proba[1])
            }
        }
        
        # Add prediction to template context
        if request.headers.get('Content-Type') == 'application/json':
            return jsonify(response)
        else:
            return render_template('index.html', prediction=prediction, probabilities=response['probability'])
            
    except Exception as e:
        error_response = {'status': 'error', 'message': str(e)}
        if request.headers.get('Content-Type') == 'application/json':
            return jsonify(error_response)
        else:
            return render_template('index.html', error=str(e))

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Get port from environment variable or default to 5005
    port = int(os.environ.get('PORT', 5005))
    # Run app
    app.run(host='0.0.0.0', port=port, debug=False) 