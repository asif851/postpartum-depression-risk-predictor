from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Load models and preprocessor
MODEL_PATH = 'models'
anxiety_model = joblib.load(os.path.join(MODEL_PATH, 'anxiety_model.pkl'))
suicide_model = joblib.load(os.path.join(MODEL_PATH, 'suicide_model.pkl'))
preprocessor = joblib.load(os.path.join(MODEL_PATH, 'preprocessor.pkl'))

# Extract preprocessor components
scaler = preprocessor['scaler']
encoders = preprocessor['encoders']
feature_names = preprocessor['feature_names']
categorical_cols = preprocessor['categorical_cols']

def preprocess_patient_data(patient_data):
    """
    Preprocess patient data for prediction
    
    Parameters:
    patient_data: dict with keys matching the input form
    """
    # Convert to DataFrame
    df_pred = pd.DataFrame([patient_data])
    
    # Extract age numeric value
    if 'Age' in patient_data:
        age_range = patient_data['Age'].split('-')
        age_num = (int(age_range[0]) + int(age_range[1])) / 2
    else:
        age_num = 35  # default median age
    
    df_pred['Age_num'] = age_num
    
    # Extract hour from timestamp
    if 'Timestamp' in patient_data:
        hour = pd.to_datetime(patient_data['Timestamp']).hour
    else:
        hour = 12  # default noon
    
    df_pred['Hour'] = hour
    
    # Encode categorical features
    for col in categorical_cols:
        if col in patient_data:
            value = patient_data[col]
            # Handle unknown values
            if value in encoders[col].classes_:
                df_pred[col + '_enc'] = encoders[col].transform([value])[0]
            else:
                df_pred[col + '_enc'] = -1
        else:
            df_pred[col + '_enc'] = -1
    
    # Prepare features in correct order
    X_pred = df_pred[feature_names].copy()
    
    # Scale numerical features
    X_pred[['Age_num', 'Hour']] = scaler.transform(X_pred[['Age_num', 'Hour']])
    
    return X_pred

def get_risk_level(probability):
    """Convert probability to risk level"""
    if probability >= 0.7:
        return 'HIGH'
    elif probability >= 0.4:
        return 'MODERATE'
    else:
        return 'LOW'

@app.route('/')
def index():
    """Render the main form page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        patient_data = {
            'Timestamp': request.form.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M')),
            'Age': request.form.get('age', '35-40'),
            'Sad': request.form.get('sad', 'No'),
            'Irritable': request.form.get('irritable', 'No'),
            'Sleep': request.form.get('sleep', 'No'),
            'Concentration': request.form.get('concentration', 'No'),
            'Appetite': request.form.get('appetite', 'No'),
            'Anxious': request.form.get('anxious', 'No'),
            'Guilt': request.form.get('guilt', 'No'),
            'Bonding': request.form.get('bonding', 'No'),
            'Suicide': request.form.get('suicide', 'No')
        }
        
        # Preprocess data
        X_pred = preprocess_patient_data(patient_data)
        
        # Make predictions
        anxiety_pred = anxiety_model.predict(X_pred)[0]
        anxiety_proba = anxiety_model.predict_proba(X_pred)[0]
        
        suicide_pred = suicide_model.predict(X_pred)[0]
        suicide_proba = suicide_model.predict_proba(X_pred)[0]
        
        # Calculate overall risk
        overall_score = (anxiety_proba[1] * 0.4 + suicide_proba[1] * 0.6)
        
        # Prepare results
        results = {
            'anxiety_prediction': 'High Risk' if anxiety_pred == 1 else 'Low Risk',
            'anxiety_probability': f"{anxiety_proba[1]:.2%}",
            'suicide_prediction': 'High Risk' if suicide_pred == 1 else 'Low Risk',
            'suicide_probability': f"{suicide_proba[1]:.2%}",
            'overall_risk': get_risk_level(overall_score),
            'overall_score': f"{overall_score:.2%}",
            'patient_data': patient_data
        }
        
        return render_template('result.html', results=results)
        
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['Age', 'Sad', 'Irritable', 'Sleep', 'Concentration', 
                          'Appetite', 'Anxious', 'Guilt', 'Bonding', 'Suicide']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Add timestamp if not provided
        if 'Timestamp' not in data:
            data['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # Preprocess and predict
        X_pred = preprocess_patient_data(data)
        
        anxiety_pred = anxiety_model.predict(X_pred)[0]
        anxiety_proba = anxiety_model.predict_proba(X_pred)[0]
        
        suicide_pred = suicide_model.predict(X_pred)[0]
        suicide_proba = suicide_model.predict_proba(X_pred)[0]
        
        overall_score = (anxiety_proba[1] * 0.4 + suicide_proba[1] * 0.6)
        
        response = {
            'success': True,
            'predictions': {
                'anxiety': {
                    'prediction': 'High Risk' if anxiety_pred == 1 else 'Low Risk',
                    'probability': float(anxiety_proba[1]),
                    'probability_percent': f"{anxiety_proba[1]:.2%}"
                },
                'suicide': {
                    'prediction': 'High Risk' if suicide_pred == 1 else 'Low Risk',
                    'probability': float(suicide_proba[1]),
                    'probability_percent': f"{suicide_proba[1]:.2%}"
                },
                'overall': {
                    'risk_level': get_risk_level(overall_score),
                    'score': float(overall_score),
                    'score_percent': f"{overall_score:.2%}"
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)