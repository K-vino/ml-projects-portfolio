"""
🚀 ML Model Deployment with Flask
A complete example of deploying machine learning models to production
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and scaler
model = None
scaler = None
feature_names = None

def train_and_save_model():
    """Train a simple model and save it for deployment"""
    logger.info("Training model...")
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'model.pkl')
    
    # Save feature names
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(iris.feature_names, f)
    
    # Save target names
    with open('target_names.pkl', 'wb') as f:
        pickle.dump(iris.target_names, f)
    
    accuracy = model.score(X_test, y_test)
    logger.info(f"Model trained with accuracy: {accuracy:.3f}")
    
    return model, iris.feature_names, iris.target_names

def load_model():
    """Load the trained model and associated data"""
    global model, feature_names
    
    try:
        # Load model
        model = joblib.load('model.pkl')
        
        # Load feature names
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Load target names
        with open('target_names.pkl', 'rb') as f:
            target_names = pickle.load(f)
        
        logger.info("Model loaded successfully")
        return model, feature_names, target_names
    
    except FileNotFoundError:
        logger.info("Model not found, training new model...")
        return train_and_save_model()

# Load model on startup
model, feature_names, target_names = load_model()

@app.route('/')
def home():
    """Home page with API documentation"""
    return render_template('index.html', 
                         feature_names=feature_names,
                         target_names=target_names)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on input data"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input
        if 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        features = data['features']
        
        # Validate feature count
        if len(features) != len(feature_names):
            return jsonify({
                'error': f'Expected {len(feature_names)} features, got {len(features)}'
            }), 400
        
        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        prediction_proba = model.predict_proba(features_array)[0]
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'prediction_label': target_names[prediction],
            'probabilities': {
                target_names[i]: float(prob) 
                for i, prob in enumerate(prediction_proba)
            },
            'confidence': float(max(prediction_proba)),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log prediction
        logger.info(f"Prediction made: {response['prediction_label']} "
                   f"(confidence: {response['confidence']:.3f})")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Make predictions on multiple samples"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
        
        samples = data['samples']
        
        # Validate samples
        for i, sample in enumerate(samples):
            if len(sample) != len(feature_names):
                return jsonify({
                    'error': f'Sample {i}: Expected {len(feature_names)} features, got {len(sample)}'
                }), 400
        
        # Convert to numpy array
        features_array = np.array(samples)
        
        # Make predictions
        predictions = model.predict(features_array)
        predictions_proba = model.predict_proba(features_array)
        
        # Prepare response
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, predictions_proba)):
            results.append({
                'sample_id': i,
                'prediction': int(pred),
                'prediction_label': target_names[pred],
                'probabilities': {
                    target_names[j]: float(prob) 
                    for j, prob in enumerate(proba)
                },
                'confidence': float(max(proba))
            })
        
        response = {
            'results': results,
            'total_samples': len(samples),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch prediction made for {len(samples)} samples")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Get information about the loaded model"""
    try:
        info = {
            'model_type': type(model).__name__,
            'feature_names': feature_names,
            'target_names': list(target_names),
            'n_features': len(feature_names),
            'n_classes': len(target_names),
            'model_params': model.get_params() if hasattr(model, 'get_params') else 'Not available'
        }
        
        return jsonify(info)
    
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
