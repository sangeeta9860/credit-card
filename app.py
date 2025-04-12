from flask import Flask, render_template, request, jsonify 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os

app = Flask(__name__)

# Initialize model and scaler
model = None
scaler = None

def train_model():
    global model, scaler
    print("Training new model...")
    
    try:
        # Load and preprocess data
        import pandas as pd
        file_path = r'C:\Users\sange\OneDrive\Desktop\CARDIO DISEASE PREDICTER\cardio disease predicter\cardio_train.csv'
        # Read the CSV file using pandas
        data = pd.read_csv(file_path, sep=';')
        data['age'] = data['age'] / 365  # Convert age to years
        data = data[(data['ap_hi'] >= 80) & (data['ap_hi'] <= 200)]
        data = data[(data['ap_lo'] >= 50) & (data['ap_lo'] <= 120)]
        data['bmi'] = data['weight'] / (data['height']/100)**2
        print(f"Processed data shape: {data.shape}")
        
        # Prepare features and target
        X = data[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']]
        y = data['cardio']
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Test prediction
        test_pred = model.predict(X_scaled[:1])
        print(f"Test prediction: {test_pred[0]}, should be: {y.iloc[0]}")
        
        return True
    except Exception as e:
        print(f"Error in training: {str(e)}")
        return False

# Load or train model when starting
print("Initializing model...")
if os.path.exists('cardio_model.pkl') and os.path.exists('scaler.pkl'):
    try:
        model = pickle.load(open('cardio_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        print("Loaded existing model")
    except:
        print("Error loading model, training new one")
        if train_model():
            pickle.dump(model, open('cardio_model.pkl', 'wb'))
            pickle.dump(scaler, open('scaler.pkl', 'wb'))
else:
    if train_model():
        pickle.dump(model, open('cardio_model.pkl', 'wb'))
        pickle.dump(scaler, open('scaler.pkl', 'wb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)
        
        # Validate data
        required_fields = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Prepare features
        features = [
            float(data['age']),
            int(data['gender']),
            float(data['height']),
            float(data['weight']),
            float(data['ap_hi']),
            float(data['ap_lo']),
            int(data['cholesterol']),
            int(data['gluc']),
            int(data.get('smoke', 0)),
            int(data.get('alco', 0)),
            int(data.get('active', 1)),
            float(data['weight']) / (float(data['height'])/100)**2  # BMI
        ]
        print("Processed features:", features)
        
        # Scale features and predict
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        print(f"Prediction: {prediction}, Probability: {probability}")
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'message': 'High risk of cardiovascular disease' if prediction == 1 
                    else 'Low risk of cardiovascular disease'
        })
    
    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)