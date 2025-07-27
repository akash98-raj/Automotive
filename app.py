from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
TOMTOM_API_KEY = os.getenv('TOMTOM_API_KEY')

app = Flask(__name__, template_folder='.')  # Serve templates from current directory
CORS(app)  # Enable CORS for all routes

# Load and train the model
csv_data = pd.read_csv('ev_energy_consumption_synthetic_5000.csv')
X = csv_data[['speed_limit_kmh', 'elevation_gradient', 'weather_temperature_celsius', 'traffic_density']]
y = csv_data['energy_consumption']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

def get_route_features(origin, destination):
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(url).json()
    if response['status'] == 'OK':
        route = response['routes'][0]['legs'][0]
        distance = route['distance']['value'] / 1000
        speed_limit = np.random.uniform(40, 120)
        elevation = np.random.uniform(-0.1, 0.1)
        traffic_density = np.random.uniform(0, 0.6)
        weather_temp = np.random.uniform(0, 35)
        return {
            'speed_limit_kmh': speed_limit,
            'elevation_gradient': elevation,
            'weather_temperature_celsius': weather_temp,
            'traffic_density': traffic_density
        }
    return None

@app.route('/')
def home():
    return render_template('index.html')  # Serve index.html

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request:", request.get_json())  # Debug log
    data = request.get_json()
    origin = data.get('origin')
    destination = data.get('destination')

    if not origin or not destination:
        return jsonify({'error': 'Origin and destination are required'}), 400

    features = get_route_features(origin, destination)
    if features:
        features_array = np.array([[features['speed_limit_kmh'], features['elevation_gradient'],
                                  features['weather_temperature_celsius'], features['traffic_density']]])
        prediction = model.predict(features_array)[0]
        return jsonify({'energy_consumption': prediction, 'features': features})
    return jsonify({'error': 'Failed to fetch route data'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)