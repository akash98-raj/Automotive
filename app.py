from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from dotenv import load_dotenv
import os
import requests
import logging

# Load environment variables
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
TOMTOM_API_KEY = os.getenv('TOMTOM_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

app = Flask(__name__, template_folder='.')
CORS(app)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load and train the model
try:
    csv_data = pd.read_csv('ev_energy_consumption_synthetic_5000.csv')
    X = csv_data[['speed_limit_kmh', 'elevation_gradient', 'weather_temperature_celsius', 'traffic_density']]
    y = csv_data['energy_consumption']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    logger.info("Model trained successfully")
except Exception as e:
    logger.error(f"Error loading/training model: {e}")
    # Create dummy model for testing
    model = None

def get_weather_data(lat, lon):
    """Get weather data from OpenWeather API - using free Current Weather API"""
    if not OPENWEATHER_API_KEY:
        logger.warning("OpenWeather API key not found, using default temperature")
        return {"temperature": 20.0, "humidity": 50, "wind_speed": 0}
    
    try:
        # Using free Current Weather API (not One Call API which requires subscription)
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        main = data.get('main', {})
        wind = data.get('wind', {})
        weather = data.get('weather', [{}])[0]
        
        return {
            "temperature": main.get('temp', 20.0),
            "humidity": main.get('humidity', 50),
            "wind_speed": wind.get('speed', 0),
            "weather_description": weather.get('description', 'clear'),
            "visibility": data.get('visibility', 10000) / 1000  # Convert to km
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API error: {e}")
        logger.info("Using default weather values")
        return {"temperature": 20.0, "humidity": 50, "wind_speed": 0, "weather_description": "clear", "visibility": 10}

def get_elevation_data(lat, lon):
    """Get elevation data from Google Elevation API"""
    if not GOOGLE_MAPS_API_KEY:
        logger.warning("Google Maps API key not found, using default elevation")
        return 0.0
    
    try:
        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={GOOGLE_MAPS_API_KEY}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'OK':
            return data['results'][0]['elevation']
    except requests.exceptions.RequestException as e:
        logger.error(f"Elevation API error: {e}")
    
    return 0.0  # Default elevation

def get_charging_stations(lat, lon, radius=5000):
    """Get EV charging stations using TomTom API"""
    charging_stations = []
    
    if not TOMTOM_API_KEY:
        logger.warning("TomTom API key not found, returning empty charging stations")
        return charging_stations
    
    try:
        # TomTom Places API for EV charging stations
        url = f"https://api.tomtom.com/search/2/poiSearch/electric%20vehicle%20charging%20station.json"
        params = {
            'key': TOMTOM_API_KEY,
            'lat': lat,
            'lon': lon,
            'radius': radius,
            'limit': 10
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        for result in data.get('results', []):
            try:
                station = {
                    'name': result.get('poi', {}).get('name', 'Unknown Station'),
                    'lat': result.get('position', {}).get('lat', 0),
                    'lon': result.get('position', {}).get('lon', 0),
                    'address': result.get('address', {}).get('freeformAddress', 'Unknown Address'),
                    'distance': result.get('dist', 0)
                }
                charging_stations.append(station)
            except KeyError as e:
                logger.warning(f"Missing key in charging station data: {e}")
                continue
                
    except requests.exceptions.RequestException as e:
        logger.error(f"Charging stations API error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching charging stations: {e}")
    
    return charging_stations

def get_route_data(origin, destination):
    """Get route data from Google Directions API"""
    if not GOOGLE_MAPS_API_KEY:
        logger.error("Google Maps API key not found")
        return None
    
    try:
        url = f"https://maps.googleapis.com/maps/api/directions/json"
        params = {
            'origin': origin,
            'destination': destination,
            'key': GOOGLE_MAPS_API_KEY,
            'alternatives': 'true',
            'departure_time': 'now'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] != 'OK':
            logger.error(f"Google Directions API error: {data['status']}")
            return None
            
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Route API error: {e}")
        return None

def calculate_route_features(route_data, origin_coords, dest_coords):
    """Calculate route features for energy consumption prediction"""
    route = route_data['routes'][0]['legs'][0]
    
    # Get weather data for route midpoint
    start_lat, start_lon = origin_coords
    end_lat, end_lon = dest_coords
    mid_lat = (start_lat + end_lat) / 2
    mid_lon = (start_lon + end_lon) / 2
    
    weather_data = get_weather_data(mid_lat, mid_lon)
    weather_temp = weather_data["temperature"]
    
    # Calculate elevation gradient (simplified)
    start_elevation = get_elevation_data(start_lat, start_lon)
    end_elevation = get_elevation_data(end_lat, end_lon)
    distance_km = route['distance']['value'] / 1000
    elevation_gradient = (end_elevation - start_elevation) / (distance_km * 1000) if distance_km > 0 else 0
    
    # Estimate features based on route data
    duration_hours = route['duration']['value'] / 3600
    avg_speed = distance_km / duration_hours if duration_hours > 0 else 50
    
    # Traffic density estimation (simplified)
    duration_in_traffic = route.get('duration_in_traffic', {}).get('value', route['duration']['value'])
    traffic_factor = duration_in_traffic / route['duration']['value'] if route['duration']['value'] > 0 else 1
    traffic_density = min((traffic_factor - 1) * 2, 1.0)  # Normalize to 0-1
    
    # Weather impact factors
    wind_factor = min(weather_data.get("wind_speed", 0) / 10, 1.0)  # Normalize wind speed
    visibility_factor = min(weather_data.get("visibility", 10) / 10, 1.0)  # Normalize visibility
    
    return {
        'speed_limit_kmh': min(avg_speed * 1.2, 130),  # Estimate speed limit
        'elevation_gradient': elevation_gradient,
        'weather_temperature_celsius': weather_temp,
        'traffic_density': traffic_density,
        'distance_km': distance_km,
        'duration_minutes': route['duration']['value'] / 60,
        'weather_conditions': {
            'humidity': weather_data.get("humidity", 50),
            'wind_speed': weather_data.get("wind_speed", 0),
            'description': weather_data.get("weather_description", "clear"),
            'visibility_km': weather_data.get("visibility", 10)
        }
    }

def geocode_address(address):
    """Convert address to coordinates using Google Geocoding API"""
    if not GOOGLE_MAPS_API_KEY:
        return None
    
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': address,
            'key': GOOGLE_MAPS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            return (location['lat'], location['lng'])
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
    
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Original prediction endpoint"""
    logger.info("=== PREDICT ENDPOINT CALLED ===")
    data = request.get_json()
    origin = data.get('origin')
    destination = data.get('destination')

    if not origin or not destination:
        return jsonify({'error': 'Origin and destination are required'}), 400

    # Get route data
    route_data = get_route_data(origin, destination)
    if not route_data:
        return jsonify({'error': 'Failed to fetch route data'}), 500

    # Geocode addresses
    origin_coords = geocode_address(origin)
    dest_coords = geocode_address(destination)
    
    if not origin_coords or not dest_coords:
        return jsonify({'error': 'Failed to geocode addresses'}), 500

    # Calculate features
    features = calculate_route_features(route_data, origin_coords, dest_coords)
    
    # Predict energy consumption
    if model:
        features_df = pd.DataFrame([[
            features['speed_limit_kmh'],
            features['elevation_gradient'],
            features['weather_temperature_celsius'],
            features['traffic_density']
        ]], columns=['speed_limit_kmh', 'elevation_gradient', 'weather_temperature_celsius', 'traffic_density'])
        prediction = model.predict(features_df)[0]
    else:
        # Fallback calculation if model is not available
        prediction = features['distance_km'] * 0.2  # Rough estimate: 0.2 kWh/km
    
    return jsonify({
        'energy_consumption': prediction,
        'features': features
    })

@app.route('/optimize-routes', methods=['POST'])
def optimize_routes():
    """Enhanced route optimization endpoint"""
    logger.info("=== OPTIMIZE ROUTES ENDPOINT CALLED ===")
    
    try:
        data = request.get_json()
        origin = data.get('origin')
        destination = data.get('destination')
        
        logger.info(f"Processing routes from '{origin}' to '{destination}'")

        if not origin or not destination:
            return jsonify({'error': 'Origin and destination are required'}), 400

        # Get route data with alternatives
        route_data = get_route_data(origin, destination)
        if not route_data:
            return jsonify({'error': 'Failed to fetch route data'}), 500

        # Geocode addresses
        origin_coords = geocode_address(origin)
        dest_coords = geocode_address(destination)
        
        if not origin_coords or not dest_coords:
            return jsonify({'error': 'Failed to geocode addresses'}), 500

        # Process all route alternatives
        routes = []
        for i, route in enumerate(route_data['routes'][:3]):  # Limit to 3 routes
            try:
                # Create temporary route data for feature calculation
                temp_route_data = {'routes': [route]}
                features = calculate_route_features(temp_route_data, origin_coords, dest_coords)
                
                # Predict energy consumption
                if model:
                    features_df = pd.DataFrame([[
                        features['speed_limit_kmh'],
                        features['elevation_gradient'],
                        features['weather_temperature_celsius'],
                        features['traffic_density']
                    ]], columns=['speed_limit_kmh', 'elevation_gradient', 'weather_temperature_celsius', 'traffic_density'])
                    energy_consumption = float(model.predict(features_df)[0])
                else:
                    energy_consumption = float(features['distance_km'] * 0.2)
                
                route_info = {
                    'route_id': i,
                    'distance_km': float(features['distance_km']),
                    'duration_minutes': float(features['duration_minutes']),
                    'energy_consumption': energy_consumption,
                    'features': {
                        'speed_limit_kmh': float(features['speed_limit_kmh']),
                        'elevation_gradient': float(features['elevation_gradient']),
                        'weather_temperature_celsius': float(features['weather_temperature_celsius']),
                        'traffic_density': float(features['traffic_density']),
                        'weather_conditions': {
                            'humidity': float(features['weather_conditions'].get('humidity', 50)),
                            'wind_speed': float(features['weather_conditions'].get('wind_speed', 0)),
                            'description': str(features['weather_conditions'].get('description', 'clear')),
                            'visibility_km': float(features['weather_conditions'].get('visibility_km', 10))
                        }
                    },
                    'polyline': str(route['overview_polyline']['points'])
                }
                routes.append(route_info)
                
            except Exception as e:
                logger.error(f"Error processing route {i}: {e}")
                continue

        # Get charging stations near the destination
        charging_stations = get_charging_stations(dest_coords[0], dest_coords[1])
        
        logger.info(f"Returning {len(routes)} routes and {len(charging_stations)} charging stations")
        
        return jsonify({
            'routes': routes,
            'charging_stations': charging_stations,
            'origin_coords': origin_coords,
            'destination_coords': dest_coords
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in optimize_routes: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("Features enabled:")
    print(f"- Real weather data (OpenWeather): {'✓' if OPENWEATHER_API_KEY else '✗'}")
    print(f"- Real elevation data (Google Elevation): {'✓' if GOOGLE_MAPS_API_KEY else '✗'}")
    print(f"- Real traffic data (Google Traffic): {'✓' if GOOGLE_MAPS_API_KEY else '✗'}")
    print(f"- Multiple route alternatives: {'✓' if GOOGLE_MAPS_API_KEY else '✗'}")
    print(f"- EV charging station locations: {'✓' if TOMTOM_API_KEY else '✗'}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)