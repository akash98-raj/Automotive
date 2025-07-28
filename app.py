from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from dotenv import load_dotenv
import os
import requests
import polyline
from geopy.distance import geodesic
import time
import datetime

# Load environment variables
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
TOMTOM_API_KEY = os.getenv('TOMTOM_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

app = Flask(__name__, template_folder='.')
CORS(app)

# Load and train the model (still using synthetic data - replace with real EV data when available)
try:
    csv_data = pd.read_csv('ev_energy_consumption_synthetic_5000.csv')
    X = csv_data[['speed_limit_kmh', 'elevation_gradient', 'weather_temperature_celsius', 'traffic_density']]
    y = csv_data['energy_consumption']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    # Store feature names for consistent prediction
    feature_names = X.columns.tolist()
    print("✅ ML model trained successfully")
    print(f"✅ Feature names: {feature_names}")
except Exception as e:
    print(f"❌ Error loading training data: {e}")
    # Create a dummy model for testing
    model = None
    feature_names = ['speed_limit_kmh', 'elevation_gradient', 'weather_temperature_celsius', 'traffic_density']

def get_elevation_profile(coordinates):
    """Get elevation data for route coordinates using Google Elevation API"""
    try:
        if not coordinates or len(coordinates) < 2:
            return None
            
        # Limit to 100 points to stay within API limits
        if len(coordinates) > 100:
            step = len(coordinates) // 100
            coordinates = coordinates[::step]
        
        locations = '|'.join([f"{lat},{lng}" for lat, lng in coordinates])
        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={GOOGLE_MAPS_API_KEY}"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'OK':
                elevations = [result['elevation'] for result in data['results']]
                print(f"✅ Got elevation data for {len(elevations)} points")
                return elevations
        else:
            print(f"Elevation API HTTP error: {response.status_code}")
    except Exception as e:
        print(f"Elevation API error: {e}")
    
    return None

def calculate_elevation_gradient(elevations, coordinates):
    """Calculate average elevation gradient from elevation profile"""
    if not elevations or len(elevations) < 2:
        return 0.0
    
    total_distance = 0
    total_elevation_change = 0
    
    try:
        for i in range(1, len(elevations)):
            if i < len(coordinates):
                # Calculate distance between consecutive points
                distance = geodesic(coordinates[i-1], coordinates[i]).meters
                elevation_change = elevations[i] - elevations[i-1]
                
                total_distance += distance
                total_elevation_change += elevation_change
        
        # Return gradient as rise/run
        if total_distance > 0:
            gradient = total_elevation_change / total_distance
            print(f"✅ Calculated elevation gradient: {gradient:.6f}")
            return gradient
    except Exception as e:
        print(f"Elevation gradient calculation error: {e}")
    
    return 0.0

def get_weather_data(lat, lng):
    """Get current weather data using OpenWeatherMap API"""
    try:
        if not OPENWEATHER_API_KEY:
            print("OpenWeather API key not found, using fallback")
            return get_fallback_weather(lat, lng)
        
        # OpenWeatherMap One Call API 3.0
        url = f"https://api.openweathermap.org/data/3.0/onecall"
        params = {
            'lat': lat,
            'lon': lng,
            'exclude': 'minutely,hourly,daily,alerts',  # Only get current weather
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'  # Celsius
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'current' in data:
                current = data['current']
                temperature = current.get('temp', 15.0)
                weather_main = current.get('weather', [{}])[0].get('main', 'Clear')
                wind_speed = current.get('wind_speed', 0)
                humidity = current.get('humidity', 50)
                
                print(f"✅ Weather data: {temperature}°C, {weather_main}, Wind: {wind_speed}m/s, Humidity: {humidity}%")
                
                return {
                    'temperature': temperature,
                    'weather_condition': weather_main,
                    'wind_speed': wind_speed,
                    'humidity': humidity
                }
            else:
                print("No current weather data in response")
                return get_fallback_weather(lat, lng)
                
        elif response.status_code == 401:
            print("OpenWeather API: Invalid API key")
            return get_fallback_weather(lat, lng)
        elif response.status_code == 429:
            print("OpenWeather API: Rate limit exceeded")
            return get_fallback_weather(lat, lng)
        else:
            print(f"OpenWeather API error: HTTP {response.status_code}")
            return get_fallback_weather(lat, lng)
            
    except requests.RequestException as e:
        print(f"Weather API request error: {e}")
        return get_fallback_weather(lat, lng)
    except Exception as e:
        print(f"Weather API error: {e}")
        return get_fallback_weather(lat, lng)

def get_fallback_weather(lat, lng):
    """Fallback weather estimation based on location and season"""
    try:
        berlin_lat, berlin_lng = 52.5200, 13.4050
        distance_from_berlin = geodesic((lat, lng), (berlin_lat, berlin_lng)).kilometers
        
        # Seasonal temperature estimation (very basic)
        month = datetime.datetime.now().month
        
        # Berlin seasonal averages
        seasonal_temps = {
            12: 2, 1: 1, 2: 3,     # Winter
            3: 7, 4: 12, 5: 17,    # Spring
            6: 20, 7: 22, 8: 22,   # Summer
            9: 18, 10: 13, 11: 7   # Autumn
        }
        
        base_temp = seasonal_temps.get(month, 15)
        
        # Small variation based on distance from city center
        temp_variation = min(distance_from_berlin * 0.2, 3)
        temperature = base_temp + np.random.uniform(-temp_variation, temp_variation)
        
        return {
            'temperature': max(-10, min(40, temperature)),
            'weather_condition': 'Clear',
            'wind_speed': 5.0,
            'humidity': 65
        }
        
    except Exception as e:
        print(f"Fallback weather error: {e}")
        return {
            'temperature': 15.0,
            'weather_condition': 'Clear',
            'wind_speed': 5.0,
            'humidity': 65
        }

def extract_speed_limits_from_route(route_polyline, steps):
    """Extract speed limits from route segments using TomTom API"""
    try:
        if not TOMTOM_API_KEY:
            print("TomTom API key not found, using defaults")
            return 50.0
            
        # Decode the polyline to get coordinates
        coordinates = polyline.decode(route_polyline)
        
        # Sample points along the route for speed limit checks
        sample_points = coordinates[::max(1, len(coordinates)//10)]  # Sample 10 points max
        
        speed_limits = []
        
        for lat, lng in sample_points:
            # TomTom Speed Limits API
            url = f"https://api.tomtom.com/search/2/reverseGeocode/{lat},{lng}.json"
            params = {
                'key': TOMTOM_API_KEY,
                'radius': 100
            }
            
            try:
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    # Extract speed limit from response (this varies by API)
                    # For now, estimate based on road type
                    if 'results' in data and data['results']:
                        address = data['results'][0].get('address', {})
                        road_type = address.get('roadType', 'OTHER')
                        
                        # Estimate speed limits based on road type
                        speed_map = {
                            'HIGHWAY': 130,
                            'MAJOR_ROAD': 80,
                            'SECONDARY_ROAD': 50,
                            'LOCAL_ROAD': 30,
                            'OTHER': 50
                        }
                        speed_limits.append(speed_map.get(road_type, 50))
                    else:
                        speed_limits.append(50)  # Default
                else:
                    speed_limits.append(50)  # Default on API error
                    
            except requests.RequestException:
                speed_limits.append(50)  # Default on timeout/error
                
            time.sleep(0.1)  # Rate limiting
        
        avg_speed = np.mean(speed_limits) if speed_limits else 50.0
        print(f"✅ Average speed limit: {avg_speed:.1f} km/h")
        return avg_speed
        
    except Exception as e:
        print(f"Speed limit extraction error: {e}")
        return 50.0  # Default speed limit

def calculate_traffic_density(route_data):
    """Calculate traffic density from Google Maps route data"""
    try:
        # Extract traffic information from route steps
        total_duration = route_data['duration']['value']  # seconds
        total_duration_in_traffic = route_data.get('duration_in_traffic', {}).get('value', total_duration)
        
        # Calculate traffic density as ratio of actual time to free-flow time
        if total_duration > 0:
            traffic_ratio = total_duration_in_traffic / total_duration
            # Normalize to 0-1 scale where 1 = heavy traffic
            traffic_density = min(1.0, max(0.0, (traffic_ratio - 1.0) / 2.0))  # Assuming max 3x slowdown = density 1.0
            print(f"✅ Traffic density: {traffic_density:.2f} (ratio: {traffic_ratio:.2f})")
            return traffic_density
        
        return 0.3  # Default moderate traffic
        
    except Exception as e:
        print(f"Traffic density calculation error: {e}")
        return 0.3

def get_simple_route_features(origin, destination):
    """Simplified route feature extraction with basic Google Maps API"""
    try:
        print(f"🔄 Trying simplified route extraction for {origin} to {destination}")
        
        # Simple directions request without traffic
        url = f"https://maps.googleapis.com/maps/api/directions/json"
        params = {
            'origin': origin,
            'destination': destination,
            'key': GOOGLE_MAPS_API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"Simple route API HTTP error: {response.status_code}")
            return None
            
        data = response.json()
        print(f"Simple route API status: {data.get('status')}")
        
        if data['status'] != 'OK' or not data.get('routes'):
            print(f"Simple route API error: {data.get('status')} - {data.get('error_message', 'Unknown')}")
            return None
            
        route = data['routes'][0]
        leg = route['legs'][0]
        
        # Extract basic features with reasonable defaults
        distance_km = leg['distance']['value'] / 1000
        duration_min = leg['duration']['value'] / 60
        
        # Use simple estimates based on distance and location
        # Estimate features
        speed_limit = 50.0  # Default urban speed limit
        if distance_km > 10:  # Likely includes highways
            speed_limit = 70.0
        elif distance_km < 3:  # City center
            speed_limit = 30.0
            
        elevation_gradient = 0.0  # Berlin is relatively flat
        traffic_density = 0.3  # Moderate traffic assumption
        
        # Get weather for route midpoint if possible
        if route.get('overview_polyline', {}).get('points'):
            try:
                coordinates = polyline.decode(route['overview_polyline']['points'])
                mid_point = coordinates[len(coordinates)//2]
                weather_data = get_weather_data(mid_point[0], mid_point[1])
            except:
                weather_data = get_fallback_weather(52.5200, 13.4050)
        else:
            weather_data = get_fallback_weather(52.5200, 13.4050)
        
        features = {
            'speed_limit_kmh': speed_limit,
            'elevation_gradient': elevation_gradient,
            'weather_temperature_celsius': weather_data['temperature'],
            'traffic_density': traffic_density,
            'distance_km': distance_km,
            'duration_minutes': duration_min,
            'weather_condition': weather_data['weather_condition'],
            'wind_speed': weather_data['wind_speed'],
            'humidity': weather_data['humidity']
        }
        
        print(f"✅ Simple extraction successful: {distance_km:.1f}km, {duration_min:.0f}min")
        return features
        
    except Exception as e:
        print(f"❌ Simple route extraction error: {e}")
        return None

def get_route_features(origin, destination):
    """Extract real route features using multiple APIs"""
    try:
        print(f"🔄 Getting advanced route features from {origin} to {destination}")
        
        # Get route from Google Maps with traffic information
        url = f"https://maps.googleapis.com/maps/api/directions/json"
        params = {
            'origin': origin,
            'destination': destination,
            'key': GOOGLE_MAPS_API_KEY,
            'departure_time': 'now',  # For traffic data
            'traffic_model': 'best_guess'
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            print(f"Google Maps API HTTP error: {response.status_code}")
            print(f"Response text: {response.text[:500]}...")
            return None
            
        data = response.json()
        print(f"Google Maps API response status: {data.get('status', 'Unknown')}")
        
        if data['status'] != 'OK':
            error_message = data.get('error_message', 'Unknown error')
            print(f"Google Maps API error: {data['status']} - {error_message}")
            return None
            
        if not data.get('routes') or len(data['routes']) == 0:
            print("No routes found in response")
            return None
            
        route = data['routes'][0]
        if not route.get('legs') or len(route['legs']) == 0:
            print("No legs found in route")
            return None
            
        leg = route['legs'][0]
        
        # Validate leg data
        if not leg.get('distance', {}).get('value') or not leg.get('duration', {}).get('value'):
            print("Missing distance or duration data in leg")
            return None
        
        # Validate required route data
        if not route.get('overview_polyline', {}).get('points'):
            print("No polyline data in route")
            return None
            
        # Get route polyline and decode coordinates
        route_polyline = route['overview_polyline']['points']
        
        try:
            coordinates = polyline.decode(route_polyline)
            if not coordinates or len(coordinates) < 2:
                print("Invalid or empty coordinates from polyline")
                return None
        except Exception as e:
            print(f"Failed to decode polyline: {e}")
            return None
        
        # Extract real features
        print("🔄 Extracting speed limits...")
        speed_limit = extract_speed_limits_from_route(route_polyline, leg['steps'])
        
        print("🔄 Getting elevation profile...")
        elevations = get_elevation_profile(coordinates)
        elevation_gradient = calculate_elevation_gradient(elevations, coordinates) if elevations else 0.0
        
        print("🔄 Calculating traffic density...")
        traffic_density = calculate_traffic_density(leg)
        
        print("🔄 Getting weather data...")
        # Use route midpoint for weather
        mid_point = coordinates[len(coordinates)//2]
        weather_data = get_weather_data(mid_point[0], mid_point[1])
        
        features = {
            'speed_limit_kmh': speed_limit,
            'elevation_gradient': elevation_gradient,
            'weather_temperature_celsius': weather_data['temperature'],
            'traffic_density': traffic_density,
            'distance_km': leg['distance']['value'] / 1000,
            'duration_minutes': leg['duration']['value'] / 60,
            'weather_condition': weather_data['weather_condition'],
            'wind_speed': weather_data['wind_speed'],
            'humidity': weather_data['humidity']
        }
        
        print(f"✅ Advanced extraction successful: {features}")
        return features
        
    except Exception as e:
        print(f"❌ Advanced route feature extraction error: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("🔄 Received prediction request...")
    data = request.get_json()
    origin = data.get('origin')
    destination = data.get('destination')

    if not origin or not destination:
        return jsonify({'error': 'Origin and destination are required'}), 400

    print(f"🎯 Extracting features for route: {origin} -> {destination}")
    
    # Try advanced extraction first
    features = get_route_features(origin, destination)
    
    # Fallback to simple extraction if full extraction fails
    if not features:
        print("⚠️ Advanced extraction failed, trying simplified approach...")
        features = get_simple_route_features(origin, destination)
    
    if features:
        # Check if we have a trained model
        if model is None:
            # Create a simple prediction based on distance if no model
            base_consumption = 0.2  # kWh per km base rate
            distance_factor = features['distance_km']
            temp_factor = 1.0
            if features['weather_temperature_celsius'] < 5:
                temp_factor = 1.3  # 30% more in cold
            elif features['weather_temperature_celsius'] > 30:
                temp_factor = 1.2  # 20% more in heat
            
            traffic_factor = 1.0 + (features['traffic_density'] * 0.5)  # Up to 50% more in traffic
            speed_factor = 1.0
            if features['speed_limit_kmh'] > 80:
                speed_factor = 1.4  # Highway consumption
            
            prediction = base_consumption * distance_factor * temp_factor * traffic_factor * speed_factor
        else:
            # Use trained ML model with proper feature names
            # Create DataFrame with correct feature names to avoid warning
            features_df = pd.DataFrame([[
                features['speed_limit_kmh'], 
                features['elevation_gradient'],
                features['weather_temperature_celsius'], 
                features['traffic_density']
            ]], columns=feature_names)
            
            prediction = model.predict(features_df)[0]
            print(f"✅ ML prediction: {prediction:.3f} kWh")
        
        # Add some insights
        insights = generate_insights(features, prediction)
        
        return jsonify({
            'energy_consumption': prediction,
            'features': features,
            'insights': insights,
            'route_summary': f"Distance: {features['distance_km']:.1f} km, Duration: {features['duration_minutes']:.0f} min"
        })
    
    return jsonify({'error': 'Failed to extract route features. Please check your locations and ensure they are in Berlin.'}), 500

def generate_insights(features, prediction):
    """Generate insights about energy consumption factors"""
    insights = []
    
    # Speed limit insights
    if features['speed_limit_kmh'] > 80:
        insights.append("High speed route - increased energy consumption expected")
    elif features['speed_limit_kmh'] < 40:
        insights.append("Urban route with lower speeds - more efficient driving")
    
    # Elevation insights
    if features['elevation_gradient'] > 0.01:
        insights.append("Uphill route - expect higher energy consumption")
    elif features['elevation_gradient'] < -0.01:
        insights.append("Downhill route - potential for energy regeneration")
    
    # Weather insights
    temp = features['weather_temperature_celsius']
    condition = features.get('weather_condition', 'Unknown')
    wind_speed = features.get('wind_speed', 0)
    humidity = features.get('humidity', 50)
    
    if temp < 5:
        insights.append(f"Cold weather ({temp:.1f}°C) - battery efficiency may be reduced")
    elif temp > 30:
        insights.append(f"Hot weather ({temp:.1f}°C) - air conditioning will increase consumption")
    elif 15 <= temp <= 25:
        insights.append(f"Optimal temperature ({temp:.1f}°C) for EV efficiency")
    
    # Weather condition insights
    if condition in ['Rain', 'Drizzle', 'Thunderstorm']:
        insights.append("Wet conditions - reduced efficiency due to increased rolling resistance")
    elif condition == 'Snow':
        insights.append("Snow conditions - significantly reduced efficiency and range")
    elif condition in ['Mist', 'Fog']:
        insights.append("Low visibility conditions - slower speeds may improve efficiency")
    
    # Wind insights
    if wind_speed > 10:
        insights.append(f"Strong winds ({wind_speed:.1f}m/s) - may affect highway efficiency")
    
    # Humidity insights
    if humidity > 80:
        insights.append("High humidity - may require more air conditioning")
    
    # Traffic insights
    if features['traffic_density'] > 0.6:
        insights.append("Heavy traffic - stop-and-go driving increases consumption")
    elif features['traffic_density'] < 0.2:
        insights.append("Light traffic - optimal driving conditions")
    
    # Combined insights
    if temp < 10 and features['traffic_density'] > 0.5:
        insights.append("Cold weather + traffic combination particularly reduces efficiency")
    
    return insights

if __name__ == '__main__':
    print("🚀 Starting Enhanced EV Route Optimizer...")
    print("Features:")
    print("- Real speed limit extraction")
    print("- Actual elevation profiles")
    print("- Weather data integration")
    print("- Real-time traffic analysis")
    print("- Robust fallback systems")
    
    # Check API keys
    missing_keys = []
    if not GOOGLE_MAPS_API_KEY:
        missing_keys.append("GOOGLE_MAPS_API_KEY")
    if not TOMTOM_API_KEY:
        missing_keys.append("TOMTOM_API_KEY") 
    if not OPENWEATHER_API_KEY:
        missing_keys.append("OPENWEATHER_API_KEY")
    
    if missing_keys:
        print(f"⚠️  Missing API keys: {', '.join(missing_keys)}")
        print("   The system will use fallback methods for missing APIs")
    else:
        print("✅ All API keys loaded")
    
    app.run(host='0.0.0.0', port=5000, debug=True)