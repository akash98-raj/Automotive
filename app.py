from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from dotenv import load_dotenv
import os
import requests
import json
from datetime import datetime

# Load environment variables
load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
TOMTOM_API_KEY = os.getenv('TOMTOM_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

app = Flask(__name__, template_folder='.')
CORS(app)

# Load and train the model
csv_data = pd.read_csv('ev_energy_consumption_synthetic_5000.csv')
X = csv_data[['speed_limit_kmh', 'elevation_gradient', 'weather_temperature_celsius', 'traffic_density']]
y = csv_data['energy_consumption']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

def get_weather_data(lat, lon):
    """Get real weather data from OpenWeather API"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data['main']['temp']
    except Exception as e:
        print(f"Weather API error: {e}")
        return 15.0  # Default temperature

def get_elevation_data(lat, lon):
    """Get elevation data from Google Elevation API"""
    try:
        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={GOOGLE_MAPS_API_KEY}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'OK' and data['results']:
            return data['results'][0]['elevation']
    except Exception as e:
        print(f"Elevation API error: {e}")
    return 50.0  # Default elevation

def calculate_elevation_gradient(route_points):
    """Calculate elevation gradient along the route"""
    if len(route_points) < 2:
        return 0.0
    
    elevations = []
    for point in route_points[::max(1, len(route_points)//10)]:  # Sample 10 points max
        elevation = get_elevation_data(point['lat'], point['lng'])
        elevations.append(elevation)
    
    if len(elevations) < 2:
        return 0.0
    
    # Calculate average gradient
    total_elevation_change = elevations[-1] - elevations[0]
    # Approximate distance (this is simplified)
    return total_elevation_change / 1000  # Gradient per km

def get_traffic_density(route_polyline):
    """Get traffic data from Google Maps Traffic"""
    try:
        # Use departure time to get current traffic
        departure_time = int(datetime.now().timestamp())
        url = f"https://maps.googleapis.com/maps/api/directions/json"
        params = {
            'origin': route_polyline[0] if route_polyline else '52.5200,13.4050',
            'destination': route_polyline[-1] if route_polyline else '52.5200,13.4050',
            'departure_time': departure_time,
            'traffic_model': 'best_guess',
            'key': GOOGLE_MAPS_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data['status'] == 'OK':
            route = data['routes'][0]['legs'][0]
            duration_in_traffic = route.get('duration_in_traffic', route['duration'])['value']
            normal_duration = route['duration']['value']
            
            # Calculate traffic density ratio
            traffic_ratio = duration_in_traffic / normal_duration if normal_duration > 0 else 1.0
            return min(max((traffic_ratio - 1) * 2, 0), 1)  # Normalize to 0-1
    except Exception as e:
        print(f"Traffic API error: {e}")
    
    return 0.3  # Default moderate traffic

def get_charging_stations(bounds):
    """Get EV charging stations in the area using Google Places API with expanded search"""
    try:
        # Calculate center and radius from bounds
        center_lat = (bounds['north'] + bounds['south']) / 2
        center_lng = (bounds['east'] + bounds['west']) / 2
        
        # Calculate radius based on bounds (to cover the entire route area)
        from math import radians, sin, cos, sqrt, atan2
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth's radius in meters
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c
        
        # Calculate dynamic radius based on route bounds
        radius = min(max(
            haversine_distance(bounds['south'], bounds['west'], bounds['north'], bounds['east']) / 2,
            2000  # Minimum 2km
        ), 15000)  # Maximum 15km for Berlin area
        
        stations = []
        
        # Multiple search strategies for better coverage
        search_terms = [
            'electric vehicle charging station',
            'EV charging',
            'Tesla Supercharger',
            'charging point',
            'Ladesäule'  # German for charging station
        ]
        
        for keyword in search_terms[:2]:  # Limit to avoid too many API calls
            url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                'location': f"{center_lat},{center_lng}",
                'radius': int(radius),
                'keyword': keyword,
                'key': GOOGLE_MAPS_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data['status'] == 'OK':
                for place in data.get('results', []):
                    # Avoid duplicates by checking if station already exists
                    station_exists = any(
                        abs(station['lat'] - place['geometry']['location']['lat']) < 0.001 and
                        abs(station['lng'] - place['geometry']['location']['lng']) < 0.001
                        for station in stations
                    )
                    
                    if not station_exists:
                        stations.append({
                            'name': place['name'],
                            'lat': place['geometry']['location']['lat'],
                            'lng': place['geometry']['location']['lng'],
                            'rating': place.get('rating', 0),
                            'vicinity': place.get('vicinity', ''),
                            'types': place.get('types', []),
                            'place_id': place.get('place_id', '')
                        })
        
        # Sort by rating and limit results
        stations.sort(key=lambda x: x['rating'], reverse=True)
        return stations[:15]  # Return top 15 stations
        
    except Exception as e:
        print(f"Charging stations API error: {e}")
        return []

def decode_polyline(polyline_str):
    """Decode Google Maps polyline to coordinates"""
    index = 0
    lat = 0
    lng = 0
    coordinates = []
    
    while index < len(polyline_str):
        # Decode latitude
        result = 1
        shift = 0
        while True:
            b = ord(polyline_str[index]) - 63 - 1
            index += 1
            result += b << shift
            shift += 5
            if b < 0x1f:
                break
        dlat = (~result >> 1) if (result & 1) != 0 else (result >> 1)
        lat += dlat
        
        # Decode longitude
        result = 1
        shift = 0
        while True:
            b = ord(polyline_str[index]) - 63 - 1
            index += 1
            result += b << shift
            shift += 5
            if b < 0x1f:
                break
        dlng = (~result >> 1) if (result & 1) != 0 else (result >> 1)
        lng += dlng
        
        coordinates.append({'lat': lat / 1e5, 'lng': lng / 1e5})
    
    return coordinates

def get_route_with_alternatives(origin, destination):
    """Get multiple route options from Google Directions API with Berlin-wide support"""
    try:
        # Add "Berlin, Germany" to locations if not already specified
        def ensure_berlin_context(location):
            if 'berlin' not in location.lower() and 'germany' not in location.lower():
                return f"{location}, Berlin, Germany"
            return location
        
        origin_full = ensure_berlin_context(origin)
        destination_full = ensure_berlin_context(destination)
        
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            'origin': origin_full,
            'destination': destination_full,
            'alternatives': True,
            'departure_time': int(datetime.now().timestamp()),
            'traffic_model': 'best_guess',
            'region': 'de',  # Germany region bias
            'language': 'en',
            'key': GOOGLE_MAPS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=20)  # Increased timeout
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'OK':
            routes = []
            for i, route in enumerate(data['routes'][:3]):  # Max 3 routes
                leg = route['legs'][0]
                
                # Decode polyline for detailed analysis
                polyline_points = decode_polyline(route['overview_polyline']['points'])
                
                # Sample points for analysis (to reduce API calls)
                sample_points = polyline_points[::max(1, len(polyline_points)//5)]  # Sample 5 points
                
                # Get real weather data (using route midpoint)
                mid_point = polyline_points[len(polyline_points)//2] if polyline_points else {'lat': 52.5200, 'lng': 13.4050}
                weather_temp = get_weather_data(mid_point['lat'], mid_point['lng'])
                
                # Calculate elevation gradient (using fewer points to reduce API calls)
                elevation_gradient = calculate_elevation_gradient(sample_points)
                
                # Get traffic density
                traffic_density = get_traffic_density([sample_points[0], sample_points[-1]])
                
                # Estimate speed limit based on route type and Berlin context
                distance_km = route.get('legs', [{}])[0].get('distance', {}).get('value', 0) / 1000
                duration_hours = route.get('legs', [{}])[0].get('duration', {}).get('value', 3600) / 3600
                avg_speed = distance_km / duration_hours if duration_hours > 0 else 50
                
                # Berlin-specific speed limit estimation
                if 'autobahn' in route.get('summary', '').lower() or avg_speed > 80:
                    speed_limit = min(max(avg_speed * 1.1, 80), 130)  # Autobahn speeds
                elif 'ring' in route.get('summary', '').lower() or avg_speed > 60:
                    speed_limit = min(max(avg_speed * 1.2, 60), 100)  # Ring roads
                else:
                    speed_limit = min(max(avg_speed * 1.3, 30), 70)   # City streets
                
                route_features = {
                    'speed_limit_kmh': speed_limit,
                    'elevation_gradient': elevation_gradient,
                    'weather_temperature_celsius': weather_temp,
                    'traffic_density': traffic_density
                }
                
                # Calculate energy consumption
                features_df = pd.DataFrame([list(route_features.values())], 
                                         columns=['speed_limit_kmh', 'elevation_gradient', 
                                                'weather_temperature_celsius', 'traffic_density'])
                energy_consumption = model.predict(features_df)[0]
                
                route_info = {
                    'route_index': i,
                    'distance': leg['distance']['text'],
                    'distance_value': leg['distance']['value'],
                    'duration': leg['duration']['text'],
                    'duration_value': leg['duration']['value'],
                    'duration_in_traffic': leg.get('duration_in_traffic', leg['duration'])['text'],
                    'polyline': route['overview_polyline']['points'],
                    'bounds': route['bounds'],
                    'features': route_features,
                    'energy_consumption': float(energy_consumption),
                    'summary': route.get('summary', f'Route {i+1}'),
                    'route_type': 'energy_optimized' if i == 0 else 'alternative',
                    'warnings': route.get('warnings', []),
                    'copyrights': route.get('copyrights', '')
                }
                
                routes.append(route_info)
            
            # Sort routes: first by energy consumption (for energy optimized), then by distance
            routes.sort(key=lambda x: (x['energy_consumption'], x['distance_value']))
            if routes:
                routes[0]['route_type'] = 'energy_optimized'
                if len(routes) > 1:
                    # Find shortest distance route
                    shortest_route = min(routes[1:], key=lambda x: x['distance_value'])
                    shortest_route['route_type'] = 'shortest_distance'
            
            return routes
        else:
            error_msg = data.get('error_message', data['status'])
            print(f"Google Directions API error: {error_msg}")
            if data['status'] == 'ZERO_RESULTS':
                return {'error': f"No routes found between '{origin}' and '{destination}' in Berlin area"}
            
    except Exception as e:
        print(f"Error fetching routes: {e}")
        return {'error': f'Route calculation failed: {str(e)}'}
    
    return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/optimize-routes', methods=['POST'])
def optimize_routes():
    """Get multiple optimized routes with real data"""
    print("=== OPTIMIZE ROUTES ENDPOINT CALLED ===")
    
    try:
        data = request.get_json()
        origin = data.get('origin')
        destination = data.get('destination')

        if not origin or not destination:
            return jsonify({'error': 'Origin and destination are required'}), 400

        print(f"Processing routes from '{origin}' to '{destination}'")
        
        # Get multiple routes with real data
        routes = get_route_with_alternatives(origin, destination)
        
        if not routes:
            return jsonify({'error': 'No routes found'}), 404
        
        # Get charging stations along the route area
        route_bounds = routes[0]['bounds']
        charging_stations = get_charging_stations(route_bounds)
        
        result = {
            'routes': routes,
            'charging_stations': charging_stations,
            'weather_source': 'OpenWeather API',
            'traffic_source': 'Google Maps Traffic',
            'elevation_source': 'Google Elevation API'
        }
        
        print(f"Returning {len(routes)} routes and {len(charging_stations)} charging stations")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in optimize-routes endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

# Keep the old endpoint for backward compatibility
@app.route('/predict', methods=['POST'])
def predict():
    """Legacy endpoint - redirects to new optimize-routes"""
    return optimize_routes()

if __name__ == '__main__':
    print("Starting EV Route Optimizer with enhanced features...")
    print("Features enabled:")
    print("- Real weather data (OpenWeather)")
    print("- Real elevation data (Google Elevation)")
    print("- Real traffic data (Google Traffic)")
    print("- Multiple route alternatives")
    print("- EV charging station locations")
    app.run(host='0.0.0.0', port=5000, debug=True)