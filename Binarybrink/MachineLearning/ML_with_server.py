from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import numpy as np
import json

app = Flask(__name__)

charging_stations = {
    "Station 1": (25, -70),
    "Station 2": (-40, 60),
    "Station 3": (80, -10),
    "Station 4": (-90, -20),
    "Station 5": (10, 90),
    "Station 6": (-50, -80)
}

logged_requests = []
ml_model = RandomForestRegressor()
kmeans = KMeans(n_clusters=3)

def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def find_closest_station(user_coords):
    # Implementation of find_closest_station function

def log_request(user_coords, closest_station, closest_distance):
    # Implementation of log_request function

def prepare_data_for_training(data):
    X = np.array([entry['user_coords'] for entry in data])
    y = np.array([entry['new_station_coords'] for entry in data])
    return X, y

@app.route('/find_station', methods=['POST'])
def find_station():
    data = request.json
    user_coords = tuple(map(float, data['coordinates'].split(',')))
    closest_station, closest_distance = find_closest_station(user_coords)
    log_request(user_coords, closest_station, closest_distance)
    
    # Perform K-means clustering
    X = np.array([req['user_coords'] for req in logged_requests])
    if len(X) > 2:
        kmeans.fit(X)
        cluster_centers = kmeans.cluster_centers_.tolist()
    else:
        cluster_centers = []

    return jsonify({
        'suggestion': closest_station,
        'k_means_data': cluster_centers
    })

if __name__ == '__main__':
    app.run(debug=True)
