import json
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# Coordinates of existing charging stations
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

def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def find_closest_station(user_coords):
    closest_station = None
    closest_distance = float('inf')
    
    for station, coords in charging_stations.items():
        distance = euclidean_distance(user_coords, coords)
        if distance < closest_distance:
            closest_station = station
            closest_distance = distance
    
    return closest_station, closest_distance

def log_request(user_coords, closest_station, closest_distance):
    if closest_distance > 5:
        logged_requests.append({'user_coords': user_coords, 'new_station_coords': charging_stations[closest_station]})
        return True
    return False

def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def prepare_data_for_training(data):
    X = np.array([entry['user_coords'] for entry in data])
    y = np.array([entry['new_station_coords'] for entry in data])
    return X, y

# Load and prepare generated data for training
json_data = load_data_from_json('training_data_with_constraint.json')
with tqdm(total=len(json_data)) as pbar:
    X_train, y_train = [], []
    for entry in json_data:
        X_train.append(entry['user_coords'])
        y_train.append(entry['new_station_coords'])
        # Update the progress bar
        pbar.update(1)

X_train, y_train = prepare_data_for_training(json_data)
ml_model.fit(X_train, y_train)
print(ml_model.n_estimators)
print("Model trained successfully!")

# Program loop
while True:
    user_input = input("Enter user coordinates (latitude, longitude) or 'exit' to stop: ")
    if user_input.lower() == 'exit':
        break
    
    try:
        user_coords = tuple(map(float, user_input.split(',')))
        closest_station, closest_distance = find_closest_station(user_coords)
        
        print(f"Closest charging station to user coordinates: {closest_station}")
        print(f"Distance to {closest_station}: {closest_distance}")
        
        if log_request(user_coords, closest_station, closest_distance):
            X = np.array([req['user_coords'] for req in logged_requests])
            if len(X) > 2:  # Checking if there are enough requests for clustering
                kmeans = KMeans(n_clusters=3)  # Set the number of clusters appropriately
                
                # Create a tqdm progress bar for the clustering process
                with tqdm(desc='Clustering Progress', total=100) as pbar:
                    kmeans.fit(X)
                    pbar.update(100)  # Update progress to indicate completion
                    
                cluster_centers = kmeans.cluster_centers_
                print(f"Request logged for ML.")
                print(f"Suggestions for new charging station locations based on logged requests: {cluster_centers}")
            else:
                print("Not enough logged requests to make suggestions based on clustering.")
        else:
            print("No logged requests to make suggestions.")
            
    except ValueError:
        print("Invalid input. Please enter latitude and longitude separated by a comma.")







