import json
import random

def generate_data_with_constraint(num_samples):
    data = []
    charging_stations = set()  ##set create kiye hai for keeping track of generated charging station coordinates(type of list)
    ###################################
    
    for _ in range(num_samples):
        #### for generating the usera co oordinates
        user_coords = generate_user_coords(charging_stations)
        
        new_station_coords = generate_suggested_station(user_coords)
        
        data.append({
            "user_coords": user_coords,
            "new_station_coords": new_station_coords
        })
    return data

def generate_user_coords(charging_stations):
    max_distance = 5
    
    while True:
        user_coords = (random.uniform(-90, 90), random.uniform(-180, 180))
        if all(distance_between_points(user_coords, coords) >= max_distance for coords in charging_stations):
            return user_coords

def generate_suggested_station(user_coords):
    max_distance = 5
    while True:
        suggested_coords = (
            user_coords[0] + random.uniform(-max_distance, max_distance),
            user_coords[1] + random.uniform(-max_distance, max_distance)
        )
        #########################################################################
        ###to ccheck if the suggested station is within the 5-unit distance constraint.
        #########
        if distance_between_points(user_coords, suggested_coords) <= max_distance:
            return suggested_coords

def distance_between_points(coord1, coord2):
    return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5

# generating anyy dummy samples with the 5unit distance constraint
dummy_data = generate_data_with_constraint(5000)

#ye data write karne k liye haii json filee me
with open('training_data_with_constraint.json', 'w') as file:
    json.dump(dummy_data, file)
