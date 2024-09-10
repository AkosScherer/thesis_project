# occupancy_map.py

import heapq
import numpy as np
import cv2
from collections import deque
from scipy.interpolate import splprep, splev

first_iteration = True

# Manhattan distance heuristic for A*
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star_search(occupancy_map, start, goal):
    rows, cols = occupancy_map.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    g_score = {start: 0}
    
    def heuristic(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        
        neighbors = [
            (current[0] + 1, current[1]),       # down
            (current[0] - 1, current[1]),       # up
            (current[0], current[1] + 1),       # right
            (current[0], current[1] - 1),       # left
            (current[0] + 1, current[1] + 1),  # down-right
            (current[0] - 1, current[1] - 1),  # up-left
            (current[0] + 1, current[1] - 1),  # down-left
            (current[0] - 1, current[1] + 1)   # up-right
        ]
        
        for neighbor in neighbors:
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if occupancy_map[neighbor] <= 1:  # Avoid cells with values > 1
                    # Diagonal move cost adjustment
                    if (neighbor[0] != current[0] and neighbor[1] != current[1]):
                        move_cost = np.sqrt(2)
                    else:
                        move_cost = 1
                    
                    tentative_g_score = g_score[current] + move_cost

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))
                        came_from[neighbor] = current

    return [] 

def smooth_path(path, window_size=25):
    if len(path) < 3:
        return path
    
    path = np.array(path)
    smoothed_path = []

    for i in range(len(path)):
        window_start = max(0, i - window_size)
        window_end = min(len(path), i + window_size + 1)
        window = path[window_start:window_end]
        
        avg_x = np.mean(window[:, 0])
        avg_y = np.mean(window[:, 1])
        
        smoothed_path.append((int(round(avg_x)), int(round(avg_y))))
    
    return smoothed_path

def update_grid(LiDAR_data_flow, RT_data_flow, path_planning, occupancy_map, polygons, grid_size, num_rows, num_cols, submatrix_size, veh_xy, vehicle_polygon, target_xy, map_heading):
    global first_iteration
    global last_object_cnt

    trajectory_col = int(veh_xy[0] / grid_size + num_cols / 2) if RT_data_flow else int(num_cols / 2)
    trajectory_row = int(veh_xy[1] / grid_size) if RT_data_flow else 0
    
    vehicle_pos = [trajectory_row, trajectory_col]
    
    if first_iteration and RT_data_flow:
        target_col = int(target_xy[0] / grid_size + num_cols / 2)
        target_row = int(target_xy[1] / grid_size)
        
        radius = 10
        y_grid, x_grid = np.ogrid[:occupancy_map.shape[0], :occupancy_map.shape[1]]
        distance_from_center = np.sqrt((x_grid - target_col) ** 2 + (y_grid - target_row) ** 2)
        target_mask = distance_from_center <= radius
        
        occupancy_map[target_mask] = 0
        first_iteration = False

    # Define the submatrix boundaries
    submatrix_col_start = int(max(0, vehicle_pos[1] - (submatrix_size / grid_size / 2)))
    submatrix_col_end = int(min(num_cols, submatrix_col_start + (submatrix_size / grid_size)))
    submatrix_row_start = int(max(0, vehicle_pos[0] + 1 - (submatrix_size / grid_size * 0.20)))
    submatrix_row_end = int(min(num_rows, vehicle_pos[0] + 1 + (submatrix_size / grid_size * 0.80)))

    submatrix = np.copy(occupancy_map[submatrix_row_start:submatrix_row_end, submatrix_col_start:submatrix_col_end])
    
    if LiDAR_data_flow:
        for polygon in polygons:
            enlarged_polygon_cells = np.array([
                [(polygon[0] - 2) // grid_size + num_cols // 2 - submatrix_col_start, (polygon[1] - 2) // grid_size - submatrix_row_start],
                [(polygon[2] + 2) // grid_size + num_cols // 2 - submatrix_col_start, (polygon[3] - 2) // grid_size - submatrix_row_start],
                [(polygon[4] + 2) // grid_size + num_cols // 2 - submatrix_col_start, (polygon[5] + 2) // grid_size - submatrix_row_start],
                [(polygon[6] - 2) // grid_size + num_cols // 2 - submatrix_col_start, (polygon[7] + 2) // grid_size - submatrix_row_start]
            ], dtype=np.int32).reshape((-1, 1, 2))

            enlarged_mask = np.zeros(submatrix.shape, dtype=np.uint8)
            cv2.fillPoly(enlarged_mask, [enlarged_polygon_cells], 1)

            submatrix = np.where((enlarged_mask == 1) & (submatrix != 100) & (submatrix != 75) & (submatrix != 0), 50, submatrix)
            
            enlarged_polygon_cells = np.array([
                [(polygon[0] - 1) // grid_size + num_cols // 2 - submatrix_col_start, (polygon[1] - 1) // grid_size - submatrix_row_start],
                [(polygon[2] + 1) // grid_size + num_cols // 2 - submatrix_col_start, (polygon[3] - 1) // grid_size - submatrix_row_start],
                [(polygon[4] + 1) // grid_size + num_cols // 2 - submatrix_col_start, (polygon[5] + 1) // grid_size - submatrix_row_start],
                [(polygon[6] - 1) // grid_size + num_cols // 2 - submatrix_col_start, (polygon[7] + 1) // grid_size - submatrix_row_start]
            ], dtype=np.int32).reshape((-1, 1, 2))

            enlarged_mask = np.zeros(submatrix.shape, dtype=np.uint8)
            cv2.fillPoly(enlarged_mask, [enlarged_polygon_cells], 1)

            submatrix = np.where((enlarged_mask == 1) & (submatrix != 100) & (submatrix != 0), 75, submatrix)
            
            polygon_cells = np.array([
                [polygon[0] // grid_size + num_cols // 2 - submatrix_col_start, polygon[1] // grid_size - submatrix_row_start],
                [polygon[2] // grid_size + num_cols // 2 - submatrix_col_start, polygon[3] // grid_size - submatrix_row_start],
                [polygon[4] // grid_size + num_cols // 2 - submatrix_col_start, polygon[5] // grid_size - submatrix_row_start],
                [polygon[6] // grid_size + num_cols // 2 - submatrix_col_start, polygon[7] // grid_size - submatrix_row_start]
            ], dtype=np.int32).reshape((-1, 1, 2))
            
            mask = np.zeros_like(submatrix, dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_cells], 1)
            
            submatrix[mask == 1] = 100

    if RT_data_flow:      
        submatrix[submatrix == -2] = 1
        
        vehicle_mask_coords = np.array([
            [vehicle_polygon[0] // grid_size + num_cols // 2 - submatrix_col_start, vehicle_polygon[1] // grid_size - submatrix_row_start],
            [vehicle_polygon[2] // grid_size + num_cols // 2 - submatrix_col_start, vehicle_polygon[3] // grid_size - submatrix_row_start],
            [vehicle_polygon[4] // grid_size + num_cols // 2 - submatrix_col_start, vehicle_polygon[5] // grid_size - submatrix_row_start],
            [vehicle_polygon[6] // grid_size + num_cols // 2 - submatrix_col_start, vehicle_polygon[7] // grid_size - submatrix_row_start]
        ], dtype=np.int32)
        
        vehicle_mask_filled = np.zeros_like(submatrix, dtype=np.uint8)
        cv2.fillPoly(vehicle_mask_filled, [vehicle_mask_coords], 1)
        submatrix[(vehicle_mask_filled == 1) & (submatrix != -1)] = -2
        
        submatrix_trajectory_col = vehicle_pos[1] - submatrix_col_start
        submatrix_trajectory_row = vehicle_pos[0] - submatrix_row_start
        
        submatrix[submatrix_trajectory_row, submatrix_trajectory_col] = -1

    occupancy_map[submatrix_row_start:submatrix_row_end, submatrix_col_start:submatrix_col_end] = submatrix

    object = (occupancy_map > 1)
    object_cnt = np.sum(object)

    if path_planning:
        occupancy_map[occupancy_map == -3] = 1

        target_col = int(target_xy[0] / grid_size + num_cols / 2)
        target_row = int(target_xy[1] / grid_size)

        heading_rad = np.deg2rad(map_heading)

        # Calculate the point in front of the vehicle
        front_distance = 5
        front_x = veh_xy[0] + front_distance * np.sin(heading_rad)
        front_y = veh_xy[1] + front_distance * np.cos(heading_rad)

        # Calculate the grid position for the starting point
        trajectory_col = int(front_x / grid_size + num_cols / 2) if RT_data_flow else int(num_cols / 2)
        trajectory_row = int(front_y / grid_size) if RT_data_flow else 0
    
        front_pos = [trajectory_row, trajectory_col]

        # Run A* search from vehicle's position to the point ahead of the vehicle
        start = (vehicle_pos[0], vehicle_pos[1])
        front_pos = (front_pos[0], front_pos[1])
        path1 = a_star_search(occupancy_map, start, front_pos)

        # Run A* search from the point ahead of the vehicle to the target point
        goal = (target_row, target_col)
        path2 = a_star_search(occupancy_map, front_pos, goal)

        # Combine paths
        combined_path = path1[:-1] + path2  # Exclude the overlap point between the two paths

        # Smooth the combined path
        smoothed_path = smooth_path(combined_path)

        for (r, c) in smoothed_path:
            if occupancy_map[r, c] != 0:
                occupancy_map[r, c] = -3
                
        last_object_cnt = object_cnt
        
    target_col = int(target_xy[0] / grid_size + num_cols / 2)
    target_row = int(target_xy[1] / grid_size)
    target = [target_row, target_col]

    return occupancy_map, submatrix, vehicle_pos, smoothed_path, target
