# occupancy_map.py

import heapq
import numpy as np
import cv2
from collections import deque
from scipy.interpolate import splprep, splev

first_iteration = True
cycle_counter = 0
vehicle_mask_filled = []
local_planning = False
first_path = True
prev_object_cell_count = 0
base_path = []

# Manhattan distance heuristic for A*
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(occupancy_map, start, goal):#, local_planning):
    rows, cols, cell_values = occupancy_map.shape
    open_set = []
    
    # Convert start and goal to tuples
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
            (current[0] + 1, current[1] + 1),   # down-right
            (current[0] - 1, current[1] - 1),   # up-left
            (current[0] + 1, current[1] - 1),   # down-left
            (current[0] - 1, current[1] + 1)    # up-right
        ]
        
        #if not local_planning:
        #    for neighbor in neighbors:
        #        if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
        #            move_cost = np.sqrt(2) if (neighbor[0] != current[0] and neighbor[1] != current[1]) else 1
        #            tentative_g_score = g_score[current] + move_cost

        #            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
        #                g_score[neighbor] = tentative_g_score
        #                f_score = tentative_g_score + heuristic(neighbor, goal)
        #                heapq.heappush(open_set, (f_score, neighbor))
        #                came_from[neighbor] = current
        #else:
        for neighbor in neighbors:
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Handle cells with specific flags
                if not (occupancy_map[neighbor[0], neighbor[1], 3] or 
                        occupancy_map[neighbor[0], neighbor[1], 4]):
                    
                    # Base move cost (diagonal move is sqrt(2), otherwise 1)
                    move_cost = np.sqrt(2) if (neighbor[0] != current[0] and neighbor[1] != current[1]) else 1
                    
                    # Check if the neighbor is in a high-cost area (channel 7)
                    if occupancy_map[neighbor[0], neighbor[1], 7]:
                        move_cost *= 100 

                    # Check if the neighbor is in a very high-cost area (channel 5)
                    if occupancy_map[neighbor[0], neighbor[1], 5]:
                        move_cost *= 1000 
                    
                    tentative_g_score = g_score[current] + move_cost

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))
                        came_from[neighbor] = current
    return []

def smooth_path(path, window_size=20):
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
    global cycle_counter
    global vehicle_mask_filled
    global local_planning
    global first_path
    global prev_object_cell_count
    global base_path
    
    cycle_counter += 1

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
        
        occupancy_map[:,:,6][target_mask] = True 
        first_iteration = False
    
    if cycle_counter == 30:
        LiDAR_data_flow = True
        polygons = [[-6, 12, -5, 12, -5, 13, -6, 13]]
        
    if cycle_counter == 40:
      LiDAR_data_flow = True
      polygons = [[-3.5, 12, -2.5, 12, -2.5, 13, -3.5, 13], [-0.5, 30, 0.5, 30, 0.5, 31, -0.5, 31]]
    
    if LiDAR_data_flow:
        for polygon in polygons:
            
            object_mask = np.zeros((occupancy_map.shape[0], occupancy_map.shape[1]), dtype=np.uint8)
            enlarged_polygon_cells = np.array([
                [(polygon[0] - 4) // grid_size + num_cols // 2, (polygon[1] - 4) // grid_size],
                [(polygon[2] + 4) // grid_size + num_cols // 2, (polygon[3] - 4) // grid_size],
                [(polygon[4] + 4) // grid_size + num_cols // 2, (polygon[5] + 4) // grid_size],
                [(polygon[6] - 4) // grid_size + num_cols // 2, (polygon[7] + 4) // grid_size]
            ], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(object_mask, [enlarged_polygon_cells], 1)
            occupancy_map[:,:,7][object_mask == 1] = True 
            
            object_mask = np.zeros((occupancy_map.shape[0], occupancy_map.shape[1]), dtype=np.uint8)
            enlarged_polygon_cells = np.array([
                [(polygon[0] - 2) // grid_size + num_cols // 2, (polygon[1] - 2) // grid_size],
                [(polygon[2] + 2) // grid_size + num_cols // 2, (polygon[3] - 2) // grid_size],
                [(polygon[4] + 2) // grid_size + num_cols // 2, (polygon[5] + 2) // grid_size],
                [(polygon[6] - 2) // grid_size + num_cols // 2, (polygon[7] + 2) // grid_size]
            ], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(object_mask, [enlarged_polygon_cells], 1)
            occupancy_map[:,:,5][object_mask == 1] = True 
            
            object_mask = np.zeros((occupancy_map.shape[0], occupancy_map.shape[1]), dtype=np.uint8)
            enlarged_polygon_cells = np.array([
                [(polygon[0] - 1) // grid_size + num_cols // 2, (polygon[1] - 1) // grid_size],
                [(polygon[2] + 1) // grid_size + num_cols // 2, (polygon[3] - 1) // grid_size],
                [(polygon[4] + 1) // grid_size + num_cols // 2, (polygon[5] + 1) // grid_size],
                [(polygon[6] - 1) // grid_size + num_cols // 2, (polygon[7] + 1) // grid_size]
            ], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(object_mask, [enlarged_polygon_cells], 1)
            occupancy_map[:,:,4][object_mask == 1] = True 
            
            object_mask = np.zeros((occupancy_map.shape[0], occupancy_map.shape[1]), dtype=np.uint8)
            polygon_cells = np.array([
                [polygon[0] // grid_size + num_cols // 2, polygon[1] // grid_size],
                [polygon[2] // grid_size + num_cols // 2, polygon[3] // grid_size],
                [polygon[4] // grid_size + num_cols // 2, polygon[5] // grid_size],
                [polygon[6] // grid_size + num_cols // 2, polygon[7] // grid_size]
            ], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(object_mask, [polygon_cells], 1)
            occupancy_map[:,:,3][object_mask == 1] = True 
            
        del polygon

    if RT_data_flow:      
        occupancy_map[:,:,2] = False 
        
        # Initialize vehicle_mask_filled as a single-channel matrix
        vehicle_mask_filled = np.zeros((occupancy_map.shape[0], occupancy_map.shape[1]), dtype=np.uint8)

        # Ensure vehicle_mask_coords has the correct shape
        vehicle_mask_coords = np.array([
            [vehicle_polygon[0] // grid_size + num_cols // 2, vehicle_polygon[1] // grid_size],
            [vehicle_polygon[2] // grid_size + num_cols // 2, vehicle_polygon[3] // grid_size],
            [vehicle_polygon[4] // grid_size + num_cols // 2, vehicle_polygon[5] // grid_size],
            [vehicle_polygon[6] // grid_size + num_cols // 2, vehicle_polygon[7] // grid_size]
        ], dtype=np.int32).reshape((-1, 1, 2))

        # Use cv2.fillPoly to fill the vehicle mask
        cv2.fillPoly(vehicle_mask_filled, [vehicle_mask_coords], 1)

        # Overwrite the area in the occupancy map with -2, but keep -1 unchanged
        # Update the third channel of occupancy map based on vehicle_mask_filled
        occupancy_map[:,:,2][vehicle_mask_filled == 1] = True  

        # Update the trajectory position
        occupancy_map[vehicle_pos[0], vehicle_pos[1], 0] = True 


    if path_planning:
        occupancy_map[:,:,1][occupancy_map[:,:,1]] = False 
        
        if first_path:
            target_row = int(target_xy[1] / grid_size)
            target_col = int(target_xy[0] / grid_size + num_cols / 2)
            target_rc = (target_row, target_col)
            
            base_path = a_star_search(occupancy_map, tuple(vehicle_pos), tuple(target_rc))
            
            first_path = False
        
        object_cell_count = np.sum(occupancy_map[:, :, 3])
        
        if object_cell_count > prev_object_cell_count + 5:
            #objects = []
            object_indexes = []
            
            # Iterate over the original path
            for index, (row, col) in enumerate(base_path):
                # Check if any of the relevant boolean layers (3, 4, 5, or 7) are True
                if occupancy_map[row, col, 3] or occupancy_map[row, col, 4] or occupancy_map[row, col, 5] or occupancy_map[row, col, 7]:
                    object_indexes.append(index)
                    # Check if the next point is not part of an object
                    if not occupancy_map[base_path[index+1][0], base_path[index+1][1], 3] and \
                    not occupancy_map[base_path[index+1][0], base_path[index+1][1], 4] and \
                    not occupancy_map[base_path[index+1][0], base_path[index+1][1], 5] and \
                    not occupancy_map[base_path[index+1][0], base_path[index+1][1], 7]:
                        
                        #print("objektum")
                        
                        last_valid_point = base_path[object_indexes[0]-1]
                        first_valid_point = base_path[object_indexes[-1]+1]
                        #objects.append(object_indexes)
                        
                        local_replanned_path = a_star_search(occupancy_map, last_valid_point, first_valid_point)
                        #local_replanned_path = smooth_path(local_replanned_path)
                        
                        del base_path[(object_indexes[0]-1):(object_indexes[-1]+1)]
                        base_path = base_path[:object_indexes[0]] + local_replanned_path + base_path[object_indexes[0]:]
                        
                        object_indexes = []  # Reset for the next object
                        
            #smoothed_path = smooth_path(base_path)
            #smoothed_path = base_path
                        
                        # ide kell betenni az objektum körüli újratervezést
                        # egy szegmens újratervezése után ki kell szedni indexek alapján base_path-ból azt a részt ami ütközik az objektummal és a helyére
                        # be kell tenni az újratervezett szekciót és azt tenni a base-path-á
                        # ez után újra futtatni kell a base_path elejéől az objektum ütközés keresést és annyiszor loopolni amíg már nem talál új ütközést
                        # ez után smooth-olni kell a path-t és azt kell megjeleníteni és controlnak tovább küldeni, de a következő iterációban nem a smoothed path-t
                        # hanem a base_path-ot kell újra megvizsgálni, ezért a base_path-ot mindig menteni kell, a smoothed_path-ot nem kell menteniobject[0] - 10
            
            #local_replan_data = []
            
            #for object in objects:
            #    last_valid_point = base_path[object[0] - 10]
            #    first_valid_point = base_path[object[-1] + 10]
                
            #    local_replan_data += [object[0] - 9, object[-1] + 9, last_valid_point, first_valid_point]
                
                
                
            #    target_points.append(last_valid_point)
            #    target_points.append(first_valid_point)
        else:
            object_indexes = []
            
            # Iterate over the original path
            for index, (row, col) in enumerate(base_path):
                # Check if any of the relevant boolean layers (3, 4, 5, or 7) are True
                if occupancy_map[row, col, 3] or occupancy_map[row, col, 4] or occupancy_map[row, col, 5] or occupancy_map[row, col, 7]:
                    object_indexes.append(index)
                    # Check if the next point is not part of an object
                    if not occupancy_map[base_path[index+1][0], base_path[index+1][1], 3] and \
                    not occupancy_map[base_path[index+1][0], base_path[index+1][1], 4] and \
                    not occupancy_map[base_path[index+1][0], base_path[index+1][1], 5] and \
                    not occupancy_map[base_path[index+1][0], base_path[index+1][1], 7]:
                        
                        #print("objektum")
                        
                        last_valid_point = base_path[object_indexes[0]-1]
                        first_valid_point = base_path[object_indexes[-1]+1]
                        #objects.append(object_indexes)
                        
                        local_replanned_path = a_star_search(occupancy_map, last_valid_point, first_valid_point)
                        #local_replanned_path = smooth_path(local_replanned_path)
                        
                        del base_path[(object_indexes[0]-1):(object_indexes[-1]+1)]
                        base_path = base_path[:object_indexes[0]] + local_replanned_path + base_path[object_indexes[0]:]
                        
                        object_indexes = []  # Reset for the next object
        #    smoothed_path = base_path
        
        smoothed_path = smooth_path(base_path)
        
        for (r, c) in smoothed_path:
            occupancy_map[r, c, 1] = True 
            
        prev_object_cell_count = object_cell_count
        
        #target_points = []
        #replanned_path = []

        #heading_rad = np.deg2rad(map_heading)
        #front_distance = 3
        #front_x = veh_xy[0] + front_distance * np.sin(heading_rad)
        #front_y = veh_xy[1] + front_distance * np.cos(heading_rad)
        
        #trajectory_col = int(front_x / grid_size + num_cols / 2)
        #trajectory_row = int(front_y / grid_size)
    
        #front_pos = [trajectory_row, trajectory_col]
        #start = (front_pos[0], front_pos[1])
        
        #local_planning = False
        #first_route = a_star_search(occupancy_map, tuple(vehicle_pos), tuple(start), local_planning)
        #local_planning = False
        #replanned_path = replanned_path + first_route
        
        #target_row = int(target_xy[1] / grid_size)
        #target_col = int(target_xy[0] / grid_size + num_cols / 2)
        #target_rc = (target_row, target_col)
        
        #local_planning = False
        #original_path = a_star_search(occupancy_map, start, goal, local_planning)
        #local_planning = True
        #target_points.append(front_pos)
        #+
        
        #objects = []
        #object_indexes = []

        # Iterate over the original path
        #for index, (row, col) in enumerate(original_path):
        #    # Check if any of the relevant boolean layers (3, 4, 5, or 7) are True
        #    if occupancy_map[row, col, 3] or occupancy_map[row, col, 4] or occupancy_map[row, col, 5] or occupancy_map[row, col, 7]:
        #        object_indexes.append(index)
        #        # Check if the next point is not part of an object
        #        if not occupancy_map[original_path[index+1][0], original_path[index+1][1], 3] and \
        #        not occupancy_map[original_path[index+1][0], original_path[index+1][1], 4] and \
        #        not occupancy_map[original_path[index+1][0], original_path[index+1][1], 5] and \
        #        not occupancy_map[original_path[index+1][0], original_path[index+1][1], 7]:
        #            objects.append(object_indexes)
        #            object_indexes = []  # Reset for the next object

        #for object in objects:
        #    last_valid_index = original_path[object[0] - 10]
        #    first_valid_index = original_path[object[-1] + 10]
        #    
        #    last_valid_point = (last_valid_index[0], last_valid_index[1])
        #    first_valid_point = (first_valid_index[0], first_valid_index[1])
        #    
        #    target_points.append(last_valid_point)
        #    target_points.append(first_valid_point)
            
        #target_points.append(goal)
        
        #for index, _ in enumerate(target_points[:-1]):  # Avoid index error on the last point
        #    start = tuple([target_points[index][0], target_points[index][1]])
        #    finish = tuple([target_points[index+1][0], target_points[index+1][1]])
        #    path_segment = a_star_search(occupancy_map, start, finish, local_planning)
        #    replanned_path = replanned_path + path_segment

        #smoothed_path = smooth_path(replanned_path)

        # Update the occupancymap directly after replanning
        #for (r, c) in smoothed_path:
        #    occupancy_map[r, c, 1] = True 
     
    #elif path_planning and RT_data_flow and first_tarjectory:
    #    heading_rad = np.deg2rad(map_heading)
    #    front_distance = 3
    #    front_x = veh_xy[0] + front_distance * np.sin(heading_rad)
    #    front_y = veh_xy[1] + front_distance * np.cos(heading_rad)
    #    
    #    trajectory_col = int(front_x / grid_size + num_cols / 2)
    #    trajectory_row = int(front_y / grid_size)
    # 
    #     front_pos = [trajectory_row, trajectory_col]
    #     start = (front_pos[0], front_pos[1])
    #     
    #    local_planning = False
    #    first_route = a_star_search(occupancy_map, tuple(vehicle_pos), tuple(start), local_planning)
    #    local_planning = False
    #    replanned_path = replanned_path + first_route
    #    
    #    target_col = int(target_xy[0] / grid_size + num_cols / 2)
    #    target_row = int(target_xy[1] / grid_size)
    #    goal = (target_row, target_col)
    #    
    #    local_planning = False
    #    replanned_path = replanned_path + a_star_search(occupancy_map, start, goal, local_planning)
    #    local_planning = True
    #    #target_points.append(front_pos)
    #    
    #    smoothed_path = smooth_path(replanned_path)

    #    # Update the occupancymap directly after replanning
    #    for (r, c) in smoothed_path:
    #        occupancy_map[r, c, 1] = True
    #    
    #    first_tarjectory = False
        
    else:
        smoothed_path = []    
        
    target_col = int(target_xy[0] / grid_size + num_cols / 2)
    target_row = int(target_xy[1] / grid_size)
    target = [target_row, target_col]
    
    # Define the submatrix boundaries
    submatrix_col_start = int(max(0, vehicle_pos[1] - (submatrix_size / grid_size / 2)))
    submatrix_col_end = int(min(num_cols, submatrix_col_start + (submatrix_size / grid_size)))
    submatrix_row_start = int(max(0, vehicle_pos[0] + 1 - (submatrix_size / grid_size * 0.20)))
    submatrix_row_end = int(min(num_rows, vehicle_pos[0] + 1 + (submatrix_size / grid_size * 0.80)))

    submatrix = np.copy(occupancy_map[submatrix_row_start:submatrix_row_end, submatrix_col_start:submatrix_col_end])
    
    return occupancy_map, submatrix, vehicle_pos, smoothed_path, target