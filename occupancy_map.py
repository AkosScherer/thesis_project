import numpy as np
import cv2

# Initialize a global variable to store the previous vehicle mask region
previous_vehicle_region = None
previous_vehicle_mask = None

first_iteration = True

def update_grid(LiDAR_data_flow, RT_data_flow, occupancy_map, polygons, grid_size, num_rows, num_cols, veh_xy, vehicle_polygon, target_xy):
    global previous_vehicle_region
    global previous_vehicle_mask
    global first_iteration

    # Copy the occupancy_map to avoid modifying the original
    updated_map = np.copy(occupancy_map)
    
    if first_iteration and RT_data_flow:
        # Convert target coordinates from meters to grid cells
        target_col = int(target_xy[0] / grid_size + num_cols / 2)
        target_row = int(target_xy[1] / grid_size)
        
        # Target Point
        # Create a circular mask with a 25-grid-cell radius
        radius = 25  # 25 grid cells
        y_grid, x_grid = np.ogrid[:num_rows, :num_cols]
        distance_from_center = np.sqrt((x_grid - target_col) ** 2 + (y_grid - target_row) ** 2)
        target_mask = distance_from_center <= radius
        
        # Set the second element within the circular mask to "target"
        for i in range(num_rows):
            for j in range(num_cols):
                if target_mask[i, j]:
                    updated_map[i, j][1] = "target"
        
        first_iteration = False

    if RT_data_flow:
        # Convert vehicle trajectory position to grid coordinates
        trajectory_col = int(veh_xy[0] / grid_size + num_cols / 2)
        trajectory_row = int(veh_xy[1] / grid_size)
        
        # Update trajectory position, keeping the first value unchanged
        updated_map[trajectory_row, trajectory_col][1] = "trajectory"
    
    if LiDAR_data_flow:
        for polygon in polygons:
            # Convert polygon vertices from meters to grid cells
            polygon_cells = np.array([[int(polygon[0] / grid_size + num_cols / 2), int(polygon[1] / grid_size)],
                                    [int(polygon[2] / grid_size + num_cols / 2), int(polygon[3] / grid_size)],
                                    [int(polygon[4] / grid_size + num_cols / 2), int(polygon[5] / grid_size)],
                                    [int(polygon[6] / grid_size + num_cols / 2), int(polygon[7] / grid_size)]], dtype=np.int32)
            polygon_cells = polygon_cells.reshape((-1, 1, 2))
            # Fill polygon in the map, keeping the first value unchanged
            mask = np.zeros((num_rows, num_cols), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_cells], 1)
            for i in range(num_rows):
                for j in range(num_cols):
                    updated_map[i, j][0] = 100

    if RT_data_flow:
        # Convert vehicle_polygon vertices from meters to grid cells
        vehicle_mask = np.array([[int(vehicle_polygon[0] / grid_size + num_cols / 2), int(vehicle_polygon[1] / grid_size)],
                                [int(vehicle_polygon[2] / grid_size + num_cols / 2), int(vehicle_polygon[3] / grid_size)],
                                [int(vehicle_polygon[4] / grid_size + num_cols / 2), int(vehicle_polygon[5] / grid_size)],
                                [int(vehicle_polygon[6] / grid_size + num_cols / 2), int(vehicle_polygon[7] / grid_size)]], dtype=np.int32)
        vehicle_mask = vehicle_mask.reshape((-1, 1, 2))
        
        # Create a mask for the vehicle region
        vehicle_mask_filled = np.zeros((num_rows, num_cols), dtype=np.uint8)
        cv2.fillPoly(vehicle_mask_filled, [vehicle_mask], 1)
        
        if previous_vehicle_region is not None:
            # Reset the previous vehicle region in the map, but only overwrite cells that are not 1 or 4
            for i in range(num_rows):
                for j in range(num_cols):
                    updated_map[i, j][1] = previous_vehicle_region[i, j][1]

        # Save the current vehicle region before overwriting
        previous_vehicle_mask = vehicle_mask_filled == 1
        previous_vehicle_region = np.copy(updated_map)
        
        # Apply the vehicle mask to the map, keeping the first value unchanged
        for i in range(num_rows):
            for j in range(num_cols):
                updated_map[i, j][1] = "vehicle"
    
    return updated_map
