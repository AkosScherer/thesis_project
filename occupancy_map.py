import numpy as np
import cv2

first_iteration = True

def update_grid(LiDAR_data_flow, RT_data_flow, occupancy_map, polygons, grid_size, num_rows, num_cols, veh_xy, vehicle_polygon, target_xy):
    global first_iteration

    if RT_data_flow:
        # Convert vehicle trajectory position to grid coordinates
        trajectory_col = int(veh_xy[0] / grid_size + num_cols / 2)
        trajectory_row = int(veh_xy[1] / grid_size)
    else:
        trajectory_col = int(num_cols / 2)
        trajectory_row = int(0)
    
    if first_iteration and RT_data_flow:
        # Convert target coordinates from meters to grid cells within the submatrix
        target_col = int(target_xy[0] / grid_size + num_cols / 2)
        target_row = int(target_xy[1] / grid_size)
        
        # Create a circular mask with a 25-grid-cell radius
        radius = 25  # 25 grid cells
        y_grid, x_grid = np.ogrid[:occupancy_map.shape[0], :occupancy_map.shape[1]]
        distance_from_center = np.sqrt((x_grid - target_col) ** 2 + (y_grid - target_row) ** 2)
        target_mask = distance_from_center <= radius
        
        # Set the values within the circular mask to 0, but only overwrite cells that are not 1 or 4
        occupancy_map[target_mask == 1] = 0
        first_iteration = False
        
    # Define the submatrix boundaries
    submatrix_size = 100 # [m]
    submatrix_col_start = int(max(0, trajectory_col - (submatrix_size / grid_size / 2)))
    submatrix_col_end = int(min(num_cols, submatrix_col_start + (submatrix_size / grid_size)))
    submatrix_row_start = int(max(0, trajectory_row + 1 - (submatrix_size / grid_size * 0.20)))
    submatrix_row_end = int(min(num_rows, trajectory_row + 1 + (submatrix_size / grid_size * 0.80)))

    # Extract the submatrix from the occupancy_map
    submatrix = np.copy(occupancy_map[submatrix_row_start:submatrix_row_end, submatrix_col_start:submatrix_col_end])
    
    if LiDAR_data_flow:
        for polygon in polygons:
            # Adjust polygon vertices to submatrix coordinates
            enlarged_polygon_cells = np.array([
                [int((polygon[0] - 2) / grid_size + num_cols / 2) - submatrix_col_start, int((polygon[1] - 2) / grid_size) - submatrix_row_start],
                [int((polygon[2] + 2) / grid_size + num_cols / 2) - submatrix_col_start, int((polygon[3] - 2) / grid_size) - submatrix_row_start],
                [int((polygon[4] + 2) / grid_size + num_cols / 2) - submatrix_col_start, int((polygon[5] + 2) / grid_size) - submatrix_row_start],
                [int((polygon[6] - 2) / grid_size + num_cols / 2) - submatrix_col_start, int((polygon[7] + 2) / grid_size) - submatrix_row_start]
            ], dtype=np.int32)
            enlarged_polygon_cells = enlarged_polygon_cells.reshape((-1, 1, 2))

            # Fill the enlarged polygon in the submatrix directly
            enlarged_mask = np.zeros(submatrix.shape, dtype=np.uint8)
            cv2.fillPoly(enlarged_mask, [enlarged_polygon_cells], 1)

            # Update the submatrix with the new values
            submatrix = np.where((enlarged_mask == 1) & (submatrix != 100) & (submatrix != 75) & (submatrix != 0), 50, submatrix)
            
            enlarged_polygon_cells = np.array([
                [int((polygon[0] - 1) / grid_size + num_cols / 2) - submatrix_col_start, int((polygon[1] - 1) / grid_size) - submatrix_row_start],
                [int((polygon[2] + 1) / grid_size + num_cols / 2) - submatrix_col_start, int((polygon[3] - 1) / grid_size) - submatrix_row_start],
                [int((polygon[4] + 1) / grid_size + num_cols / 2) - submatrix_col_start, int((polygon[5] + 1) / grid_size) - submatrix_row_start],
                [int((polygon[6] - 1) / grid_size + num_cols / 2) - submatrix_col_start, int((polygon[7] + 1) / grid_size) - submatrix_row_start]
            ], dtype=np.int32)
            enlarged_polygon_cells = enlarged_polygon_cells.reshape((-1, 1, 2))

            # Fill the enlarged polygon in the submatrix directly
            enlarged_mask = np.zeros(submatrix.shape, dtype=np.uint8)
            cv2.fillPoly(enlarged_mask, [enlarged_polygon_cells], 1)

            # Update the submatrix with the new values
            submatrix = np.where((enlarged_mask == 1) & (submatrix != 100) & (submatrix != 0), 75, submatrix)
            
            # Adjust the polygon vertices to submatrix coordinates
            polygon_cells = np.array([
                [int(polygon[0] / grid_size + num_cols / 2) - submatrix_col_start, int(polygon[1] / grid_size) - submatrix_row_start],
                [int(polygon[2] / grid_size + num_cols / 2) - submatrix_col_start, int(polygon[3] / grid_size) - submatrix_row_start],
                [int(polygon[4] / grid_size + num_cols / 2) - submatrix_col_start, int(polygon[5] / grid_size) - submatrix_row_start],
                [int(polygon[6] / grid_size + num_cols / 2) - submatrix_col_start, int(polygon[7] / grid_size) - submatrix_row_start]
            ], dtype=np.int32)
            polygon_cells = polygon_cells.reshape((-1, 1, 2))
            
            # Fill polygon in the submatrix
            mask = np.zeros_like(submatrix, dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_cells], 1)
            
            # Update the submatrix regardless of the actual value of the cells
            submatrix[mask == 1] = 100

    if RT_data_flow:      
        submatrix[submatrix == -2] = 1
        
        # Adjust vehicle polygon coordinates for submatrix indexing
        vehicle_mask_coords = [
            [int(vehicle_polygon[0] / grid_size + num_cols / 2) - submatrix_col_start, int(vehicle_polygon[1] / grid_size) - submatrix_row_start],
            [int(vehicle_polygon[2] / grid_size + num_cols / 2) - submatrix_col_start, int(vehicle_polygon[3] / grid_size) - submatrix_row_start],
            [int(vehicle_polygon[4] / grid_size + num_cols / 2) - submatrix_col_start, int(vehicle_polygon[5] / grid_size) - submatrix_row_start],
            [int(vehicle_polygon[6] / grid_size + num_cols / 2) - submatrix_col_start, int(vehicle_polygon[7] / grid_size) - submatrix_row_start]
        ]
        
        # Apply the vehicle mask to the submatrix
        vehicle_mask_filled = np.zeros_like(submatrix, dtype=np.uint8)
        cv2.fillPoly(vehicle_mask_filled, [np.array(vehicle_mask_coords, dtype=np.int32)], 1)
        submatrix[(vehicle_mask_filled == 1) & (submatrix != -1)] = -2
        
        # Adjust trajectory position to submatrix coordinates
        submatrix_trajectory_col = trajectory_col - submatrix_col_start
        submatrix_trajectory_row = trajectory_row - submatrix_row_start
        
        # Update trajectory position in the submatrix
        submatrix[submatrix_trajectory_row, submatrix_trajectory_col] = -1

    # Put the updated submatrix back into the original occupancy_map
    occupancy_map[submatrix_row_start:submatrix_row_end, submatrix_col_start:submatrix_col_end] = submatrix
    
    return occupancy_map, submatrix
