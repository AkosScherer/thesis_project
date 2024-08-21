# main.py

import numpy as np
import threading
import queue
import os
import matplotlib.pyplot as plt
from RT_processor import process_RT_data
from LiDAR_processor import receive_bounding_boxes
from detection_adjustment import detection_adjustment
from occupancy_map import update_grid
from plotting import plot
from control import angle_and_distance

# Initialize the grid map
grid_size = 0.25  # 10 cm grid size
map_width = 100.0  # meters
map_height = 200.0  # meters

submatrix_size = 70 # meters

# Number of grid cells
num_cols = int(map_width / grid_size)
num_rows = int(map_height / grid_size)

# Initialize the occupancy grid map
occupancy_map = np.ones((num_rows, num_cols))

# Initialize threads for simultaneous RT and LiDAR data processing 
def RT_processor_thread(ip, port, target_xy, RT_queue):
    for data in process_RT_data(ip, port, target_xy):
        RT_queue.put(data)

def LiDAR_processor_thread(grid_queue):
    receive_bounding_boxes(grid_queue)

def main():
    # Customizable variables
    plot_enabled = True
    continuous_plot = True
    print_data = False
    path_planning = True
    target_xy = [0, 30]
    lookahead = 8
    
    # RT connection data
    ip_address = '0.0.0.0'
    port = 3000
    
    # Default variable values
    RT_data_flow = False
    LiDAR_data_flow = False
    veh_xy = [None, None]
    RT_data = None
    LiDAR_data = None
    map_heading = None
    polygons = None
    vehicle_polygon = None
    global occupancy_map
    wheelbase = 2.789 # [meter]

    # Queues for inter-thread communication
    RT_queue = queue.Queue()
    LiDAR_queue = queue.Queue()

    try:
        # Start the RT processor in a separate thread
        RT_thread = threading.Thread(target=RT_processor_thread, args=(ip_address, port, target_xy, RT_queue))
        RT_thread.daemon = True
        RT_thread.start()

        # Start the LiDAR processor in a separate thread
        grid_thread = threading.Thread(target=LiDAR_processor_thread, args=(LiDAR_queue,))
        grid_thread.daemon = True 
        grid_thread.start()

        while True:
            # Process data from the RT queue
            if not RT_queue.empty():
                # Only keeping latest value in the queue
                while not RT_queue.empty():
                    RT_data = RT_queue.get()
                if RT_data:
                    lat_lon, map_heading, timestamp, distance, veh_xy = RT_data
                    RT_data_flow = True
            else:
                RT_data_flow = False
                RT_data = None

            # Process data from the LiDAR queue
            if not LiDAR_queue.empty():
                # Only keeping latest value in the queue
                while not LiDAR_queue.empty():
                    LiDAR_data = LiDAR_queue.get()
                if LiDAR_data:
                    LiDAR_data_flow = True
            else:
                LiDAR_data_flow = False
            
            if LiDAR_data_flow or RT_data_flow:
                # Recalculating the coordinates to the gridmap's coordinate system
                polygons, vehicle_polygon = detection_adjustment(LiDAR_data_flow, RT_data_flow, LiDAR_data, veh_xy, map_heading)
                # Updating the occupancy grid map with the detections, position
                occupancy_map, submatrix, vehicle_pos, smoothed_path, target = update_grid(LiDAR_data_flow, RT_data_flow, path_planning, occupancy_map, polygons, grid_size, num_rows, num_cols, submatrix_size, veh_xy, vehicle_polygon, target_xy, map_heading)
                
                # Updating the plot
                if plot_enabled and continuous_plot:
                    plot(occupancy_map, submatrix, veh_xy, grid_size, map_width, map_height, RT_data_flow, mode='submatrix')
                    
                angle_and_distance(vehicle_pos, smoothed_path, map_heading, target, wheelbase, lookahead)
            
            # Printing out the given values    
            if print_data:
                os.system('cls' if os.name == 'nt' else 'clear')
                if RT_data_flow and LiDAR_data_flow:
                    print(f"Distance to target point: {distance:.2f} meters")
                    print(f"RT:     OK  ")
                    print(f"LiDAR:  OK  ")
                elif not RT_data_flow and LiDAR_data_flow:
                    print(f"Distance to target point: -- meters    ")
                    print(f"RT:    NOK  ")
                    print(f"LiDAR:  OK  ")
                elif RT_data_flow and not LiDAR_data_flow:
                    print(f"Distance to target point: {distance:.2f} meters")
                    print(f"RT:     OK  ")
                    print(f"LiDAR: NOK  ")
                elif not RT_data_flow and not LiDAR_data_flow:
                    print(f"Distance to target point: -- meters    ")
                    print(f"RT:    NOK  ")
                    print(f"LiDAR: NOK  ")

    except KeyboardInterrupt:
        # Making plot to remain open after exiting the code
        print("\nProgram exited by user.")
        if plot_enabled:
            # Keep the submatrix plot open
            plt.ioff()
            # Create a new figure for the full map
            plot(occupancy_map, submatrix, veh_xy, grid_size, map_width, map_height, RT_data_flow, mode='full')
            plt.show()
        
if __name__ == "__main__":
    main()