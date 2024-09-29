# main.py

import numpy as np
import threading
import queue
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
from RT_processor import process_RT_data
from LiDAR_processor import receive_bounding_boxes
from detection_adjustment import detection_adjustment
from occupancy_map import update_grid
from plotting import plot
from control import pure_pursuit
#from CAN_msg_send import steering

# Initialize the grid map
grid_size = 0.25    # meters
map_width = 100.0   # meters
map_height = 100.0  # meters
submatrix_size = 70  # meters

# Number of grid cells
num_cols = int(map_width / grid_size)
num_rows = int(map_height / grid_size)

# Initialize the occupancy grid map with 9 boolean values per cell
occupancy_map = np.zeros((num_rows, num_cols, 9), dtype=bool)

# Set the 9th boolean value (ground) to True in every cell
occupancy_map[:, :, 8] = True

steering_values = []

# Global variable to control the steering thread
steering_thread_active = False
steering_data = {
    'vehicle_pos': None,
    'smoothed_path': None,
    'map_heading': None,
    'wheelbase': 2.789,  # [m]
    'lookahead_distance': 24,  # [grid cell]
    'steering_angle_tolerance': 360  # [deg]
}
steering_data_lock = threading.Lock()  # Thread lock to safely update steering data

# Initialize threads for simultaneous RT and LiDAR data processing
def RT_processor_thread(ip, port, target_xy, RT_queue):
    for data in process_RT_data(ip, port, target_xy):
        RT_queue.put(data)

def LiDAR_processor_thread(grid_queue):
    receive_bounding_boxes(grid_queue)

def steering_thread():
    global steering_values
    while steering_thread_active:
        with steering_data_lock:
            vehicle_pos = steering_data['vehicle_pos']
            smoothed_path = steering_data['smoothed_path']
            map_heading = steering_data['map_heading']
            wheelbase = steering_data['wheelbase']
            lookahead_distance = steering_data['lookahead_distance']
            steering_angle_tolerance = steering_data['steering_angle_tolerance']
        
        # If data is available, calculate steering angle
        if vehicle_pos is not None and smoothed_path is not None and map_heading is not None:
            # Calculate the steering angle
            steering_angle = pure_pursuit(lookahead_distance, wheelbase, vehicle_pos, smoothed_path, map_heading)
            # Call the steering function
            #steering_values = steering(steering_angle, steering_angle_tolerance)

        time.sleep(0.1)

def main():
    global steering_thread_active

    # Customizable variables
    plot_enabled = True
    continuous_plot = True
    print_data = True
    path_planning = True    
    control = True
    target_xy = [0, 50]  # [m] [X,Y] target position relative to the vehicle
    
    # RT connection data
    ip_address = '0.0.0.0'
    port = 3000
    
    # File name with current timestamp
    recording_filename = f"recording-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

    # Open the file in append mode
    with open(recording_filename, 'a') as file:
        # Write the header row
        file.write("GPS time, Distance to target, Current steering angle, Calculated steering angle, Transmitted steering angle, Velocity\n")

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

        timestamp = "N/A"

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

            # Start the steering thread if control is enabled
            if control and not steering_thread_active:
                steering_thread_active = True
                steering_thread_instance = threading.Thread(target=steering_thread)
                steering_thread_instance.daemon = True
                steering_thread_instance.start()

            while True:
                # Process data from the RT queue
                if not RT_queue.empty():
                    while not RT_queue.empty():
                        RT_data = RT_queue.get()
                    if RT_data:
                        lat_lon, map_heading, timestamp, target_distance, veh_xy, normal_velocity, velocity_x, velocity_y = RT_data
                        RT_data_flow = True
                else:
                    RT_data_flow = False
                    RT_data = None

                # Process data from the LiDAR queue
                if not LiDAR_queue.empty():
                    while not LiDAR_queue.empty():
                        LiDAR_data = LiDAR_queue.get()
                        LiDAR_data_flow = True
                else:
                    LiDAR_data_flow = False
                
                if RT_data_flow or LiDAR_data_flow:
                    # Recalculating the coordinates to the gridmap's coordinate system
                    polygons, vehicle_polygon = detection_adjustment(LiDAR_data_flow, RT_data_flow, LiDAR_data, veh_xy, map_heading)
                    
                    # Updating the occupancy grid map with the detections, position
                    occupancy_map, submatrix, vehicle_pos, smoothed_path, target = update_grid(
                        LiDAR_data_flow, RT_data_flow, path_planning, occupancy_map, polygons,
                        grid_size, num_rows, num_cols, submatrix_size, veh_xy, vehicle_polygon, target_xy, map_heading
                    )

                    if control:
                        # Update the data used by the steering thread
                        with steering_data_lock:
                            steering_data['vehicle_pos'] = vehicle_pos
                            steering_data['smoothed_path'] = smoothed_path
                            steering_data['map_heading'] = map_heading
                    
                    # Prepare data for logging
                    current_steering_angle = steering_values[0] if len(steering_values) >= 1 else "N/A"
                    calculated_steering_angle = steering_values[1] if len(steering_values) >= 2 else "N/A"
                    transmitted_steering_angle = steering_values[2] if len(steering_values) >= 3 else "N/A"
                    distance_to_target = f"{target_distance:.2f}" if RT_data_flow else "N/A"
                    
                    # Log data to the file
                    file.write(f"{timestamp}, {distance_to_target}, {current_steering_angle}, {calculated_steering_angle}, {transmitted_steering_angle}, {normal_velocity}\n")

                    # Updating the plot
                    if plot_enabled and continuous_plot:
                        plot(occupancy_map, submatrix, veh_xy, grid_size, map_width, map_height, RT_data_flow, mode='submatrix')
                else:
                    time.sleep(0.1)

                # Printing out the given values    
                if print_data:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    if RT_data_flow and LiDAR_data_flow:
                        print(f"GPS time: {timestamp}")
                        print(f"Plot:\t\t{plot_enabled}\tContinuous Plot:\t{continuous_plot}\tPrint Data:\t{print_data}")
                        print(f"Traj Planning:\t{path_planning}\tControl:\t\t{control}")
                        print(f"Normal velocity:\t{normal_velocity:.3f}\tm/s\tY velocity:\t{velocity_x:.3f}\tm/s\tX velocity:\t{velocity_y:.3f}\tm/s")
                        print(f"Distance to target point: {target_distance:.2f} meters")
                        print(f"RT:     OK  ")
                        print(f"LiDAR:  OK  ")
                    elif not RT_data_flow and LiDAR_data_flow:
                        print(f"GPS time: N/A")
                        print(f"Plot:\t\t{plot_enabled}\tContinuous Plot:\t{continuous_plot}\tPrint Data:\t{print_data}")
                        print(f"Traj Planning:\t{path_planning}\tControl:\t\t{control}")
                        print(f"Normal velocity:\tN/A\tm/s\tY velocity:\tN/A\tm/s\tX velocity:\tN/A\tm/s")
                        print(f"Distance to target point: N/A meters    ")
                        print(f"RT:    NOK  ")
                        print(f"LiDAR:  OK  ")
                    elif RT_data_flow and not LiDAR_data_flow:
                        print(f"GPS time: {timestamp}")
                        print(f"Plot:\t\t{plot_enabled}\tContinuous Plot:\t{continuous_plot}\tPrint Data:\t{print_data}")
                        print(f"Traj Planning:\t{path_planning}\tControl:\t\t{control}")
                        print(f"Normal velocity:\t{normal_velocity:.3f}\tm/s\tY velocity:\t{velocity_x:.3f}\tm/s\tX velocity:\t{velocity_y:.3f}\tm/s")
                        print(f"Distance to target point: {target_distance:.2f} meters")
                        print(f"RT:     OK  ")
                        print(f"LiDAR: NOK  ")
                    elif not RT_data_flow and not LiDAR_data_flow:
                        print(f"GPS time: N/A")
                        print(f"Plot:\t\t{plot_enabled}\tContinuous Plot:\t{continuous_plot}\tPrint Data:\t{print_data}")
                        print(f"Traj Planning:\t{path_planning}\tControl:\t\t{control}")
                        print(f"Normal velocity:\tN/A\tm/s\tY velocity:\tN/A\tm/s\tX velocity:\tN/A\tm/s")
                        print(f"Distance to target point: N/A meters    ")
                        print(f"RT:    NOK  ")
                        print(f"LiDAR: NOK  ")
                    
                    if RT_data_flow and control and len(steering_values) == 3:
                        print(f"Current steering angle:\t\t{steering_values[0]:.5f} \tdeg")
                        print(f"Calculated steering angle:\t{steering_values[1]:.5f} \tdeg")
                        print(f"Transmitted steering angle:\t{steering_values[2]:.5f} \tdeg")
                    else:
                        print(f"Current steering angle:\t\tN/A\tdeg")
                        print(f"Calculated steering angle:\tN/A\tdeg")
                        print(f"Transmitted steering angle:\tN/A\tdeg")

        except KeyboardInterrupt:
            # Stop the steering thread
            steering_thread_active = False
            print("\nProgram exited by user.")
            if plot_enabled:
                plt.ioff()
                plot(occupancy_map, submatrix, veh_xy, grid_size, map_width, map_height, RT_data_flow, mode='full')
                plt.show()

if __name__ == "__main__":
    main()
