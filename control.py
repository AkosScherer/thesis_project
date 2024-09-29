# control.py - 2024.09.18.

import numpy as np

current_steering_angle = 0
steering_ang_values = [0, 0, 0, 0, 0]
window_size = 5
sum_value = 0

def pure_pursuit(lookahead_distance, wheelbase, current_pos, trajectory, heading_angle):
    global current_steering_angle
    global steering_ang_values
    global window_size
    global sum_value

    # Convert the heading angle to radians
    heading_angle_rad = np.deg2rad(heading_angle)
    
    # Find the lookahead point on the trajectory
    lookahead_point = None
    for point in reversed(trajectory):
        distance = np.linalg.norm(np.array(point) - np.array(current_pos))
        if distance <= lookahead_distance:
            lookahead_point = point
            break

    #if lookahead_point is None:
        #raise ValueError("No point found within the lookahead distance on the trajectory")
    
    if lookahead_point != None:
        # Transform the lookahead point to vehicle coordinates
        dx = lookahead_point[0] - current_pos[0]
        dy = lookahead_point[1] - current_pos[1]

        # Rotate the point to the vehicle's coordinate frame
        local_x = dx * np.cos(heading_angle_rad) + dy * np.sin(heading_angle_rad)
        local_y = -dx * np.sin(heading_angle_rad) + dy * np.cos(heading_angle_rad)

        # Calculate the steering angle
        #if local_x == 0:
        #    return 0.0  # No steering required if the vehicle is directly on the trajectory
        
        curvature = 2 * local_y / (lookahead_distance ** 2)
        steering_angle = np.arctan(curvature * wheelbase)
        steering_angle = np.rad2deg(steering_angle)
    
        steering_angle = int(steering_angle * 40) # adjusting the steering angle to the wheel angle
    
        steering_ang_values.pop(0)
        steering_ang_values.append(steering_angle)
        for ang_value in steering_ang_values:
            sum_value = sum_value + ang_value
        steering_angle = int(sum_value / window_size)
        sum_value = 0
        
        current_steering_angle = steering_angle
        return steering_angle
    
    else:
        return current_steering_angle
