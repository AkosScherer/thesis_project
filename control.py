import math
import os

def angle_and_distance(vehicle_pos, point_pos, heading_angle_deg, target, wheelbase, lookahead):
    # Unpack positions
    x1, y1 = point_pos[-33]
    x2, y2 = target
    
    # Calculate the angle from the vehicle to the point in degrees
    angle_to_point_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
    
    # Normalize the angle to be in the range [0, 360)
    angle_to_point_deg = (angle_to_point_deg + 360) % 360
    heading_angle_deg = (heading_angle_deg + 360) % 360
    
    # Calculate the difference between the two angles
    angle_diff = abs(angle_to_point_deg - heading_angle_deg)
    
    # Ensure we have the smallest angle (<= 180 degrees)
    smaller_angle = min(angle_diff, 360 - angle_diff)
    
    # Calculate the distance between the vehicle and the point
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 0.25
    
    #os.system('cls' if os.name == 'nt' else 'clear')
    #print(f"The smaller angle between the vehicle heading and the point is: {smaller_angle:.2f} degrees")
    #print(f"The distance between the vehicle and the point is: {distance:.2f} meter")
    
    steering_angle = math.degrees(math.atan((2 * wheelbase * math.sin(smaller_angle)) / distance))
    print('steering angle: ', steering_angle, ' distance: ', distance)
