# detection_adjustment.py

import math

def rotate_and_translate(x, y, angle_deg, trans_x, trans_y):
    angle_rad = math.radians(angle_deg)
    y_rot = y * math.cos(angle_rad) - x * math.sin(angle_rad)
    x_rot = y * math.sin(angle_rad) + x * math.cos(angle_rad)
    x_final = x_rot + trans_x
    y_final = y_rot + trans_y
    return x_final, y_final

def detection_adjustment(LiDAR_data_flow, RT_data_flow, detections, veh_xy, map_heading):
    polygons = []

    if RT_data_flow:
            veh_x = veh_xy[0]
            veh_y = veh_xy[1]
    else:
        veh_x = 0
        veh_y = 0
        map_heading = 0

    if LiDAR_data_flow:
        for detection in detections:
            x1, y1 = rotate_and_translate(detection[0], detection[4], map_heading, veh_x, veh_y)
            x2, y2 = rotate_and_translate(detection[1], detection[5], map_heading, veh_x, veh_y)
            x3, y3 = rotate_and_translate(detection[2], detection[6], map_heading, veh_x, veh_y)
            x4, y4 = rotate_and_translate(detection[3], detection[7], map_heading, veh_x, veh_y)
            
            polygon = [x1, y1, x2, y2, x3, y3, x4, y4]
            polygons.append(polygon)
     
    if RT_data_flow:  
        vehicle_x1, vehicle_y1 = rotate_and_translate(1.05, 2.5, map_heading, veh_x, veh_y)
        vehicle_x2, vehicle_y2 = rotate_and_translate(-1.05, 2.5, map_heading, veh_x, veh_y)
        vehicle_x3, vehicle_y3 = rotate_and_translate(-1.05, -2.5, map_heading, veh_x, veh_y)
        vehicle_x4, vehicle_y4 = rotate_and_translate(1.05, -2.5, map_heading, veh_x, veh_y)
        vehicle_polygon = [vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, vehicle_x3, vehicle_y3, vehicle_x4, vehicle_y4]
    
    if LiDAR_data_flow and RT_data_flow:
        return polygons, vehicle_polygon
    elif LiDAR_data_flow and not RT_data_flow:
        return polygons, None
    elif not LiDAR_data_flow and RT_data_flow:
        return None, vehicle_polygon

