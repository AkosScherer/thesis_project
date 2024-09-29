#RT_processor.py

import socket
import struct
import math

BUFFER_SIZE = 4096
SOCKET_TIMEOUT = 2  # Timeout in seconds

def decimal_to_dms(deg):
    d = int(deg)
    min_float = abs(deg - d) * 60
    m = int(min_float)
    s = (min_float - m) * 60
    return d, m, s

# Approximate conversion using equirectangular projection
def latlon_to_xy(lat0, lon0, lat, lon):
    R = 6371000  # Earth's radius in meters
    x = R * math.radians(lon - lon0) * math.cos(math.radians((lat0 + lat) / 2))
    y = R * math.radians(lat - lat0)
    return x, y

# Rotating x and y coordinates with a given angle
def rotate_coordinates(x, y, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    x_rot = cos_angle * x - sin_angle * y
    y_rot = sin_angle * x + cos_angle * y
    return x_rot, y_rot

# Calculating the distance between current vehicle position and target point position
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def process_RT_data(ip, port, target_xy):
    # Connect to the UDP server
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(SOCKET_TIMEOUT)  # Set timeout for the socket
    server_address = (ip, port)
    sock.bind(server_address)

    full_time = 0
    lat0, lon0 = None, None
    initial_heading = None

    try:
        while True:
            try:
                # Receive data from the socket
                data, address = sock.recvfrom(BUFFER_SIZE)
                
                # Process RT data
                if len(data) >= 72:
                    sync_byte = data[0]
                    navigation_status_byte = data[21]
                    status_channel_byte = data[62]
                    if sync_byte == 0xE7 and navigation_status_byte == 0x04:
                        if status_channel_byte == 0x00:
                            full_time = struct.unpack_from('<I', data, offset=63)[0] * 60
                        # Extract data from bytes according to the NCOM format
                        milisec = struct.unpack_from('<H', data, offset=1)[0]  # Extracting 2 bytes starting from byte 4
                        # Building the timestamp
                        timestamp = full_time + milisec

                        # Extract latitude, longitude, heading, nort and east velocity
                        latitude_rad = struct.unpack_from('<d', data, offset=23)[0]
                        longitude_rad = struct.unpack_from('<d', data, offset=31)[0]
                        heading_rad = int.from_bytes(data[52:55], byteorder='little', signed=True) * 1e-6

                        north_velocity = int.from_bytes(data[43:46], byteorder='little', signed=True) * 1e-4
                        east_velocity = int.from_bytes(data[46:49], byteorder='little', signed=True) * 1e-4
                        
                        # Calculatin vehicle x, y and normal velicity
                        normal_velocity = math.sqrt(north_velocity**2 + east_velocity**2)
                        velocity_x = normal_velocity * math.sin(heading_rad) * -1
                        velocity_y = normal_velocity * math.cos(heading_rad) * -1

                        # Convert from radians to degrees
                        latitude_deg = math.degrees(latitude_rad)
                        longitude_deg = math.degrees(longitude_rad)
                        heading_deg = math.degrees(heading_rad)
                        
                        lat_lon = [latitude_deg, longitude_deg]
                        
                        # Initialize the origin (lat0, lon0) with the first received coordinates
                        if lat0 is None and lon0 is None:
                            lat0, lon0 = latitude_deg, longitude_deg
                            initial_heading = heading_deg  # Store the initial heading
                        
                        # Convert lat/lon to x/y coordinates
                        x, y = latlon_to_xy(lat0, lon0, latitude_deg, longitude_deg)
                        
                        # Rotate coordinates based on initial heading
                        x_rot, y_rot = rotate_coordinates(x, y, initial_heading)
                        veh_xy = [x_rot, y_rot]

                        # Calculate distance to target point
                        distance = calculate_distance(x_rot, y_rot, target_xy[0], target_xy[1])
                        
                        map_heading= heading_deg - initial_heading
                        
                        yield lat_lon, map_heading, timestamp, distance, veh_xy, normal_velocity, velocity_x, velocity_y

            except socket.timeout:
                timeout = True
                

    finally:
        # Close the socket connection
        sock.close()