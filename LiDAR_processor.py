# LiDAR_processor.py

import socket
import queue 

def receive_bounding_boxes(grid_queue):
    # Create a TCP/IP socket
    server_address = ('localhost', 54320)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to the server address
    sock.bind(server_address)
    
    # Listen for incoming connections (only one at a time)
    sock.listen(1)
    
    while True:
        connection, client_address = sock.accept()
        
        try:
            # Receive bounding box coordinates
            while True:
                data = connection.recv(1024)
                if data:
                    data_str = data.decode()
                    # Split the incoming string by semicolon to separate different bounding boxes
                    boxes = data_str.strip().split(';')
                    # Create a list of lists to store each bounding box's coordinates (detections)
                    detections = []
                    for box in boxes:
                        if box:
                            try:
                                coords = list(map(float, box.split(',')))
                                if len(coords) == 8:  # Each bounding box should have 8 coordinates
                                    detections.append(coords)
                                else:
                                    print(f'Error: Expected 8 coordinates, got {len(coords)}')
                            except ValueError as e:
                                print(f'Error converting to float: {e}, box: {box}')
                    
                    # Put detections into the queue
                    grid_queue.put(detections)
        finally:
            # Close the connection
            connection.close()

if __name__ == '__main__':
    grid_queue = queue.Queue()
    receive_bounding_boxes(grid_queue)
