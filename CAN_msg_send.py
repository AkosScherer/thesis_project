import can

def send_can_message(interface, message_id, data_bytes, message_type, is_extended_id):
    try:
        # Create a CAN bus instance
        bus = can.interface.Bus(channel=interface, bustype=message_type)

        # Create a CAN message
        message = can.Message(arbitration_id=message_id, data=data_bytes, is_extended_id=is_extended_id)

        # Send the message on the bus
        bus.send(message)
        print(f"Message sent on {interface} with ID {message_id} and data {data_bytes}")

    except can.CanError as e:
        print(f"Failed to send message: {e}")

if __name__ == "__main__":
    # Define input arguments as variables
    interface = 'can0'  # CAN interface, e.g., 'can0'
    message_id = 123  # Example message ID
    data_bytes = [1, 2, 3, 4]  # Example data bytes
    message_type = 'socketcan'  # Example message type, e.g., 'socketcan', 'pcan', 'kvaser', 'virtual'
    is_extended_id = False  # False for normal ID, True for extended ID

    # Call the function with the defined variables
    send_can_message(interface, message_id, data_bytes, message_type, is_extended_id)
