import can
import math
import time

control = True
new_steering_angle = 0
current_steering_angle = 0
alive_counter = 0

# Initialize the CAN bus once
bus = can.Bus(interface='pcan', channel='PCAN_USBBUS1', bitrate=500)

def steering(new_steering_angle, alive_counter, control):
    global current_steering_angle
    
    if new_steering_angle > 180:
        new_steering_angle = 180
    elif new_steering_angle < -180:
        new_steering_angle = -180
        
    angles = []
    if current_steering_angle - 2 <= new_steering_angle <= current_steering_angle + 2:
        angles.append(new_steering_angle)
    elif new_steering_angle > current_steering_angle + 2:
        iterations = math.ceil((new_steering_angle  - current_steering_angle) / 2)
        for i in range(iterations-1):
            angles.append(current_steering_angle + ((i+1) * 2))
        angles.append(new_steering_angle)
    elif new_steering_angle < current_steering_angle - 2:
        iterations = math.ceil((current_steering_angle  - new_steering_angle) / 2)
        for i in range(iterations-1):
            angles.append(current_steering_angle - ((i+1) * 2))
        angles.append(new_steering_angle)
        
    for angle in angles:
        send_one(angle, control)
    
    current_steering_angle = new_steering_angle

def encode_can_data(data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9):
    data = (
        (data_1 & 0xFF) |
        ((data_2 & 0xF) << 8) |
        ((data_3 & 0x7) << 12) |
        ((data_4 & 0xFF) << 15) |
        ((data_5 & 0x1) << 23) |
        ((data_6 & 0x7) << 24) |
        ((data_7 & 0xFFF) << 27) |
        ((data_8 & 0x7) << 39) |
        ((data_9 & 0xFFFF) << 42)
    )
    return bytearray(data.to_bytes(8, byteorder='little'))

def send_one(steering_angle, control):
    global alive_counter
    
    if control:
        Alive_DFLaDMCOutput01 = alive_counter
        LaDMC_Status__nu = 1
        LaDMC_SteerAngReq__deg = int(65535 * ((steering_angle + 800) / 1638.375))

        data = encode_can_data(0, Alive_DFLaDMCOutput01, LaDMC_Status__nu, 0, 0, 0, 0, 0, LaDMC_SteerAngReq__deg)

        data[0] = 0x00
        data[2] = 0x80
        data[3] = 0x80
        data[4] = 0xBE
    
    else:
        Alive_DFLaDMCOutput01 = alive_counter
        LaDMC_Status__nu = 0
        LaDMC_SteerAngReq__deg = 0

        data = encode_can_data(0, Alive_DFLaDMCOutput01, LaDMC_Status__nu, 0, 0, 0, 0, 0, LaDMC_SteerAngReq__deg)

        data[0] = 0x00
        data[2] = 0x00
        data[3] = 0x80
        data[4] = 0x3E
        data[5] = 0x00
        data[6] = 0xF4
        data[7] = 0x01

    msg = can.Message(
        arbitration_id=0x74, data=data, is_extended_id=False
    )
    
    if alive_counter < 14:
        alive_counter += 1
    else:
        alive_counter = 0

    try:
        bus.send(msg)
        print(f"Message sent on {bus.channel_info}")
    except can.CanError:
        print("Message NOT sent")

    print('steering angle:', steering_angle, 'control: ', control)   
    time.sleep(0.1) 

if __name__ == "__main__":

    for i in range(50):
        if control:
            steering(new_steering_angle, alive_counter, control)
        else:
            send_one(0, control)
            
        if new_steering_angle == 0:
            new_steering_angle = -200



#BO_ 116 AP_DFLaDMCOutput01: 8 AP
# SG_ LaDMC_SteerAngReq__deg : 42|16@1+ (0.025,-800) [-800|838.375] "deg"  Gateway
# SG_ LaDMC_SteerAngReqQF__nu : 39|3@1+ (1,0) [0|7] ""  Gateway
# SG_ LaDMC_SteerTrqReq__nm : 27|12@1+ (0.004,-8) [-8|8.38] "Nm"  Gateway
# SG_ LaDMC_SteerTrqReqQF__nu : 24|3@1+ (1,0) [0|7] ""  Gateway
# SG_ LaDMC_ActEPSInterface__nu : 23|1@1+ (1,0) [0|1] ""  Gateway
# SG_ LaDMC_Eps_Damping_Level_Req__per : 15|8@1+ (0.01,0) [0|2.55] "%"  Gateway
# SG_ LaDMC_Status__nu : 12|3@1+ (1,0) [0|7] ""  Gateway
# SG_ Alive_DFLaDMCOutput01 : 8|4@1+ (1,0) [0|15] ""  Gateway
# SG_ CRC_DFLaDMCOutput01 : 0|8@1+ (1,0) [0|255] ""  Gateway