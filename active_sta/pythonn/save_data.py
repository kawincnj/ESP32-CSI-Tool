import serial
import time
import re

ser = serial.Serial(port='COM7', baudrate=921600, timeout=1)

time.sleep(2)

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline()
            
            decoded_line = line.decode('utf-8').rstrip()
            data = re.findall(r'\[(.*?)\]', decoded_line)
            data = [int(x) for x in data[0].split()] if data else None
            print(data)
            
except KeyboardInterrupt:
    print("Closing connection...")
finally:
    # 5. Always close the port when finished
    ser.close()
