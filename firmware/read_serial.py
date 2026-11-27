import serial
import sys
import time

port = sys.argv[1] if len(sys.argv) > 1 else '/dev/cu.wchusbserial5B910000981'
duration_s = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0

ser = serial.Serial(port, 115200, timeout=1)
t0 = time.time()
while time.time() - t0 < duration_s:
    line = ser.readline()
    if line:
        try:
            print(line.decode('utf-8', errors='replace'), end='', flush=True)
        except Exception:
            pass
ser.close()
