import serial, sys, time
port, out, limit = sys.argv[1], sys.argv[2], float(sys.argv[3])
ser = serial.Serial(port, 115200, timeout=1)
t0 = time.time()
with open(out, 'w') as f:
    while time.time() - t0 < limit:
        line = ser.readline()
        if not line:
            continue
        txt = line.decode('utf-8', errors='replace')
        f.write(txt); f.flush()
        if 'Profiling selesai' in txt:
            break
ser.close()
