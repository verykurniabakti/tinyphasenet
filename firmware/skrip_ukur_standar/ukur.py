import serial, sys, time
port, out, limit = sys.argv[1], sys.argv[2], float(sys.argv[3]); marks = sys.argv[4:]
ser = serial.Serial(port, 115200, timeout=1); ser.dtr = False; ser.rts = False
t0 = time.time(); ok = False
with open(out, 'w') as f:
    while time.time() - t0 < limit:
        line = ser.readline()
        if not line: continue
        txt = line.decode('utf-8', errors='replace'); f.write(txt); f.flush()
        if any(m in txt for m in marks): ok = True; break
ser.close(); print('SELESAI' if ok else 'TIMEOUT', round(time.time() - t0), 's')
