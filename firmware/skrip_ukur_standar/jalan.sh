#!/bin/bash
# Measures one session on the ESP32-S3: one quantized standard-convolution PhaseNet model (16x8 or INT8), one seed.
# Usage:  PORT=<serial port> WORK=<work folder> ./jalan.sh <seed> <env> <name> <marker> <time limit, seconds>
#   <env>      PlatformIO environment (int8, 16x8, 3ch_int8, 3ch_16x8); <name> label of the log file; <marker> text that ends the session on the serial output
# WORK must contain: fw_std/ (a copy of the firmware folder, with main_3ch_16x8_wdt.cpp replacing src/main_3ch_16x8.cpp for 3ch_16x8),
#   inc_std/s<seed>/ (headers from scripts/pipeline/230_make_fw_standar.py for that seed), ukur.py (the file in this folder), and hw/ (log output).
# PIO and PY can be set (defaults: pio and python3). The board is not reset through RTS (the upload already resets it).
: "${PORT:?atur PORT}"; : "${WORK:?atur WORK}"; PIO=${PIO:-pio}; PY=${PY:-python3}
cd "$WORK/fw_std" && rsync -a "$WORK/inc_std/s$1/" include/ && $PIO run -e "$2" -t upload --upload-port "$PORT" > "$WORK/hw/build_$3_seed$1.log" 2>&1 || { echo UPLOAD_GAGAL; tail -5 "$WORK/hw/build_$3_seed$1.log"; exit 1; }
$PY "$WORK/ukur.py" "$PORT" "$WORK/hw/padrao_seed$1_$3.log" "$5" "$4" "[ERROR]"
