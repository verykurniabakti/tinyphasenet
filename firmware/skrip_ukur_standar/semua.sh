#!/bin/bash
# Twelve sessions of the quantized standard-convolution PhaseNet (4 configurations x seeds 42, 43, 44). Usage: PORT=<serial port> WORK=<work folder> ./semua.sh
# See jalan.sh for the contents of WORK. One three-channel 16x8 session takes about 16 minutes, hence the read limit of 2400 s.
: "${PORT:?atur PORT}"; : "${WORK:?atur WORK}"; D=$(cd "$(dirname "$0")" && pwd)
for sd in 42 43 44; do
  for cfg in "int8 1ch_int8 Profiling" "16x8 1ch_16x8 Profiling" "3ch_int8 3ch_int8 Benchmark" "3ch_16x8 3ch_16x8 Benchmark"; do
    set -- $cfg
    echo "$(date +%H:%M:%S) mulai seed$sd $2" >> "$WORK/hw/progres.txt"
    r=$("$D/jalan.sh" $sd $1 $2 "$3 selesai" 2400 2>&1 | tail -2 | tr '\n' ' ')
    echo "$(date +%H:%M:%S) seed$sd $2: $r" >> "$WORK/hw/progres.txt"
  done
done
echo SEMUA_SELESAI >> "$WORK/hw/progres.txt"
