#!/bin/sh
# Builds and runs the kernel equivalence test on a computer (no board). From this folder: sh jalankan.sh
set -e
cd "$(dirname "$0")"
M=../managed_components/espressif__esp-tflite-micro
c++ -std=c++17 -O2 -ffp-contract=off -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON \
  -I$M -I$M/third_party/flatbuffers/include -I$M/third_party/gemmlowp -I$M/third_party/ruy -I$M/third_party/kissfft \
  teste_ekuivalensi.cc $M/tensorflow/lite/kernels/internal/common.cc -o teste_ekuivalensi -Wl,-undefined,dynamic_lookup
./teste_ekuivalensi
