"""
Step 163 (answer to the advisor, comparable ESP32-S3 measurement): firmware input for FINAL full INT8 (models/quant_spectrum/1ch_int8_seed42.tflite, 159_) and final 16x8 seed 42 in ONE board session.
Does not change files in the repository: the headers are written to the include directory of a COPY of the firmware project given as an argument. The test data and procedure are the same as 88_make_fw_seed_inputs.py (100 STEAD windows, scaled int16 trace).
Replaces: tinyphasenet_int8_model.h, tinyphasenet_16x8_model.h, and HW_THR_INT8, HW_PY_INT8, HW_THR_16X8, HW_PY_16X8 in hw_eval_stead.h (base file: the copy of hw_eval_stead.h in the target directory).
Usage: python 163_make_fw_int8_final.py <include directory of the copy> [seed; default 42]
"""
import os, re, sys, json, importlib
import numpy as np
from scipy.signal import sosfilt
import config as C
from data import load_stead, PREFILTER_SOS

mk = importlib.import_module('05_make_firmware_inputs'); hw = importlib.import_module('10_make_hw_eval_inputs')
dest = sys.argv[1]; SEED = sys.argv[2] if len(sys.argv) > 2 else '42'   # seed model (bawaan 42)
S = json.load(open('results/final/spektrum_kuantisasi_3seed.json'))['1ch'][SEED]; q16 = json.load(open('results/final/quant16x8_per_seed.json'))[SEED]
paths = {'INT8': (f'models/quant_spectrum/1ch_int8_seed{SEED}.tflite', S['int8']['thr'], 'tinyphasenet_int8_model.h'), '16X8': (f'models/final_16x8/seed{SEED}.tflite', q16['thr_16x8'], 'tinyphasenet_16x8_model.h')}
meta, z = load_stead(); pick = hw.select(meta[meta.split == 'test'], z)
full = np.asarray(z[pick.index.to_numpy()], dtype=np.float64); scale = np.abs(full).max(axis=1) / 32767.0; scale[scale == 0] = 1.0
qd = np.clip(np.round(full / scale[:, None]), -32768, 32767).astype(np.int16); deq = (qd.astype(np.float32) * scale[:, None].astype(np.float32)).astype(np.float32)
starts = pick.eval_start.to_numpy().astype(int)
pre = sosfilt(PREFILTER_SOS, deq.astype(np.float64) - deq.astype(np.float64).mean(axis=1, keepdims=True), axis=1)
windows = np.stack([pre[k, st:st + C.WINDOW] for k, st in enumerate(starts)])
h = os.path.join(dest, 'hw_eval_stead.h'); s = open(h).read(); y = pick.label.to_numpy(); info = {}
for tag, (path, thr, hdr) in paths.items():
    py = np.array(mk.tflite_scores(path, windows))
    s, n1 = re.subn(rf'const float HW_THR_{tag} = [^;]*;', f'const float HW_THR_{tag} = {mk.c_float(thr)};', s)
    s, n2 = re.subn(rf'const float HW_PY_{tag}\[HW_N\] = \{{[^}}]*\}};', f'const float HW_PY_{tag}[HW_N] = {{' + ', '.join(mk.c_float(v) for v in py) + '};', s)
    assert n1 == 1 and n2 == 1, (tag, n1, n2)
    blob = open(path, 'rb').read()
    with open(os.path.join(dest, hdr), 'w') as m:
        m.write(f'#pragma once\n// Dibuat oleh 163_make_fw_int8_final.py dari {path}\nalignas(16) const unsigned char g_model[] = {{\n' + mk.c_array(blob, 12, lambda b: f'0x{b:02x}') + f'\n}};\nconst unsigned int g_model_len = {len(blob)};\n')
    pred = py >= thr; info[tag] = dict(bytes=len(blob), thr=float(thr), py_tp=int((pred & (y == 1)).sum()), py_fp=int((pred & (y == 0)).sum()), py_fn=int((~pred & (y == 1)).sum()), py_tn=int((~pred & (y == 0)).sum()))
    print(tag, info[tag], flush=True)
open(h, 'w').write(s); json.dump(info, open(f'results/final/fw_input_int8_16x8_seed{SEED}_sesi.json', 'w'), indent=1)
