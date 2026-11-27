"""
Step 88: ESP32-S3 firmware input for ONE final TinyPhaseNet seed (16x8 variant). Uses the SAME 100 STEAD windows as 10_make_hw_eval_inputs.py (50 earthquakes + 50 noise, scaled int16
trace) and only replaces: (1) include/tinyphasenet_16x8_model.h from models/final_16x8/seed<N>.tflite; (2) HW_THR_16X8 and HW_PY_16X8 in include/hw_eval_stead.h
(the Python TFLite score of that seed on the same int16 data; threshold = the 16x8 STEAD val threshold of that seed from results/final/quant16x8_per_seed.json).
Usage: python 88_make_fw_seed_inputs.py <seed>
"""
import os, re, sys, json, importlib
import numpy as np
from scipy.signal import sosfilt
import config as C
from data import load_stead, PREFILTER_SOS

mk = importlib.import_module('05_make_firmware_inputs')
hw = importlib.import_module('10_make_hw_eval_inputs')
FW = hw.FW

if __name__ == '__main__':
    seed = int(sys.argv[1])
    q = json.load(open('results/final/quant16x8_per_seed.json'))[str(seed)]
    thr = q['thr_16x8']
    path = f'models/final_16x8/seed{seed}.tflite'
    meta, z = load_stead()
    pick = hw.select(meta[meta.split == 'test'], z)
    full = np.asarray(z[pick.index.to_numpy()], dtype=np.float64)
    scale = np.abs(full).max(axis=1) / 32767.0
    scale[scale == 0] = 1.0
    qd = np.clip(np.round(full / scale[:, None]), -32768, 32767).astype(np.int16)
    deq = (qd.astype(np.float32) * scale[:, None].astype(np.float32)).astype(np.float32)
    starts = pick.eval_start.to_numpy().astype(int)
    pre = sosfilt(PREFILTER_SOS, deq.astype(np.float64) - deq.astype(np.float64).mean(axis=1, keepdims=True), axis=1)
    windows = np.stack([pre[k, st:st + C.WINDOW] for k, st in enumerate(starts)])
    py = np.array(mk.tflite_scores(path, windows))

    h = os.path.join(FW, 'hw_eval_stead.h')
    s = open(h).read()
    s, n1 = re.subn(r'const float HW_THR_16X8 = [^;]*;', f'const float HW_THR_16X8 = {mk.c_float(thr)};', s)
    s, n2 = re.subn(r'const float HW_PY_16X8\[HW_N\] = \{[^}]*\};', 'const float HW_PY_16X8[HW_N] = {' + ', '.join(mk.c_float(v) for v in py) + '};', s)
    assert n1 == 1 and n2 == 1
    open(h, 'w').write(s)
    blob = open(path, 'rb').read()
    with open(os.path.join(FW, 'tinyphasenet_16x8_model.h'), 'w') as m:
        m.write(f'#pragma once\n// Dibuat oleh 88_make_fw_seed_inputs.py dari {path}\nalignas(16) const unsigned char g_model[] = {{\n')
        m.write(mk.c_array(blob, 12, lambda b: f'0x{b:02x}'))
        m.write(f'\n}};\nconst unsigned int g_model_len = {len(blob)};\n')
    pred = py >= thr
    y = pick.label.to_numpy()
    json.dump({'seed': seed, 'thr': thr, 'py_scores': py.tolist(), 'labels': y.tolist(), 'py_tp': int((pred & (y == 1)).sum()), 'py_fp': int((pred & (y == 0)).sum())},
              open(f'results/final/fw_input_seed{seed}.json', 'w'))
    print(f'seed {seed}: model {len(blob)} bytes, threshold {thr:.6f}, Python TP {int((pred & (y == 1)).sum())} FP {int((pred & (y == 0)).sum())}')
