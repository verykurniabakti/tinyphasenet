"""
Step 10: device test data for STEAD, INSTANCE, and ETHZ (100 windows per dataset).
- 50 earthquakes + 50 noise windows per dataset (fixed seed) from the test set used by 04_evaluate.py,
  flat windows (std = 0) are not drawn.
- The trace is stored as int16 with a per-trace scale (so that it fits in flash) and cut at the
  end of the test window: STA/LTA is causal, so samples after the window do not affect the score.
  The mean of the FULL trace is stored separately because the STA/LTA in Python subtracts the mean of the full trace.
- The Python reference scores (TinyPhaseNet INT8/16x8 via TFLite, STA/LTA without filter and band-pass via
  04_evaluate.scores_stalta) are computed from the SAME int16 data (after dequantization), so that
  the port validation stays exact.
Output: firmware/tinyphasenet_profiling/include/hw_eval_<dataset>.h and
        results/hw_eval_python_ref.json (Python accuracy on the same 100 windows).
"""
import os
import json
import importlib
import numpy as np
import pandas as pd
from scipy.signal import butter
import config as C
from data import load_stead, load_instance, load_ethz, PREFILTER_SOS
from scipy.signal import sosfilt

FW = os.path.join(C.ROOT, 'firmware', 'tinyphasenet_profiling', 'include')
ev = importlib.import_module('04_evaluate')
mk = importlib.import_module('05_make_firmware_inputs')
N_PER_CLASS = 50
METHODS = {'INT8': 'TinyPhaseNet INT8', '16X8': 'TinyPhaseNet 16x8',
           'PLAIN': 'STA/LTA', 'BP': 'STA/LTA + bandpass'}


def select(meta, z):
    flat = [np.std(z[i, st:st + C.WINDOW]) == 0 for i, st in zip(meta.index, meta.eval_start)]
    meta = meta[~np.array(flat)]
    return (pd.concat([meta[meta.label == c].sample(n=N_PER_CLASS, random_state=C.SEED) for c in (0, 1)])
              .sample(frac=1, random_state=C.SEED))


def metrics(y, s, thr):
    p = s >= thr
    tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum())
    tn = int((~p & (y == 0)).sum()); fn = int((~p & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0, fpr=fp / (fp + tn))


def build(name, meta, z, thresholds, sos):
    pick = select(meta, z)
    full = np.asarray(z[pick.index.to_numpy()], dtype=np.float64)
    scale = np.abs(full).max(axis=1) / 32767.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(full / scale[:, None]), -32768, 32767).astype(np.int16)
    deq = (q.astype(np.float32) * scale[:, None].astype(np.float32)).astype(np.float32)
    starts = pick.eval_start.to_numpy().astype(int)
    labels = pick.label.to_numpy().astype(int)

    # Python reference scores from the same int16 data
    ref_meta = pd.DataFrame({'eval_start': starts})          # index 0..N-1 = row in deq
    # TinyPhaseNet: causal 1-45 Hz band-pass on the trace (full-trace mean subtracted), then the window
    pre = sosfilt(PREFILTER_SOS, deq.astype(np.float64) - deq.astype(np.float64).mean(axis=1, keepdims=True), axis=1)
    windows = np.stack([pre[k, st:st + C.WINDOW] for k, st in enumerate(starts)])
    py = {'INT8': np.array(mk.tflite_scores(os.path.join(C.MODEL_DIR, 'tinyphasenet_int8.tflite'), windows)),
          '16X8': np.array(mk.tflite_scores(os.path.join(C.MODEL_DIR, 'tinyphasenet_16x8.tflite'), windows)),
          'PLAIN': ev.scores_stalta(ref_meta, deq, bandpass=False),
          'BP': ev.scores_stalta(ref_meta, deq, bandpass=True)}
    means = deq.astype(np.float64).mean(axis=1)

    lens = starts + C.WINDOW
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    flat = np.concatenate([q[k, :lens[k]] for k in range(len(q))])
    path = os.path.join(FW, f'hw_eval_{name}.h')
    with open(path, 'w') as f:
        f.write('#pragma once\n#include <stdint.h>\n')
        f.write(f'// Dibuat oleh 10_make_hw_eval_inputs.py: {len(pick)} jendela uji {name.upper()} '
                f'({N_PER_CLASS} gempa + {N_PER_CLASS} noise), trace int16 berskala, dipotong di akhir jendela.\n')
        f.write('// trace: ' + ', '.join(pick.trace_name) + '\n')
        f.write(f'static const char* HW_DATASET = "{name.upper()}";\n')
        f.write(f'const int HW_N = {len(pick)};\nconst int HW_WINDOW = {C.WINDOW};\n')
        f.write(f'const int HW_NSTA = {int(C.STA_S * C.FS)};\nconst int HW_NLTA = {int(C.LTA_S * C.FS)};\n')
        f.write('const int HW_LABELS[HW_N] = {' + ', '.join(map(str, labels)) + '};\n')
        f.write('const int HW_EVAL_START[HW_N] = {' + ', '.join(map(str, starts)) + '};\n')
        f.write('const int HW_OFFSET[HW_N] = {' + ', '.join(map(str, offs)) + '};\n')
        f.write('const float HW_SCALE[HW_N] = {' + ', '.join(mk.c_float(v) for v in scale) + '};\n')
        f.write('// rata-rata trace PENUH (setelah dekuantisasi), dipakai STA/LTA seperti di Python\n')
        f.write('const float HW_MEAN[HW_N] = {' + ', '.join(mk.c_float(v) for v in means) + '};\n')
        f.write(f'const int HW_NSOS = {len(sos)};\n')
        f.write('const float HW_SOS[HW_NSOS][6] = {\n' +
                ',\n'.join('  {' + ', '.join(mk.c_float(v) for v in row) + '}' for row in sos) + '\n};\n')
        f.write(f'const int HW_PRE_NSOS = {len(PREFILTER_SOS)};\n')
        f.write('// bandpass kausal 1-45 Hz untuk input TinyPhaseNet (setara prapemrosesan STEAD)\n')
        f.write('const float HW_PRE_SOS[HW_PRE_NSOS][6] = {\n' +
                ',\n'.join('  {' + ', '.join(mk.c_float(v) for v in row) + '}' for row in PREFILTER_SOS) + '\n};\n')
        for tag, method in METHODS.items():
            f.write(f'const float HW_THR_{tag} = {mk.c_float(thresholds[method])};  // dari split val STEAD\n')
            f.write(f'const float HW_PY_{tag}[HW_N] = {{' + ', '.join(mk.c_float(v) for v in py[tag]) + '};\n')
        f.write(f'const int16_t HW_DATA[{len(flat)}] = {{\n' + mk.c_array(flat, 20, str) + '\n};\n')
    ref = {METHODS[t]: metrics(labels, py[t], thresholds[METHODS[t]]) for t in METHODS}
    print(f"[{name}] {len(pick)} windows, {len(flat) * 2 / 1024:.0f} KB data int16 -> {path}")
    for m, r in ref.items():
        print(f"   {m:28s} P {r['precision']:.2f} R {r['recall']:.2f} F1 {r['f1']:.3f} FPR {r['fpr']:.2f}")
    return ref


if __name__ == '__main__':
    thresholds = json.load(open(os.path.join(C.RESULTS_DIR, 'metrics.json')))['thresholds_from_stead_val']
    sos = butter(C.BANDPASS_ORDER, C.BANDPASS_HZ, btype='band', fs=C.FS, output='sos')
    stead, sz = load_stead()
    inst, iz = load_instance()
    ethz, ez = load_ethz()
    out = {}
    for name, (meta, z) in {'stead': (stead[stead.split == 'test'], sz),
                            'instance': (inst, iz), 'ethz': (ethz, ez)}.items():
        out[name] = build(name, meta, z, thresholds, sos)
    with open(os.path.join(C.RESULTS_DIR, 'hw_eval_python_ref.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("SELESAI.")
