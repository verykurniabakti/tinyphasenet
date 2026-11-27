"""
Step 191 (device manuscript, Q, methodology improvement stage 4): firmware headers for the STANDARD-CONVOLUTION Float32 models on the ESP32-S3, the counterpart of 170_make_fw_float32.py
for the depthwise separable models. One channel: models/ablation_final/standard_seed<S>.keras (plain Conv1D, 76,195 parameters); three channels: models/ablation_final/pn_seed<S>.keras
(PhaseNet with plain Conv1D, 269,675 parameters). The training recipe is the same as the depthwise separable models (89_train_ablation_final.py).
Input: the SAME 100 STEAD test windows as 170_ (select() in 10_make_hw_eval_inputs.py, random_state=C.SEED; 50 earthquakes + 50 noise), the same preprocessing procedure (163_ one channel, 165_ three channels), stored as int16 with one global scale and dequantized on the board.
Threshold = 'threshold' (chosen on the STEAD validation) from results/ablation_final/eval_standard.json and eval_pn.json for the corresponding seed.
The output file names are the SAME as 170_ (phasenet_fp32_{1ch,3ch}_model.h, hw_eval_fp32_{1ch,3ch}.h) so that src/fp32_bench.cpp can be used as it is; for this reason the headers are written to the include directory of
a COPY of the firmware project (argument), not to the repository. Build with env fp32s_1ch (internal SRAM arena, fits) and fp32f_3ch (PSRAM arena), both with the replacement TRANSPOSE_CONV kernel.
Usage (from this folder): python 191_make_fw_float32_standard.py <include directory of the copy> [seed; default 42]
Output: the headers above and results/final/fw_input_float32_standard_seed<S>.json.
Needs: STEAD data at config.STEAD_HDF5 (the external SSD must be mounted).
"""
import os, sys, json, importlib
import numpy as np
import tensorflow as tf
from scipy.signal import sosfilt
import config as C
import data as D
import data3ch as D3

mk = importlib.import_module('05_make_firmware_inputs'); hw = importlib.import_module('10_make_hw_eval_inputs')
dest = sys.argv[1]; SEED = sys.argv[2] if len(sys.argv) > 2 else '42'   # seed model (bawaan 42)


def to_fp32(path, K):
    m = tf.keras.models.load_model(path, compile=False)
    conc = tf.function(lambda x: m(x)).get_concrete_function(tf.TensorSpec([1, C.WINDOW, K], tf.float32))
    return tf.lite.TFLiteConverter.from_concrete_functions([conc], m).convert()


def scores(blob, X):
    it = tf.lite.Interpreter(model_content=blob); it.allocate_tensors(); inp, out = it.get_input_details()[0], it.get_output_details()[0]
    assert inp['dtype'] == np.float32
    sc = []
    for k in range(len(X)):
        it.set_tensor(inp['index'], X[k:k + 1].astype(np.float32)); it.invoke(); sc.append(float(it.get_tensor(out['index'])[0, :, 0].max()))
    return np.array(sc)


def quant_input(X):
    s = float(np.abs(X).max() / 32767.0); q = np.clip(np.round(X / s), -32768, 32767).astype(np.int16)
    return q, s, (q.astype(np.float32) * np.float32(s)).astype(np.float32)


meta, z = D.load_stead(); pick = hw.select(meta[meta.split == 'test'], z); y = pick.label.to_numpy()
# one channel: procedure of 163_ (scaled int16 trace -> band-pass -> cut window -> z-score)
full = np.asarray(z[pick.index.to_numpy()], dtype=np.float64); scale = np.abs(full).max(axis=1) / 32767.0; scale[scale == 0] = 1.0
qd = np.clip(np.round(full / scale[:, None]), -32768, 32767).astype(np.int16); deq = (qd.astype(np.float32) * scale[:, None].astype(np.float32)).astype(np.float32)
starts = pick.eval_start.to_numpy().astype(int)
pre = sosfilt(D.PREFILTER_SOS, deq.astype(np.float64) - deq.astype(np.float64).mean(axis=1, keepdims=True), axis=1)
X1 = np.stack([mk.zscore(pre[k, st:st + C.WINDOW]).reshape(C.WINDOW, 1) for k, st in enumerate(starts)]).astype(np.float32); del z
# three channels: procedure of 165_
meta3, w3 = D3.load_stead3(); X3, y3 = D3.eval_windows3(meta3.loc[pick.index], w3, prefilter_input=True); assert (y3 == y).all(); del w3

THR = lambda fn: {int(e['model'].split('seed')[1].split('.')[0]): e['threshold'] for e in json.load(open(fn))}
CFG = {'1ch': (f'models/ablation_final/standard_seed{SEED}.keras', 1, X1, THR('results/ablation_final/eval_standard.json')[int(SEED)]),
       '3ch': (f'models/ablation_final/pn_seed{SEED}.keras', 3, X3, THR('results/ablation_final/eval_pn.json')[int(SEED)])}
info = {}
for tag, (path, K, X, thr) in CFG.items():
    blob = to_fp32(path, K); q, s, Xd = quant_input(X.astype(np.float32).reshape(len(X), C.WINDOW, K)); py = scores(blob, Xd); py_raw = scores(blob, X.astype(np.float32).reshape(len(X), C.WINDOW, K))
    pred = py >= thr
    info[tag] = dict(bytes=len(blob), thr=float(thr), scale=s, max_abs_dequant_score_diff=float(np.abs(py - py_raw).max()), py_tp=int((pred & (y == 1)).sum()), py_fp=int((pred & (y == 0)).sum()), py_fn=int((~pred & (y == 1)).sum()), py_tn=int((~pred & (y == 0)).sum()),
                     decisions_changed_by_int16_input=int(((py_raw >= thr) != pred).sum()))
    print(tag, info[tag], flush=True)
    with open(os.path.join(dest, f'phasenet_fp32_{tag}_model.h'), 'w') as f:
        f.write(f'#pragma once\n// Dibuat oleh 191_make_fw_float32_standard.py dari {path} (TFLite Float32, tanpa optimasi)\nalignas(16) const unsigned char g_model_fp32[] = {{\n' + mk.c_array(blob, 12, lambda b: f'0x{b:02x}') + f'\n}};\nconst unsigned int g_model_fp32_len = {len(blob)};\n')
    with open(os.path.join(dest, f'hw_eval_fp32_{tag}.h'), 'w') as f:
        f.write(f'// Dibuat oleh 191_make_fw_float32_standard.py: 100 jendela STEAD (50 gempa + 50 noise), masukan z-score disimpan int16 dengan skala global, skor Python Float32 pada masukan terdekuantisasi\n#pragma once\n#include <stdint.h>\n')
        f.write(f'#define FP_N 100\n#define FP_WINDOW {C.WINDOW}\n#define FP_CH {K}\nstatic const float FP_SCALE = {mk.c_float(s)};\nstatic const float FP_THR = {mk.c_float(thr)};\n')
        f.write('static const int FP_LABELS[FP_N] = {' + ', '.join(str(int(v)) for v in y) + '};\n')
        f.write('static const float FP_PY_SCORES[FP_N] = {' + ', '.join(mk.c_float(v) for v in py) + '};\n')
        f.write(f'static const int16_t FP_INPUTS[FP_N][FP_WINDOW * FP_CH] = {{\n')
        for k in range(len(q)): f.write('  {' + ', '.join(str(int(v)) for v in q[k].reshape(-1)) + '},\n')
        f.write('};\n')
json.dump(info, open(f'results/final/fw_input_float32_standard_seed{SEED}.json', 'w'), indent=1)
