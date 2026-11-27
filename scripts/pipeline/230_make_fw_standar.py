"""
Step 230 (companion compression manuscript: ESP32-S3 measurement of the STANDARD-convolution model, 16x8 and INT8): firmware input for the quantized standard PhaseNet
(models/quant_padrao/{1ch,3ch}_standard_{q16x8,int8}_seed<N>.tflite, made by 199_; threshold from results/final/ptq_padrao_3seed.json).
Writes to the include directory of a COPY of the firmware project (argument), with the same FILE AND SYMBOL NAMES as the TinyPhaseNet variants, so that the envs int8, 16x8, 3ch_int8, 3ch_16x8 in platformio.ini can be used unchanged:
  one channel: tinyphasenet_int8_model.h, tinyphasenet_16x8_model.h, and HW_THR_*/HW_PY_* in hw_eval_stead.h (procedure of 163_; the copy of hw_eval_stead.h in the target directory is the base)
  three channels: phasenet_3ch_int8_model.h, phasenet_3ch_16x8_model.h, hw_eval_3ch_100_int8.h, hw_eval_3ch_100.h (procedure of 165_; the 3-channel windows are quantized with the input scale and zero point of the standard model)
The same 100 STEAD test windows as the TinyPhaseNet measurements (10_make_hw_eval_inputs.select).
Does not change files in the repository. Usage (from this folder): python 230_make_fw_standar.py <include directory of the copy> <1ch|3ch> [seed; default 42]
"""
import os, re, sys, json, importlib
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import config as C
import data as D

mk = importlib.import_module('05_make_firmware_inputs'); hw = importlib.import_module('10_make_hw_eval_inputs')
dest, FAM = sys.argv[1], sys.argv[2]; SEED = sys.argv[3] if len(sys.argv) > 3 else '42'
assert FAM in ('1ch', '3ch')
P = json.load(open('results/final/ptq_padrao_3seed.json'))[FAM]
FMT = {'INT8': 'int8', '16X8': 'q16x8'}
path = lambda f: f'models/quant_padrao/{FAM}_standard_{FMT[f]}_seed{SEED}.tflite'
thr = {f: P[FMT[f]][SEED]['thr'] for f in FMT}
meta, z = D.load_stead(); pick = hw.select(meta[meta.split == 'test'], z); y = pick.label.to_numpy(); info = {}


def stats(py, t):
    pred = py >= t
    return dict(thr=float(t), py_tp=int((pred & (y == 1)).sum()), py_fp=int((pred & (y == 0)).sum()), py_fn=int((~pred & (y == 1)).sum()), py_tn=int((~pred & (y == 0)).sum()))


if FAM == '1ch':
    from scipy.signal import sosfilt
    full = np.asarray(z[pick.index.to_numpy()], dtype=np.float64); scale = np.abs(full).max(axis=1) / 32767.0; scale[scale == 0] = 1.0
    qd = np.clip(np.round(full / scale[:, None]), -32768, 32767).astype(np.int16); deq = (qd.astype(np.float32) * scale[:, None].astype(np.float32)).astype(np.float32)
    starts = pick.eval_start.to_numpy().astype(int)
    pre = sosfilt(D.PREFILTER_SOS, deq.astype(np.float64) - deq.astype(np.float64).mean(axis=1, keepdims=True), axis=1)
    windows = np.stack([pre[k, st:st + C.WINDOW] for k, st in enumerate(starts)])
    h = os.path.join(dest, 'hw_eval_stead.h'); s = open(h).read()
    hdr = {'INT8': 'tinyphasenet_int8_model.h', '16X8': 'tinyphasenet_16x8_model.h'}
    for tag in FMT:
        py = np.array(mk.tflite_scores(path(tag), windows))
        s, n1 = re.subn(rf'const float HW_THR_{tag} = [^;]*;', f'const float HW_THR_{tag} = {mk.c_float(thr[tag])};', s)
        s, n2 = re.subn(rf'const float HW_PY_{tag}\[HW_N\] = \{{[^}}]*\}};', f'const float HW_PY_{tag}[HW_N] = {{' + ', '.join(mk.c_float(v) for v in py) + '};', s)
        assert n1 == 1 and n2 == 1, (tag, n1, n2)
        blob = open(path(tag), 'rb').read()
        with open(os.path.join(dest, hdr[tag]), 'w') as m:
            m.write(f'#pragma once\n// Dibuat oleh 230_make_fw_standar.py dari {path(tag)}\nalignas(16) const unsigned char g_model[] = {{\n' + mk.c_array(blob, 12, lambda b: f'0x{b:02x}') + f'\n}};\nconst unsigned int g_model_len = {len(blob)};\n')
        info[tag] = dict(bytes=len(blob), **stats(py, thr[tag])); print(tag, info[tag], flush=True)
    open(h, 'w').write(s)
else:
    import data3ch as D3
    del z
    meta3, w3 = D3.load_stead3(); X, y3 = D3.eval_windows3(meta3.loc[pick.index], w3, prefilter_input=True); assert (y3 == y).all(); del w3
    labs = ', '.join(str(int(v)) for v in y)

    def run(p):
        it = tf.lite.Interpreter(model_path=p); it.allocate_tensors(); inp, out = it.get_input_details()[0], it.get_output_details()[0]
        (si, zi), (so, zo) = inp['quantization'], out['quantization']; ii = np.iinfo(inp['dtype']); q = np.clip(np.round(X / si) + zi, ii.min, ii.max).astype(inp['dtype']); sc = []
        for k in range(len(q)):
            it.set_tensor(inp['index'], q[k:k + 1]); it.invoke(); o = (it.get_tensor(out['index']).astype(np.float32) - zo) * so; sc.append(float(o[0, :, 0].max()))
        return q, np.array(sc), (float(si), int(zi), float(so), int(zo)), inp['dtype']
    for tag, (mh, eh, ctype) in {'INT8': ('phasenet_3ch_int8_model.h', 'hw_eval_3ch_100_int8.h', 'int8_t'), '16X8': ('phasenet_3ch_16x8_model.h', 'hw_eval_3ch_100.h', 'int16_t')}.items():
        q, py, par, dt = run(path(tag)); assert dt == (np.int8 if tag == 'INT8' else np.int16), dt
        blob = open(path(tag), 'rb').read()
        with open(os.path.join(dest, mh), 'w') as f:
            f.write(f'// Dibuat oleh 230_make_fw_standar.py dari {path(tag)}\n#pragma once\n\nalignas(16) const unsigned char g_model_3ch_16x8[] = {{\n' + mk.c_array(blob, 12, lambda b: f'0x{b:02x}') + f'\n}};\nconst unsigned int g_model_3ch_16x8_len = {len(blob)};\n')
        with open(os.path.join(dest, eh), 'w') as f:
            f.write(f'// Dibuat oleh 230_make_fw_standar.py (model standar {tag} tiga kanal, seed {SEED})\n#pragma once\n#include <stdint.h>\n#define HW_3CH_N 100\n#define HW_3CH_WINDOW 3072\n#define HW_3CH_CHANNELS 3\n')
            f.write(f'static const float HW_3CH_THRESHOLD = {mk.c_float(thr[tag])};\nstatic const int HW_3CH_LABELS[HW_3CH_N] = {{{labs}}};\n')
            f.write('static const float HW_3CH_PY_SCORES[HW_3CH_N] = {' + ', '.join(mk.c_float(v) for v in py) + '};\n')
            f.write(f'static const {ctype} HW_3CH_INPUTS[HW_3CH_N][HW_3CH_WINDOW * HW_3CH_CHANNELS] = {{\n')
            for k in range(len(q)): f.write('  {' + ', '.join(str(int(v)) for v in q[k].reshape(-1)) + '},\n')
            f.write('};\n')
        info[tag] = dict(bytes=len(blob), par=par, **stats(py, thr[tag])); print(tag, info[tag], flush=True)
json.dump(info, open(f'results/final/fw_input_standar_{FAM}_seed{SEED}.json', 'w'), indent=1)
print('done', flush=True)
