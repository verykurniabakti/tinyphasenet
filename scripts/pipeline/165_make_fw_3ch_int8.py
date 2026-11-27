"""
Step 165 (Pareto optimum, ESP32-S3 three-channel INT8 measurement): firmware input for the full INT8 three-channel depthwise PhaseNet model, seed 42 (models/quant_spectrum/3ch_int8_seed42.tflite, 159_).
The SAME 100 STEAD windows as hw_eval_3ch_100.h (selection 10_make_hw_eval_inputs.select), 3-channel windows filtered and normalized (D3.eval_windows3), quantized with the input scale and zero point of each model.
Writes to the include directory of a COPY of the firmware project (argument): phasenet_3ch_int8_model.h and hw_eval_3ch_100_int8.h (the same symbol names as the 16x8 files so that main_3ch_16x8.cpp can be used with -DMODEL_3CH_INT8).
Check: the Python 16x8 score with the same procedure is compared with HW_3CH_PY_SCORES in the existing hw_eval_3ch_100.h.
Usage: python 165_make_fw_3ch_int8.py <include directory of the copy> [seed; default 42]. The check against the old header is only meaningful for seed 42.
"""
import os, re, sys, json, importlib
import numpy as np
import tensorflow as tf
import config as C
import data as D
import data3ch as D3

mk = importlib.import_module('05_make_firmware_inputs'); hw = importlib.import_module('10_make_hw_eval_inputs')
dest = sys.argv[1]; SEED = sys.argv[2] if len(sys.argv) > 2 else '42'; REPO_INC = hw.FW
S = json.load(open('results/final/spektrum_kuantisasi_3seed.json'))['3ch'][SEED]
meta, z = D.load_stead(); pick = hw.select(meta[meta.split == 'test'], z); del z
meta3, w3 = D3.load_stead3(); X, y = D3.eval_windows3(meta3.loc[pick.index], w3, prefilter_input=True)
assert (y == pick.label.to_numpy()).all()


def run(path):
    it = tf.lite.Interpreter(model_path=path); it.allocate_tensors(); inp, out = it.get_input_details()[0], it.get_output_details()[0]
    (si, zi), (so, zo) = inp['quantization'], out['quantization']; info = np.iinfo(inp['dtype']); q = np.clip(np.round(X / si) + zi, info.min, info.max).astype(inp['dtype']); sc = []
    for k in range(len(q)):
        it.set_tensor(inp['index'], q[k:k + 1]); it.invoke(); o = (it.get_tensor(out['index']).astype(np.float32) - zo) * so; sc.append(float(o[0, :, 0].max()))
    return q, np.array(sc), (float(si), int(zi), float(so), int(zo))


# pemeriksaan 16x8
q16, py16, par16 = run(f'models/final_16x8_pndw/seed{SEED}.tflite')
old = open(os.path.join(REPO_INC, 'hw_eval_3ch_100.h')).read(); m = re.search(r'HW_3CH_PY_SCORES\[HW_3CH_N\] = \{([^}]*)\}', old); old_py = np.array([float(v.replace('f', '')) for v in m.group(1).split(',') if v.strip()])
print('check 16x8: max |new score - old header score| =', float(np.abs(old_py - py16).max()), '| number of windows', len(old_py), flush=True)
# INT8
path8 = f'models/quant_spectrum/3ch_int8_seed{SEED}.tflite'; thr8 = S['int8']['thr']; q8, py8, par8 = run(path8); lab = pick.label.to_numpy(); pred = py8 >= thr8
print('INT8 3 channels: input and output scale/zero point', par8, '| Python TP', int((pred & (lab == 1)).sum()), 'FP', int((pred & (lab == 0)).sum()), 'FN', int((~pred & (lab == 1)).sum()), 'TN', int((~pred & (lab == 0)).sum()))
blob = open(path8, 'rb').read()
with open(os.path.join(dest, 'phasenet_3ch_int8_model.h'), 'w') as f:
    f.write(f'// Dibuat oleh 165_make_fw_3ch_int8.py dari {path8}\n#pragma once\n\nalignas(16) const unsigned char g_model_3ch_16x8[] = {{\n' + mk.c_array(blob, 12, lambda b: f'0x{b:02x}') + f'\n}};\nconst unsigned int g_model_3ch_16x8_len = {len(blob)};\n')
labs = ', '.join(str(int(v)) for v in lab)
with open(os.path.join(dest, 'hw_eval_3ch_100_int8.h'), 'w') as f:
    f.write('// Dibuat oleh 165_make_fw_3ch_int8.py (INT8 penuh tiga kanal, seed 42)\n#pragma once\n#include <stdint.h>\n#define HW_3CH_N 100\n#define HW_3CH_WINDOW 3072\n#define HW_3CH_CHANNELS 3\n')
    f.write(f'static const float HW_3CH_THRESHOLD = {mk.c_float(thr8)};\nstatic const int HW_3CH_LABELS[HW_3CH_N] = {{{labs}}};\n')
    f.write('static const float HW_3CH_PY_SCORES[HW_3CH_N] = {' + ', '.join(mk.c_float(v) for v in py8) + '};\n')
    f.write('static const int8_t HW_3CH_INPUTS[HW_3CH_N][HW_3CH_WINDOW * HW_3CH_CHANNELS] = {\n')
    for k in range(len(q8)): f.write('  {' + ', '.join(str(int(v)) for v in q8[k].reshape(-1)) + '},\n')
    f.write('};\n')
# three-channel 16x8 header from the SAME procedure (so that INT8 and 16x8 use identical windows; the old header used a different procedure)
thr16 = json.load(open('results/final/quant16x8_pndw_per_seed.json'))[SEED]['thr_16x8']
with open(os.path.join(dest, 'hw_eval_3ch_100.h'), 'w') as f:
    f.write('// Dibuat oleh 165_make_fw_3ch_int8.py (16x8 tiga kanal, seed 42; jendela sama dengan varian INT8)\n#pragma once\n#include <stdint.h>\n#define HW_3CH_N 100\n#define HW_3CH_WINDOW 3072\n#define HW_3CH_CHANNELS 3\n')
    f.write(f'static const float HW_3CH_THRESHOLD = {mk.c_float(thr16)};\nstatic const int HW_3CH_LABELS[HW_3CH_N] = {{{labs}}};\n')
    f.write('static const float HW_3CH_PY_SCORES[HW_3CH_N] = {' + ', '.join(mk.c_float(v) for v in py16) + '};\n')
    f.write('static const int16_t HW_3CH_INPUTS[HW_3CH_N][HW_3CH_WINDOW * HW_3CH_CHANNELS] = {\n')
    for k in range(len(q16)): f.write('  {' + ', '.join(str(int(v)) for v in q16[k].reshape(-1)) + '},\n')
    f.write('};\n')
pred16 = py16 >= thr16; print('16x8 3 kanal: Python TP', int((pred16 & (lab == 1)).sum()), 'FP', int((pred16 & (lab == 0)).sum()), 'FN', int((~pred16 & (lab == 1)).sum()), 'TN', int((~pred16 & (lab == 0)).sum()), flush=True)
json.dump(dict(par16=par16, par8=par8, thr8=float(thr8), cek_16x8_maks_selisih=float(np.abs(old_py - py16).max())), open(f'results/final/fw_input_3ch_int8_seed{SEED}_sesi.json' if SEED != '42' else 'results/final/fw_input_3ch_int8_sesi.json', 'w'), indent=1)
print('done', flush=True)
