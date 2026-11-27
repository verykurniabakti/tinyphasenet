"""
Step 5: firmware headers for the two variants (full INT8 and 16x8).
- include/tinyphasenet_int8_model.h, include/tinyphasenet_16x8_model.h :
  the model as a const array (in Flash, not SRAM).
- include/test_windows.h : RAW windows (before the z-score) from the STEAD test split,
  labels, and the Python TFLite scores and the threshold (from the val split, 04_evaluate.py) per variant.
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import config as C
from data import load_stead, zscore

N_PER_CLASS = 10
FW = os.path.join(C.ROOT, 'firmware', 'tinyphasenet_profiling', 'include')
VARIANTS = {'int8': 'TinyPhaseNet INT8', '16x8': 'TinyPhaseNet 16x8'}


def c_array(values, per_line, fmt):
    items = [fmt(v) for v in values]
    return ',\n'.join('  ' + ', '.join(items[i:i + per_line]) for i in range(0, len(items), per_line))


def c_float(v):
    """A valid C float literal: '123' -> '123.0f', '1e+05' -> '1e+05f'."""
    t = f'{float(v):.9g}'
    return (t if any(ch in t for ch in '.en') else t + '.0') + 'f'


def tflite_scores(path, windows):
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    (in_s, in_zp), (out_s, out_zp) = inp['quantization'], out['quantization']
    info = np.iinfo(inp['dtype'])
    scores = []
    for w in windows:
        q = np.clip(np.round(zscore(w).reshape(1, C.WINDOW, 1) / in_s) + in_zp, info.min, info.max)
        it.set_tensor(inp['index'], q.astype(inp['dtype']))
        it.invoke()
        o = (it.get_tensor(out['index']).astype(np.float32) - out_zp) * out_s
        scores.append(float(o[0, :, 0].max()))
    return scores


if __name__ == '__main__':
    os.makedirs(FW, exist_ok=True)
    thresholds = json.load(open(os.path.join(C.RESULTS_DIR, 'metrics.json')))['thresholds_from_stead_val']

    meta, z = load_stead()
    test = meta[meta.split == 'test']
    pick = (pd.concat([test[test.label == c].sample(n=N_PER_CLASS, random_state=C.SEED) for c in (0, 1)])
              .sample(frac=1, random_state=C.SEED))
    raw = np.stack([np.asarray(z[i, st:st + C.WINDOW], dtype=np.float32)
                    for i, st in zip(pick.index, pick.eval_start)])
    assert all(np.std(w) > 0 for w in raw), "jendela datar terpilih"

    with open(os.path.join(FW, 'test_windows.h'), 'w') as f:
        f.write('#pragma once\n// Dibuat oleh 05_make_firmware_inputs.py: jendela mentah kanal Z dari split test STEAD\n')
        f.write('// trace: ' + ', '.join(pick.trace_name) + '\n')
        f.write(f'const int NUM_WINDOWS = {len(pick)};\nconst int WINDOW_LEN = {C.WINDOW};\n')
        f.write('const int TEST_LABELS[NUM_WINDOWS] = {' + ', '.join(str(v) for v in pick.label) + '};\n')

        for tag, method in VARIANTS.items():
            path = os.path.join(C.MODEL_DIR, f'tinyphasenet_{tag}.tflite')
            model = open(path, 'rb').read()
            sym = 'int8' if tag == 'int8' else '16x8'
            with open(os.path.join(FW, f'tinyphasenet_{tag}_model.h'), 'w') as m:
                m.write(f'#pragma once\n// Dibuat oleh 05_make_firmware_inputs.py dari models/tinyphasenet_{tag}.tflite\n')
                m.write('alignas(16) const unsigned char g_model[] = {\n')
                m.write(c_array(model, 12, lambda b: f'0x{b:02x}'))
                m.write(f'\n}};\nconst unsigned int g_model_len = {len(model)};\n')
            py = tflite_scores(path, raw)
            f.write(f'const float THRESHOLD_{sym.upper()} = {thresholds[method]:.8f}f;  // dari split val STEAD\n')
            f.write(f'const float PY_SCORES_{sym.upper()}[NUM_WINDOWS] = {{' + ', '.join(f'{v:.8f}f' for v in py) + '};\n')
            print(f"[{tag}] model {len(model)} bytes, threshold {thresholds[method]:.4f}")

        f.write('const float TEST_WINDOWS[NUM_WINDOWS][WINDOW_LEN] = {\n')
        f.write(',\n'.join('  {\n' + c_array(w, 10, c_float) + '\n  }' for w in raw))
        f.write('\n};\n')
    print(f"{len(pick)} test windows written to {FW}")
