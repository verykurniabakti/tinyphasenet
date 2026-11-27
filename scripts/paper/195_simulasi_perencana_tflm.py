"""
Step 195 (device manuscript, Q, arena tracing): simulation of the TensorFlow Lite Micro greedy memory planner (GreedyMemoryPlanner) on the Float32 TFLite graphs of the four models
(one channel and three channels; depthwise separable and standard convolution), to explain why the measured arena of the three-channel standard model (489.8 KB) is larger than that of its depthwise counterpart (399.7 KB)
although the activation lower bound is the same (384 KB).
Algorithm imitated (esp-tflite-micro, tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc and micro_allocation_info.cc): buffer = non-constant tensor (model input and operator outputs),
size rounded to 16 bytes; lifetime [created, last used] in operator indices (model input from 0; an operator output at that operator; last used at the consuming operator; the model output until the end),
overlap counted inclusively; buffers sorted in descending order of size with a stable sort on an array filled from the tail, so for equal sizes the buffer with the LARGER id is placed first;
each buffer is placed in the first gap (first-fit) among the buffers alive at the same time. Measured arena = head (planner result) + persistent tail (about 10 KB).
Verified result: for the three-channel standard model, seed 44, 39 of 39 simulated offsets are identical to the memory plan printed by the board (log results_hw/sesi_10nov_rencana_memori_3ch_standar_seed44.log).
Output: results/final/simulasi_perencana_tflm.json and a summary on screen.
Usage (from this folder): python 195_simulasi_perencana_tflm.py
"""
import json, os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf
import config as C

MODELS = {'1ch_depthwise': ('models/svariants/s10_w1_seed42.keras', 1, ('hw_tiga_seed.json', '1ch_fp32')),
          '1ch_standar': ('models/ablation_final/standard_seed42.keras', 1, ('hw_padrao_tiga_seed.json', '1ch')),
          '3ch_depthwise': ('models/ablation_final/pndw_seed42.keras', 3, ('hw_tiga_seed.json', '3ch_fp32')),
          '3ch_standar': ('models/ablation_final/pn_seed42.keras', 3, ('hw_padrao_tiga_seed.json', '3ch'))}


def tflite_float32(path, K):
    m = tf.keras.models.load_model(path, compile=False)
    conc = tf.function(lambda x: m(x)).get_concrete_function(tf.TensorSpec([1, C.WINDOW, K], tf.float32))
    return tf.lite.TFLiteConverter.from_concrete_functions([conc], m).convert()


def rencana(blob, seri_menurun=True):
    it = tf.lite.Interpreter(model_content=blob, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES); it.allocate_tensors()
    ops = it._get_ops_details(); det = {d['index']: d for d in it.get_tensor_details()}
    ins = {d['index'] for d in it.get_input_details()}; outs = {d['index'] for d in it.get_output_details()}
    first, last = {i: 0 for i in ins}, {}
    for k, o in enumerate(ops):
        for t in o['outputs']: first.setdefault(t, k)
        for t in o['inputs']:
            if t >= 0: last[t] = max(last.get(t, 0), k)
    for t in outs: last[t] = len(ops)
    bufs = []
    for t, f in first.items():
        d = det[t]; ukuran = (int(np.prod(d['shape'])) * np.dtype(d['dtype']).itemsize + 15) // 16 * 16
        bufs.append(dict(t=t, size=ukuran, f=f, l=last.get(t, f)))
    bufs.sort(key=lambda b: (-b['size'], -b['t'] if seri_menurun else b['t'])); ditempatkan = []; kepala = 0
    for b in bufs:
        ov = sorted((p['off'], p['off'] + p['size']) for p in ditempatkan if not (p['l'] < b['f'] or b['l'] < p['f']))
        off = 0
        for s, e in ov:
            if s - off >= b['size']: break
            off = max(off, e)
        b['off'] = off; ditempatkan.append(b); kepala = max(kepala, off + b['size'])
    batas_bawah = max(sum(b['size'] for b in bufs if b['f'] <= k <= b['l']) for k in range(len(ops) + 1))
    return len(ops), len(bufs), batas_bawah, kepala


out = {}
print(f"{'model':14s} {'operator':>8s} {'buffer':>6s} {'lower bound':>12s} {'simulated head':>16s} {'(ascending id)':>12s} {'measured':>9s} {'difference':>8s}")
for nm, (path, K, (jf, key)) in MODELS.items():
    blob = tflite_float32(path, K)
    n, nb, lb, kep = rencana(blob); _, _, _, kep_naik = rencana(blob, seri_menurun=False)
    h = json.load(open('results/final/' + jf)); ukur = (h[key]['rerata']['arena_kb'])
    out[nm] = dict(operator=n, buffer=nb, batas_bawah_kb=lb / 1024, kepala_simulasi_kb=kep / 1024, kepala_seri_naik_kb=kep_naik / 1024, arena_terukur_kb=ukur, selisih_kb=ukur - kep / 1024)
    print(f"{nm:14s} {n:8d} {nb:6d} {lb / 1024:10.1f} KB {kep / 1024:12.1f} KB {kep_naik / 1024:9.1f} KB {ukur:7.1f} KB {ukur - kep / 1024:6.1f} KB")
json.dump(out, open('results/final/simulasi_perencana_tflm.json', 'w'), indent=1)
