"""
Step 161 (answer to the advisor, viable 4-bit quantization): W4A8 with PER-CHANNEL 4-bit weights (not per tensor as in 159_), three seeds, two model families.
The scheme was FIXED BEFORE seeing the test results (pre-registration): main = 'pcmse' (symmetric -7..7 per output channel, the scale chosen per channel with the minimum squared error over 36 clipping candidates 0.30..1.00 times the maximum absolute value).
The other variants are reported as complements: 'pc' (per channel without clipping optimization) and 'pcmse_mix' (pcmse, but the first layer and the output layer prob_output stay 8-bit per channel). The naive per-tensor scheme of 159_ ('w4a8') is kept as a reference.
Channel axis: depthwise_kernel (k, in, 1) -> axis 1; pointwise_kernel and Conv1D kernel (k, in, out) -> axis 2; Conv1DTranspose kernel (k, out, in) -> axis 1. Bias and BN are not changed.
Implementation: the weights rounded to 4 bits are put back into the Keras model and then converted to full TFLite INT8 (8-bit activations; the same as 159_). The INT8 conversion remaps the weights to the per-channel int8 scale, so there is a small extra rounding noise relative to the 4-bit step
(about 3% of a step). This emulation is valid for ACCURACY; the 4-bit latency and energy are NOT measured (there is no 4-bit kernel). Packed size = INT8 file minus floor(number of kernel weights / 2) bytes (two weights per byte; per-channel scales and biases stay).
Usage: python 161_w4a8_perkanal_3seed.py [1ch|3ch]. Output: results/final/w4a8_perkanal_3seed.json
"""
import os, sys, json, gc
import numpy as np
import pandas as pd
import tensorflow as tf
import importlib
import config as C
import data as D
import data3ch as D3

ev = importlib.import_module('04_evaluate')
SEEDS = [42, 43, 44]; N_CALIB = 300; OUT = 'results/final/w4a8_perkanal_3seed.json'; MD = 'models/quant_spectrum'; os.makedirs(MD, exist_ok=True)
SET_NAMES = ['stead_val', 'stead_test', 'instance_test', 'ethz_test', 'txed_test']
FAM = {'1ch': dict(K=1, sep='models/svariants/s10_w1_seed{sd}.keras'), '3ch': dict(K=3, sep='models/ablation_final/pndw_seed{sd}.keras')}
VARIAN = os.environ.get('VARIAN', 'pc,pcmse,pcmse_mix').split(',')   # 'pcmse_a16' = complement after the one-channel result: 4-bit pcmse weights with 16-bit activations (16x8 conversion); not part of the pre-registration
CAND = np.linspace(0.30, 1.00, 36)


def q4(w, axis, mse):
    w = np.asarray(w, dtype=np.float64); wm = np.moveaxis(w, axis, 0); sh = wm.shape; flat = wm.reshape(sh[0], -1); out = np.empty_like(flat)
    for c in range(sh[0]):
        v = flat[c]; mx = float(np.max(np.abs(v)))
        if mx == 0: out[c] = v; continue
        best, best_err = mx, None
        for f in (CAND if mse else [1.0]):
            clip = mx * f; sc = clip / 7.0; qv = np.clip(np.round(v / sc), -7, 7) * sc; err = float(np.sum((v - qv) ** 2))
            if best_err is None or err < best_err: best, best_err = clip, err
        sc = best / 7.0; out[c] = np.clip(np.round(v / sc), -7, 7) * sc
    return np.moveaxis(out.reshape(sh), 0, axis).astype(np.float32)


def q8(w, axis):
    w = np.asarray(w, dtype=np.float64); wm = np.moveaxis(w, axis, 0); sh = wm.shape; flat = wm.reshape(sh[0], -1); out = np.empty_like(flat)
    for c in range(sh[0]):
        v = flat[c]; mx = float(np.max(np.abs(v))); sc = mx / 127.0 if mx > 0 else 1.0; out[c] = np.clip(np.round(v / sc), -127, 127) * sc
    return np.moveaxis(out.reshape(sh), 0, axis).astype(np.float32)


def snap_model(path, varian):
    m = tf.keras.models.load_model(path, compile=False); mse = varian != 'pc'; nw = 0
    kernel_layers = [l for l in m.layers if any(('kernel' in w.name) for w in l.weights)]
    skip8 = {kernel_layers[0].name, 'prob_output'} if varian == 'pcmse_mix' else set()
    for l in m.layers:
        ws = l.get_weights()
        if not ws: continue
        new = []
        for var, w in zip(l.weights, ws):
            n = var.name
            if w.ndim >= 2 and 'kernel' in n:
                axis = 1 if ('depthwise_kernel' in n or isinstance(l, tf.keras.layers.Conv1DTranspose)) else 2
                new.append(q8(w, axis) if l.name in skip8 else q4(w, axis, mse))
                if l.name not in skip8: nw += w.size
            else: new.append(w)
        l.set_weights(new)
    return m, nw


def tflite_int8(model, K, gen):
    conc = tf.function(lambda x: model(x)).get_concrete_function(tf.TensorSpec([1, C.WINDOW, K], tf.float32))
    cv = tf.lite.TFLiteConverter.from_concrete_functions([conc], model)
    cv.optimizations = [tf.lite.Optimize.DEFAULT]; cv.representative_dataset = gen
    cv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; cv.inference_input_type = tf.int8; cv.inference_output_type = tf.int8
    return cv.convert()


def tflite_16x8(model, K, gen):
    conc = tf.function(lambda x: model(x)).get_concrete_function(tf.TensorSpec([1, C.WINDOW, K], tf.float32))
    cv = tf.lite.TFLiteConverter.from_concrete_functions([conc], model)
    cv.optimizations = [tf.lite.Optimize.DEFAULT]; cv.representative_dataset = gen
    cv.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]; cv.inference_input_type = tf.int16; cv.inference_output_type = tf.int16
    return cv.convert()


def run_family(name, cfg, res):
    K = cfg['K']
    if K == 1:
        stead, sz = D.load_stead(); inst, iz = D.load_instance(); ethz, ez = D.load_ethz(); txz = np.load(os.path.join(C.DATA_DIR, 'txed_z.npy'), mmap_mode='r'); win = lambda m, z: D.eval_windows(m, z); norm = lambda seg: D.zscore(seg)
    else:
        stead, sz = D3.load_stead3(); inst, iz = D3.load_instance3(); ethz, ez = D3.load_ethz3(); txz = np.load(os.path.join(C.DATA_DIR, 'txed_enz.npy'), mmap_mode='r'); win = lambda m, z: D3.eval_windows3(m, z, prefilter_input=True); norm = lambda seg: D3.zscore3(seg)
    txm = pd.read_csv(os.path.join(C.DATA_DIR, 'txed_meta.csv'), low_memory=False)
    dropped = json.load(open(os.path.join(C.RESULTS_DIR, 'metrics.json')))['dropped_flat_windows']
    SETS = {'stead_val': (stead[stead.split == 'val'], sz), 'stead_test': (stead[stead.split == 'test'], sz), 'instance_test': (inst, iz), 'ethz_test': (ethz, ez), 'txed_test': (txm, txz)}
    for n in SET_NAMES[:4]:
        m, z = SETS[n]; SETS[n] = (m[~m.trace_name.isin(dropped[n])], z)
    train = stead[stead.split == 'train'].sample(n=N_CALIB, random_state=C.SEED)

    def make_gen():
        rng = np.random.default_rng(C.SEED)
        def gen():
            for i, row in train.iterrows():
                s0 = D.random_start(rng, row.label, row.p_arrival_sample, C.STEAD_NPTS); yield [norm(sz[i, s0:s0 + C.WINDOW]).reshape(1, C.WINDOW, K)]
        return gen

    R = res.setdefault(name, {}); paths = {}; info = {}
    for sd in SEEDS:
        for v in VARIAN:
            m, nw = snap_model(cfg['sep'].format(sd=sd), 'pcmse' if v == 'pcmse_a16' else v); blob = (tflite_16x8 if v == 'pcmse_a16' else tflite_int8)(m, K, make_gen()); p = f'{MD}/{name}_w4a8_{v}_seed{sd}.tflite'; open(p, 'wb').write(blob)
            paths[(sd, v)] = p; info[(sd, v)] = dict(bytes_int8=len(blob), n_bobot_4bit=int(nw), bytes_packed=int(len(blob) - nw // 2))
            del m; tf.keras.backend.clear_session(); gc.collect()
        print(name, 'seed', sd, 'dikonversi', {v: info[(sd, v)]['bytes_packed'] for v in VARIAN}, flush=True)
    scores = {k: {} for k in paths}; labels = {}
    for n in SET_NAMES:
        meta, z = SETS[n]; X, y = win(meta, z); labels[n] = y
        for k, p in paths.items(): scores[k][n] = ev.scores_tflite(p, X)
        del X; gc.collect(); print(name, 'set', n, 'dinilai', flush=True)
    for (sd, v), sc in scores.items():
        thr = ev.pick_threshold(labels['stead_val'], sc['stead_val']); r = R.setdefault(str(sd), {}).setdefault(v, {}); r.update(info[(sd, v)]); r['thr'] = float(thr); r['kualitas'] = {}
        for n in SET_NAMES[1:]:
            mt = ev.metrics(labels[n], (sc[n] >= thr).astype(int)); r['kualitas'][n] = {k: float(mt[k]) for k in ('recall', 'f1', 'fpr')}
    del sz; gc.collect()


if __name__ == '__main__':
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in sys.argv[1:] or ['1ch', '3ch']:
        run_family(name, FAM[name], res); json.dump(res, open(OUT, 'w'), indent=1); print('tersimpan', name, flush=True)
    print('DONE', flush=True)
