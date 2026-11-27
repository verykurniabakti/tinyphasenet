"""
Step 159 (answer to the advisor, correction of the quantization-spectrum table): one script that produces ALL the numbers of the spectrum table with code that matches the claims of the manuscript, for seeds 42, 43, 44.
Model families: 1 channel (TinyPhaseNet: models/svariants/s10_w1_seed<N>.keras; standard: models/ablation_final/standard_seed<N>.keras) and 3 channels (depthwise PhaseNet: models/ablation_final/pndw_seed<N>.keras; retrained PhaseNet: pn_seed<N>.keras;
official PhaseNet through phasenet_keras.load_official_weights).
Stages: fp32 (TFLite float), 16x8 (files from 87_/151_), FULL INT8 (int8 weights and activations, int8 input and output, representative dataset of the same 300 STEAD training windows as 87_/151_), and W4A8 (simulation: the weights of tensors with ndim>=2
are mapped to 15 symmetric 4-bit levels per tensor, then converted to full TFLite INT8; 8-bit activations; there is no real 4-bit kernel, so the latency equals INT8 and the packed size is computed theoretically from the INT8 file).
Size: bytes of the TFLite file. Peak SRAM: lower bound of the activation arena from the tensor-lifetime analysis on the actual TFLite graph (constant tensors not counted; TFLM fragmentation/scratch not included).
Latency: protocol of 153_ (TFLite, 1 thread, 20 warm-ups + 300 invocations, 50 STEAD test windows; reference kernel without delegate and default configuration). INT8 and W4A8 quality: threshold from the STEAD validation of each model; flat windows dropped (protocol 90_);
fp32 and 16x8 quality is read from results/final/quant16x8_per_seed.json (1 channel) and quant16x8_pndw_per_seed.json (3 channels).
Usage: python 159_spektrum_kuantisasi_3seed.py. Output: results/final/spektrum_kuantisasi_3seed.json, models/quant_spectrum/*.tflite
"""
import os, sys, json, time, gc, platform, importlib
import numpy as np
import pandas as pd
import tensorflow as tf
import config as C
import data as D
import data3ch as D3
from phasenet_keras import create_phasenet_keras, load_official_weights

ev = importlib.import_module('04_evaluate')
SEEDS = [42, 43, 44]; N_CALIB = 300; N_WIN, N_WARM, N_REP = 50, 20, 300
OUT = 'results/final/spektrum_kuantisasi_3seed.json'; MD = 'models/quant_spectrum'; os.makedirs(MD, exist_ok=True)
SET_NAMES = ['stead_val', 'stead_test', 'instance_test', 'ethz_test', 'txed_test']
NODEL = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
FAM = {
    '1ch': dict(K=1, sep='models/svariants/s10_w1_seed{sd}.keras', other={'standard': 'models/ablation_final/standard_seed{sd}.keras'}, q16='models/final_16x8/seed{sd}.tflite', qjson='results/final/quant16x8_per_seed.json'),
    '3ch': dict(K=3, sep='models/ablation_final/pndw_seed{sd}.keras', other={'pn': 'models/ablation_final/pn_seed{sd}.keras'}, q16='models/final_16x8_pndw/seed{sd}.tflite', qjson='results/final/quant16x8_pndw_per_seed.json')}


def tflite_float(model, K):
    conc = tf.function(lambda x: model(x)).get_concrete_function(tf.TensorSpec([1, C.WINDOW, K], tf.float32))
    return tf.lite.TFLiteConverter.from_concrete_functions([conc], model).convert()


def tflite_int8(model, K, gen):
    conc = tf.function(lambda x: model(x)).get_concrete_function(tf.TensorSpec([1, C.WINDOW, K], tf.float32))
    cv = tf.lite.TFLiteConverter.from_concrete_functions([conc], model)
    cv.optimizations = [tf.lite.Optimize.DEFAULT]; cv.representative_dataset = gen
    cv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; cv.inference_input_type = tf.int8; cv.inference_output_type = tf.int8
    return cv.convert()


def snap4(path):
    m = tf.keras.models.load_model(path, compile=False); ws, nw = [], 0
    for w in m.get_weights():
        if w.ndim >= 2:
            mx = float(np.max(np.abs(w))); sc = mx / 7.0 if mx > 0 else 1.0; ws.append(np.clip(np.round(w / sc), -8, 7) * sc); nw += w.size
        else: ws.append(w)
    m.set_weights(ws); return m, nw


def arena_lower_bound(blob):
    it = tf.lite.Interpreter(model_content=blob, experimental_op_resolver_type=NODEL); it.allocate_tensors()
    ops = it._get_ops_details(); td = {t['index']: t for t in it.get_tensor_details()}
    first, last = {}, {}
    for n, op in enumerate(ops):
        for o in op['outputs']:
            if o >= 0: first.setdefault(int(o), n); last[int(o)] = max(last.get(int(o), n), n)
        for i in op['inputs']:
            if i >= 0: last[int(i)] = max(last.get(int(i), n), n)
    for d in it.get_input_details(): first[int(d['index'])] = 0; last.setdefault(int(d['index']), 0)
    for d in it.get_output_details(): last[int(d['index'])] = len(ops) - 1
    sz = {t: int(np.prod(td[t]['shape'])) * np.dtype(td[t]['dtype']).itemsize for t in first}
    return max(sum(sz[t] for t in first if first[t] <= n <= last[t]) for n in range(len(ops)))


def latency(blob, X, delegates):
    kw = dict(model_content=blob, num_threads=1)
    if not delegates: kw['experimental_op_resolver_type'] = NODEL
    it = tf.lite.Interpreter(**kw); it.allocate_tensors(); inp = it.get_input_details()[0]; s, zp = inp['quantization']
    info = np.iinfo(inp['dtype']) if np.issubdtype(inp['dtype'], np.integer) else None
    Q = [np.clip(np.round(X[k:k + 1] / s) + zp, info.min, info.max).astype(inp['dtype']) if info is not None else X[k:k + 1] for k in range(len(X))]
    t = []
    for r in range(N_WARM + N_REP):
        it.set_tensor(inp['index'], Q[r % len(Q)]); t0 = time.perf_counter_ns(); it.invoke(); dt = (time.perf_counter_ns() - t0) / 1e6
        if r >= N_WARM: t.append(dt)
    return dict(median=float(np.median(t)), q1=float(np.percentile(t, 25)), q3=float(np.percentile(t, 75)))


def profile(blob, X):
    return dict(bytes=len(blob), arena_lb=int(arena_lower_bound(blob)), lat_ref=latency(blob, X, False), lat_def=latency(blob, X, True))


def run_family(name, cfg, res):
    K = cfg['K']
    if K == 1:
        stead, sz = D.load_stead(); inst, iz = D.load_instance(); ethz, ez = D.load_ethz(); txz = np.load(os.path.join(C.DATA_DIR, 'txed_z.npy'), mmap_mode='r'); win = lambda m, z: D.eval_windows(m, z)
        norm = lambda seg: D.zscore(seg)
    else:
        stead, sz = D3.load_stead3(); inst, iz = D3.load_instance3(); ethz, ez = D3.load_ethz3(); txz = np.load(os.path.join(C.DATA_DIR, 'txed_enz.npy'), mmap_mode='r'); win = lambda m, z: D3.eval_windows3(m, z, prefilter_input=True)
        norm = lambda seg: D3.zscore3(seg)
    txm = pd.read_csv(os.path.join(C.DATA_DIR, 'txed_meta.csv'), low_memory=False)
    dropped = json.load(open(os.path.join(C.RESULTS_DIR, 'metrics.json')))['dropped_flat_windows']
    SETS = {'stead_val': (stead[stead.split == 'val'], sz), 'stead_test': (stead[stead.split == 'test'], sz), 'instance_test': (inst, iz), 'ethz_test': (ethz, ez), 'txed_test': (txm, txz)}
    for n in SET_NAMES[:4]:
        m, z = SETS[n]; SETS[n] = (m[~m.trace_name.isin(dropped[n])], z)
    train = stead[stead.split == 'train'].sample(n=N_CALIB, random_state=C.SEED)
    te = stead[stead.split == 'test'].sample(n=N_WIN, random_state=C.SEED); XL, _ = win(te, sz)

    def make_gen():
        rng = np.random.default_rng(C.SEED)
        def gen():
            for i, row in train.iterrows():
                s0 = D.random_start(rng, row.label, row.p_arrival_sample, C.STEAD_NPTS); yield [norm(sz[i, s0:s0 + C.WINDOW]).reshape(1, C.WINDOW, K)]
        return gen

    R = res.setdefault(name, {}); paths = {}
    for sd in SEEDS:
        r = R.setdefault(str(sd), {}); sp = cfg['sep'].format(sd=sd)
        m = tf.keras.models.load_model(sp, compile=False)
        r['fp32'] = profile(tflite_float(m, K), XL); r['fp32']['params'] = int(m.count_params())
        r['q16x8'] = profile(open(cfg['q16'].format(sd=sd), 'rb').read(), XL)
        b8 = tflite_int8(m, K, make_gen()); open(f'{MD}/{name}_int8_seed{sd}.tflite', 'wb').write(b8); r['int8'] = profile(b8, XL)
        m4, nw = snap4(sp); b4 = tflite_int8(m4, K, make_gen()); open(f'{MD}/{name}_w4a8_seed{sd}.tflite', 'wb').write(b4)
        r['w4a8'] = profile(b4, XL); r['w4a8']['bytes_packed_teoretis'] = int(len(b4) - 0.5 * nw); r['w4a8']['n_bobot_ndim_ge2'] = int(nw)
        for tag, p in cfg['other'].items():
            mo = tf.keras.models.load_model(p.format(sd=sd), compile=False); r[tag] = profile(tflite_float(mo, K), XL); r[tag]['params'] = int(mo.count_params()); del mo
        paths[sd] = {'int8': f'{MD}/{name}_int8_seed{sd}.tflite', 'w4a8': f'{MD}/{name}_w4a8_seed{sd}.tflite'}
        del m, m4; tf.keras.backend.clear_session(); gc.collect(); print(name, 'seed', sd, 'profile done', {k: (v['bytes'], v['arena_lb']) for k, v in r.items()}, flush=True)
    if K == 3:
        mo = load_official_weights(create_phasenet_keras(conv='standard')); R['resmi'] = profile(tflite_float(mo, K), XL); R['resmi']['params'] = int(mo.count_params()); del mo
    # INT8 and W4A8 quality
    scores = {sd: {t: {} for t in ('int8', 'w4a8')} for sd in SEEDS}; labels = {}
    for n in SET_NAMES:
        meta, z = SETS[n]; X, y = win(meta, z); labels[n] = y
        for sd in SEEDS:
            for t in ('int8', 'w4a8'): scores[sd][t][n] = ev.scores_tflite(paths[sd][t], X)
        del X; gc.collect(); print(name, 'set', n, 'dinilai', flush=True)
    for sd in SEEDS:
        r = R[str(sd)]
        for t in ('int8', 'w4a8'):
            thr = ev.pick_threshold(labels['stead_val'], scores[sd][t]['stead_val']); r[t]['thr'] = float(thr); r[t]['kualitas'] = {}
            for n in SET_NAMES[1:]:
                mt = ev.metrics(labels[n], (scores[sd][t][n] >= thr).astype(int)); r[t]['kualitas'][n] = {k: float(mt[k]) for k in ('recall', 'f1', 'fpr')}
    del sz; gc.collect()


if __name__ == '__main__':
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res['maquina'] = dict(cpu=os.popen('sysctl -n machdep.cpu.brand_string').read().strip(), os=platform.platform(), tf=tf.__version__, threads=1, n_jendela=N_WIN, n_pemanasan=N_WARM, n_ulang=N_REP)
    for name in sys.argv[1:] or ['1ch', '3ch']:
        run_family(name, FAM[name], res); json.dump(res, open(OUT, 'w'), indent=1); print('tersimpan', name, flush=True)
    print('DONE', flush=True)
