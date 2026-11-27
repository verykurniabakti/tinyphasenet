"""
Step 199 (companion compression manuscript, stage 2.1): post-training quantization of the STANDARD-CONVOLUTION MODEL (16x8, full INT8, per-channel W4A8 emulation with MSE clipping), one channel and three channels, seeds 42-44.
Goal: close the limitation "there is no quantized standard model" and test whether depthwise separable is more fragile to quantization than standard convolution (difference against Float32 of the SAME model, paired per seed).
The same protocol as the depthwise counterpart: conversion and calibration from 161_ (300 STEAD training windows, seed C.SEED; symmetric per-channel weights; per-tensor activations), the maximum-F1 threshold from the STEAD validation for each model and format (04_evaluate.pick_threshold),
flat windows dropped (metrics.json), evaluation on STEAD test, INSTANCE, ETHZ, TXED; P and S picking the same as 197_ (Recall@0.1 s and @0.5 s). Standard models: one channel models/ablation_final/standard_seed<N>.keras, three channels pn_seed<N>.keras.
Size = bytes of the TFLite file; activation-arena lower bound = the function arena_lower_bound of 159_. W4A8 and W4A16 = accuracy emulation (4-bit pcmse weights put back into Keras and then converted: full INT8 for W4A8, 16x8 for W4A16); no speed claim. W4A16 is added so that the weight-activation ladder of the standard model is parallel to the depthwise model.
Output: models/quant_padrao/*.tflite, results/final/ptq_padrao_3seed.json; per-set curves in results/final/_curvas_ptq_padrao/ (cache, skipped if present).
Usage (from this folder): python 199_ptq_padrao.py [1ch|3ch] [--limit N]   (--limit: first N windows per set, quick test; does not write the final result)
"""
import os, sys, json, gc, time, importlib, subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')   # TFLite berjalan di CPU; mencegah TensorFlow mengambil memori GPU Metal (memori terpadu)
import config as C
import data as D
import data3ch as D3

ev = importlib.import_module('04_evaluate'); q161 = importlib.import_module('161_w4a8_perkanal_3seed'); q159 = importlib.import_module('159_spektrum_kuantisasi_3seed')
FAMS = [a for a in sys.argv[1:] if a in ('1ch', '3ch')] or ['1ch', '3ch']
BATAS = int(sys.argv[sys.argv.index('--batas') + 1]) if '--batas' in sys.argv else None
SEEDS = [42, 43, 44]; N_CALIB = 300; SET_NAMES = ['stead_val', 'stead_test', 'instance_test', 'ethz_test', 'txed_test']; FMTS = ['q16x8', 'int8', 'w4a8', 'w4a16']
MD = 'models/quant_padrao'; CD = 'results/final/_curvas_ptq_padrao'; os.makedirs(MD, exist_ok=True); os.makedirs(CD, exist_ok=True)
OUT = 'results/final/ptq_padrao_3seed.json' if BATAS is None else '/tmp/ptq_padrao_uji.json'
STD = {'1ch': (1, 'models/ablation_final/standard_seed{sd}.keras'), '3ch': (3, 'models/ablation_final/pn_seed{sd}.keras')}


CHUNK = int(os.environ.get('CHUNK_199', 1000)); SWAP_NAIK_MAX_MB = 2000   # windows per chunk (three-channel X of 1000 windows = 37 MB); stop if the swap grows by more than this since the start of the process


def swap_used_mb(): return float(subprocess.check_output(['sysctl', '-n', 'vm.swapusage']).decode().split('used =')[1].split('M')[0])


SWAP0 = swap_used_mb()


def swap_guard(where):
    if swap_used_mb() - SWAP0 > SWAP_NAIK_MAX_MB: print(f'[PENJAGA] swap naik {swap_used_mb() - SWAP0:.0f} MB since the start at {where}; stopping (the cache of finished chunks is kept)', flush=True); sys.exit(2)


class Lazy:
    def __init__(self, path): self.path = path
    def __getitem__(self, k): return np.array(np.load(self.path, mmap_mode='r')[k])


def make_interp(path):
    it = tf.lite.Interpreter(model_path=path, num_threads=4); it.allocate_tensors(); return it


def curves_tflite(it, X):
    inp, out = it.get_input_details()[0], it.get_output_details()[0]; in_s, in_zp = inp['quantization']; out_s, out_zp = out['quantization']; info = np.iinfo(inp['dtype'])
    pm = np.empty(len(X), np.float32); pp = np.empty(len(X), np.int32); sp = np.empty(len(X), np.int32)
    for k in range(len(X)):
        q = np.clip(np.round(X[k:k + 1] / in_s) + in_zp, info.min, info.max).astype(inp['dtype']); it.set_tensor(inp['index'], q); it.invoke()
        o = (it.get_tensor(out['index']).astype(np.float32) - out_zp) * out_s; pm[k] = o[0, :, 0].max(); pp[k] = o[0, :, 0].argmax(); sp[k] = o[0, :, 1].argmax()
    return pm, pp, sp


def summ(res, n_total):   # idem 197_
    r = np.asarray(res); core = r[np.abs(r) < 0.5]
    return dict(n_detected=int(len(r)), n_total=int(n_total), recall_0p1=float((np.abs(r) < 0.1).sum() / n_total), recall_0p5=float((np.abs(r) < 0.5).sum() / n_total),
                mean_ms=float(1000 * core.mean()) if len(core) else None, std_ms=float(1000 * core.std()) if len(core) else None)


def avalia(pm, pp, sp, y, meta, thr, n):   # idem 197_
    fired = pm >= thr; eq = (y == 1); p_true = (meta.p_arrival_sample - meta.eval_start).to_numpy(); mP = eq & fired
    d = {'P': summ((pp[mP] - p_true[mP]) / C.FS, int(eq.sum()))}
    if 's_arrival_sample' in meta.columns and n in ('stead_test', 'txed_test'):
        s_true = (meta.s_arrival_sample - meta.eval_start).to_numpy(); sin = eq & np.isfinite(s_true) & (s_true >= 0) & (s_true < C.WINDOW); mS = fired & sin
        d['S'] = summ((sp[mS] - s_true[mS]) / C.FS, int(sin.sum()))
    return d


def build(fam, sd, K, train, sz, norm):
    """Produces three TFLite files (16x8, INT8, W4A8) for the standard model of seed sd; existing ones are skipped. Returns the path and size info."""
    path = STD[fam][1].format(sd=sd); paths, info = {}, {}
    def gen_fn():
        rng = np.random.default_rng(C.SEED)
        def gen():
            for i, row in train.iterrows():
                s0 = D.random_start(rng, row.label, row.p_arrival_sample, C.STEAD_NPTS); yield [norm(sz[i, s0:s0 + C.WINDOW]).reshape(1, C.WINDOW, K)]
        return gen
    for fmt in FMTS:
        p = f'{MD}/{fam}_standard_{fmt}_seed{sd}.tflite'; paths[fmt] = p
        if not os.path.exists(p):
            if fmt in ('w4a8', 'w4a16'): m, nw = q161.snap_model(path, 'pcmse'); blob = (q161.tflite_16x8 if fmt == 'w4a16' else q161.tflite_int8)(m, K, gen_fn())   # w4a16: 4-bit pcmse weights, 16-bit activations (16x8 conversion), as the pcmse_a16 variant in 161_
            else: m = tf.keras.models.load_model(path, compile=False); blob = (q161.tflite_16x8 if fmt == 'q16x8' else q161.tflite_int8)(m, K, gen_fn()); nw = 0
            open(p, 'wb').write(blob); del m; tf.keras.backend.clear_session(); gc.collect()
        blob = open(p, 'rb').read(); info[fmt] = dict(bytes=len(blob), arena_lb=int(q159.arena_lower_bound(blob)))
    return paths, info


res = json.load(open(OUT)) if (os.path.exists(OUT) and BATAS is None) else {}
dropped = json.load(open(os.path.join(C.RESULTS_DIR, 'metrics.json')))['dropped_flat_windows']
for fam in FAMS:
    K = STD[fam][0]
    if K == 1:
        stead, sz = D.load_stead(); inst, iz = D.load_instance(); ethz, ez = D.load_ethz(); txz = np.load(os.path.join(C.DATA_DIR, 'txed_z.npy'), mmap_mode='r'); win = lambda m, z: D.eval_windows(m, z); norm = lambda seg: D.zscore(seg)
    else:
        stead, sz = D3.load_stead3(); inst, iz = D3.load_instance3(); ethz, ez = D3.load_ethz3(); txz = np.load(os.path.join(C.DATA_DIR, 'txed_enz.npy'), mmap_mode='r'); win = lambda m, z: D3.eval_windows3(m, z, prefilter_input=True); norm = lambda seg: D3.zscore3(seg)
    txm = pd.read_csv(os.path.join(C.DATA_DIR, 'txed_meta.csv'), low_memory=False)
    sfx = 'z' if K == 1 else 'enz'   # per-trace reader: opens the memory map, copies a small slice, then releases it, so the file pages already read do not pile up in the process memory (np.load mmap holds up to the whole file, e.g. 2.9 GB)
    sz, iz, ez, txz = (Lazy(os.path.join(C.DATA_DIR, f'{n}_{sfx}.npy')) for n in ('stead', 'instance', 'ethz', 'txed'))
    SETS = {'stead_val': (stead[stead.split == 'val'], sz), 'stead_test': (stead[stead.split == 'test'], sz), 'instance_test': (inst, iz), 'ethz_test': (ethz, ez), 'txed_test': (txm, txz)}
    for n in SET_NAMES[:4]:
        m, z = SETS[n]; SETS[n] = (m[~m.trace_name.isin(dropped[n])], z)
    train = stead[stead.split == 'train'].sample(n=N_CALIB, random_state=C.SEED)
    paths, info = {}, {}
    for sd in SEEDS:
        paths[sd], info[sd] = build(fam, sd, K, train, sz, norm); print(fam, 'seed', sd, 'dikonversi', {f: (v['bytes'], v['arena_lb']) for f, v in info[sd].items()}, flush=True)
    metas, labels, curves = {}, {}, {}
    for n in SET_NAMES:   # one set at a time (saves memory for three channels)
        meta, z = SETS[n]
        if BATAS: meta = meta.iloc[:BATAS]
        metas[n] = meta; labels[n] = meta.label.to_numpy(); t0 = time.time(); todo = []   # labels the same as the output of eval_windows (meta.label); X does not need to be loaded if all the curves are already cached
        for sd in SEEDS:
            for fmt in FMTS:
                cf = f'{CD}/{fam}_{fmt}_seed{sd}_{n}.npz'
                if BATAS is None and os.path.exists(cf): z_ = np.load(cf); curves[(sd, fmt, n)] = (z_['pm'], z_['pp'], z_['sp'])
                else: todo.append((sd, fmt, cf))
        if not todo: continue
        its = {(sd, fmt): make_interp(paths[sd][fmt]) for sd, fmt, _ in todo}; acc = {k: ([], [], []) for k in its}   # one interpreter per model, reused for all chunks
        for a in range(0, len(meta), CHUNK):
            swap_guard(f'{fam} {n} potongan {a}'); X, _ = win(meta.iloc[a:a + CHUNK], z)
            for k, it in its.items():
                for j, v in enumerate(curves_tflite(it, X)): acc[k][j].append(v)
            del X
            if (a // CHUNK) % 10 == 0: print(f'{fam} {n[:8]:8s} {min(a + CHUNK, len(meta))}/{len(meta)} jendela ({time.time() - t0:.0f}s)', flush=True)
        for sd, fmt, cf in todo:
            pm, pp, sp = (np.concatenate(v) for v in acc[(sd, fmt)]); curves[(sd, fmt, n)] = (pm, pp, sp)
            if BATAS is None: np.savez(cf, pm=pm, pp=pp, sp=sp)
        del its, acc; gc.collect(); print(f'{fam} {n[:8]:8s} selesai ({time.time() - t0:.0f}s)', flush=True)
    R = res.setdefault(fam, {})
    for sd in SEEDS:
        for fmt in FMTS:
            thr = ev.pick_threshold(labels['stead_val'], curves[(sd, fmt, 'stead_val')][0]); r = R.setdefault(fmt, {}).setdefault(str(sd), {}); r.update(info[sd][fmt]); r['thr'] = float(thr); r['kualitas'] = {}; r['picking'] = {}
            for n in SET_NAMES[1:]:
                pm, pp, sp = curves[(sd, fmt, n)]; mt = ev.metrics(labels[n], (pm >= thr).astype(int)); r['kualitas'][n] = {k: float(mt[k]) for k in ('recall', 'f1', 'fpr')}
                r['picking'][n] = avalia(pm, pp, sp, labels[n], metas[n], thr, n)
            print(fam, fmt, 'seed', sd, 'F1', {n[:6]: round(100 * r['kualitas'][n]['f1'], 2) for n in SET_NAMES[1:]}, flush=True)
    json.dump(res, open(OUT, 'w'), indent=1); del sz; gc.collect(); print('tersimpan', fam, flush=True)
print('DONE', flush=True)
