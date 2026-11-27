"""
Step 203 (companion compression manuscript): arrival-time (picking) accuracy of P and S for the W4A16 format on the depthwise separable model, one channel and three channels, seeds 42-44 (197_ does not include this format).
Protocol IDENTICAL to 197_ and 199_: 'fired' = maximum P probability >= the threshold chosen on the STEAD validation for that model (w4a8_perkanal_3seed.json, variant 'pcmse_a16', key 'thr'); arrival time = sample of the maximum probability of the P (or S) channel;
Recall@0.1 s and @0.5 s over all earthquake windows (P) or windows with S inside the window (S, only STEAD and TXED). Model: models/quant_spectrum/<fam>_w4a8_pcmse_a16_seed<N>.tflite (emulated 4-bit pcmse weights, 16-bit activations; 161_).
Memory saving: evaluation per chunk of 1000 windows, per-trace reader (without holding the memory map), one interpreter per model, process-memory guard (stops if > 4000 MB).
Check: the same script is run for the depthwise 16x8 model, seed 42, on STEAD test and the result must equal results/final/picking_formatos.json (197_).
Output: results/final/picking_w4a16_separable.json. Usage (from this folder): python 203_picking_w4a16_separable.py [1ch|3ch]
"""
import os, sys, json, gc, time, subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import config as C
import data as D
import data3ch as D3

FAMS = [a for a in sys.argv[1:] if a in ('1ch', '3ch')] or ['1ch', '3ch']
SEEDS = [42, 43, 44]; DS = ['stead_test', 'instance_test', 'ethz_test', 'txed_test']; CHUNK = 1000; RSS_MAX_MB = 4000; OUT = 'results/final/picking_w4a16_separable.json'
W = json.load(open('results/final/w4a8_perkanal_3seed.json')); PF = json.load(open('results/final/picking_formatos.json')); Q = {int(k): v for k, v in json.load(open('results/final/quant16x8_per_seed.json')).items()}


def rss_mb(): return float(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())]).decode()) / 1024


class Lazy:   # per-trace reader: opens the memory map, copies a slice, then releases it (RSS does not pile up)
    def __init__(self, path): self.path = path
    def __getitem__(self, k): return np.array(np.load(self.path, mmap_mode='r')[k])


def curves(it, X):
    inp, out = it.get_input_details()[0], it.get_output_details()[0]; in_s, in_zp = inp['quantization']; out_s, out_zp = out['quantization']; info = np.iinfo(inp['dtype'])
    pm = np.empty(len(X), np.float32); pp = np.empty(len(X), np.int32); sp = np.empty(len(X), np.int32)
    for k in range(len(X)):
        q = np.clip(np.round(X[k:k + 1] / in_s) + in_zp, info.min, info.max).astype(inp['dtype']); it.set_tensor(inp['index'], q); it.invoke()
        o = (it.get_tensor(out['index']).astype(np.float32) - out_zp) * out_s; pm[k] = o[0, :, 0].max(); pp[k] = o[0, :, 0].argmax(); sp[k] = o[0, :, 1].argmax()
    return pm, pp, sp


def summ(res, n_total):
    r = np.asarray(res); core = r[np.abs(r) < 0.5]
    return dict(n_detected=int(len(r)), n_total=int(n_total), recall_0p1=float((np.abs(r) < 0.1).sum() / n_total), recall_0p5=float((np.abs(r) < 0.5).sum() / n_total),
                mean_ms=float(1000 * core.mean()) if len(core) else None, std_ms=float(1000 * core.std()) if len(core) else None)


def avalia(pm, pp, sp, y, meta, thr, n):
    fired = pm >= thr; eq = (y == 1); p_true = (meta.p_arrival_sample - meta.eval_start).to_numpy(); mP = eq & fired
    d = {'P': summ((pp[mP] - p_true[mP]) / C.FS, int(eq.sum()))}
    if 's_arrival_sample' in meta.columns and n in ('stead_test', 'txed_test'):
        s_true = (meta.s_arrival_sample - meta.eval_start).to_numpy(); sin = eq & np.isfinite(s_true) & (s_true >= 0) & (s_true < C.WINDOW); mS = fired & sin
        d['S'] = summ((sp[mS] - s_true[mS]) / C.FS, int(sin.sum()))
    return d


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
dropped = json.load(open(os.path.join(C.RESULTS_DIR, 'metrics.json')))['dropped_flat_windows']
for fam in FAMS:
    K = 1 if fam == '1ch' else 3
    if K == 1: stead, _ = D.load_stead(); inst, _ = D.load_instance(); ethz, _ = D.load_ethz(); win = lambda m, z: D.eval_windows(m, z); sfx = 'z'
    else: stead, _ = D3.load_stead3(); inst, _ = D3.load_instance3(); ethz, _ = D3.load_ethz3(); win = lambda m, z: D3.eval_windows3(m, z, prefilter_input=True); sfx = 'enz'
    txm = pd.read_csv(os.path.join(C.DATA_DIR, 'txed_meta.csv'), low_memory=False)
    Z = {n: Lazy(os.path.join(C.DATA_DIR, f'{n}_{sfx}.npy')) for n in ('stead', 'instance', 'ethz', 'txed')}
    SETS = {'stead_test': (stead[stead.split == 'test'], Z['stead']), 'instance_test': (inst, Z['instance']), 'ethz_test': (ethz, Z['ethz']), 'txed_test': (txm, Z['txed'])}
    for n in DS[:3]:
        m, z = SETS[n]; SETS[n] = (m[~m.trace_name.isin(dropped[n])], z)
    # jobs: (tag, seed, file, threshold): W4A16 separable, three seeds; and, only to check the protocol, 16x8 separable seed 42 on STEAD test
    JOBS = [('w4a16', sd, f'models/quant_spectrum/{fam}_w4a8_pcmse_a16_seed{sd}.tflite', float(W[fam][str(sd)]['pcmse_a16']['thr'])) for sd in SEEDS]
    if fam == '1ch': JOBS.append(('verif_q16x8', 42, 'models/final_16x8/seed42.tflite', float(Q[42]['thr_16x8'])))
    for n in DS:
        meta, z = SETS[n]; t0 = time.time(); its = {}
        for tag, sd, path, thr in JOBS:
            if tag == 'verif_q16x8' and n != 'stead_test': continue
            if tag == 'w4a16' and n in res.get(fam, {}).get(str(sd), {}): continue
            it = tf.lite.Interpreter(model_path=path, num_threads=4); it.allocate_tensors(); its[(tag, sd)] = (it, thr)
        if not its: continue
        acc = {k: ([], [], []) for k in its}
        for a in range(0, len(meta), CHUNK):
            if rss_mb() > RSS_MAX_MB: print(f'[PENJAGA] memori proses {rss_mb():.0f} MB > {RSS_MAX_MB} MB; berhenti', flush=True); sys.exit(2)
            X, _ = win(meta.iloc[a:a + CHUNK], z)
            for k, (it, thr) in its.items():
                for j, v in enumerate(curves(it, X)): acc[k][j].append(v)
            del X
        y = meta.label.to_numpy()
        for (tag, sd), (it, thr) in its.items():
            pm, pp, sp = (np.concatenate(v) for v in acc[(tag, sd)]); d = avalia(pm, pp, sp, y, meta, thr, n)
            if tag == 'w4a16': res.setdefault(fam, {}).setdefault(str(sd), {'threshold': thr})[n] = d
            else:
                ref = PF[fam]['q16x8']['42'][n]['P']['recall_0p5']; print(f'VERIFIKASI {fam} 16x8 seed 42 {n}: Recall@0,5 s P baru {d["P"]["recall_0p5"]:.6f} compared with 197_ {ref:.6f} (selisih {abs(d["P"]["recall_0p5"] - ref):.2e})', flush=True)
        json.dump(res, open(OUT, 'w'), indent=1); del its, acc; gc.collect(); print(f'{fam} {n} selesai ({time.time() - t0:.0f}s, memori proses {rss_mb():.0f} MB)', flush=True)
print('DONE', flush=True)
