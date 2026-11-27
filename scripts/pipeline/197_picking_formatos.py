"""
Step 197 (companion compression manuscript): arrival-time (picking) accuracy of P and S for ALL models and formats: one channel and three channels; standard-convolution Float32, depthwise separable Float32, 16x8, full INT8,
W4A8 (emulation, per-channel MSE clipping); seeds 42, 43, 44; datasets STEAD test, INSTANCE, ETHZ, TXED. The same protocol as 84_picking_final.py (one-channel Float32, 10 seeds):
'fired' = maximum P probability >= the threshold chosen on the STEAD validation for that model and format; arrival time = sample of the maximum probability of the P (or S) channel;
Recall@0.1 s and @0.5 s over ALL earthquake windows (P) or windows with S inside the window (S, only STEAD and TXED); mean and standard deviation of the residual only from picks with |residual| < 0.5 s.
Locked evaluation windows (column eval_start, the same for one channel and three channels; three channels with the 1-45 Hz prefilter as in 159_).
Model sources: Keras (standard, pn, depthwise Float32) and TFLite (models/final_16x8, models/final_16x8_pndw, models/quant_spectrum/{fam}_int8_seed*, {fam}_w4a8_pcmse_seed*).
Thresholds: eval_standard.json / eval_pn.json ('threshold'), quant16x8_per_seed.json and quant16x8_pndw_per_seed.json ('thr_fp32', 'thr_16x8'), spektrum_kuantisasi_3seed.json (int8 'thr'), w4a8_perkanal_3seed.json (pcmse 'thr').
Output: results/final/picking_formatos_{1ch,3ch,1ch_3ch}.json (saved as each model finishes; existing ones are skipped); 198_ combines and builds the tables.
Usage (from this folder): python 197_picking_formatos.py [1ch|3ch] [--limit N]   (--limit: only the first N windows of each dataset, quick test; does not write the final result)
Needs: the external SSD mounted (STEAD, INSTANCE, ETHZ data).
"""
import os, sys, json, time, gc, importlib
import numpy as np
import pandas as pd
import tensorflow as tf
import config as C
import data as D
import data3ch as D3

FAMS = [a for a in sys.argv[1:] if a in ('1ch', '3ch')] or ['1ch', '3ch']
BATAS = int(sys.argv[sys.argv.index('--batas') + 1]) if '--batas' in sys.argv else None
SEEDS = [42, 43, 44]; SETS_T = ['stead_test', 'instance_test', 'ethz_test', 'txed_test']
OUT = f"results/final/picking_formatos_{'_'.join(FAMS)}.json" if BATAS is None else '/tmp/picking_formatos_uji.json'   # one file per family that is run (parallel processes do not compete for a file)
S = json.load(open('results/final/spektrum_kuantisasi_3seed.json')); Wj = json.load(open('results/final/w4a8_perkanal_3seed.json'))
Q = {'1ch': {int(k): v for k, v in json.load(open('results/final/quant16x8_per_seed.json')).items()}, '3ch': {int(k): v for k, v in json.load(open('results/final/quant16x8_pndw_per_seed.json')).items()}}
TH = lambda fn: {int(e['model'].split('seed')[1].split('.')[0]): e['threshold'] for e in json.load(open(fn))}
THR_STD = {'1ch': TH('results/ablation_final/eval_standard.json'), '3ch': TH('results/ablation_final/eval_pn.json')}
KERAS = {('1ch', 'fp32'): 'models/svariants/s10_w1_seed{s}.keras', ('1ch', 'standard'): 'models/ablation_final/standard_seed{s}.keras',
         ('3ch', 'fp32'): 'models/ablation_final/pndw_seed{s}.keras', ('3ch', 'standard'): 'models/ablation_final/pn_seed{s}.keras'}
TFL = {('1ch', 'q16x8'): 'models/final_16x8/seed{s}.tflite', ('3ch', 'q16x8'): 'models/final_16x8_pndw/seed{s}.tflite',
       ('1ch', 'int8'): 'models/quant_spectrum/1ch_int8_seed{s}.tflite', ('3ch', 'int8'): 'models/quant_spectrum/3ch_int8_seed{s}.tflite',
       ('1ch', 'w4a8'): 'models/quant_spectrum/1ch_w4a8_pcmse_seed{s}.tflite', ('3ch', 'w4a8'): 'models/quant_spectrum/3ch_w4a8_pcmse_seed{s}.tflite'}
KEYS = ['standard', 'fp32', 'q16x8', 'int8', 'w4a8']


def threshold(fam, key, s):
    if key == 'standard': return float(THR_STD[fam][s])
    if key == 'fp32': return float(Q[fam][s]['thr_fp32'])
    if key == 'q16x8': return float(Q[fam][s]['thr_16x8'])
    if key == 'int8': return float(S[fam][str(s)]['int8']['thr'])
    return float(Wj[fam][str(s)]['pcmse']['thr'])


def curves_keras(model, X):
    pm, pp, sp = [], [], []
    for i in range(0, len(X), 512):
        p = model(X[i:i + 512], training=False).numpy(); pm.append(p[:, :, 0].max(axis=1)); pp.append(p[:, :, 0].argmax(axis=1)); sp.append(p[:, :, 1].argmax(axis=1))
    return np.concatenate(pm), np.concatenate(pp), np.concatenate(sp)


def curves_tflite(path, X):
    it = tf.lite.Interpreter(model_path=path, num_threads=4); it.allocate_tensors()
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
    tp = int((eq & fired & (np.abs((pp - p_true) / C.FS) <= 0.5)).sum()); fp = int(((~eq) & fired).sum()); fn = int(eq.sum()) - tp
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0; d['P']['f1_0p5'] = float(2 * pr * rc / (pr + rc)) if pr + rc else 0.0
    if 's_arrival_sample' in meta.columns and n in ('stead_test', 'txed_test'):
        s_true = (meta.s_arrival_sample - meta.eval_start).to_numpy(); sin = eq & np.isfinite(s_true) & (s_true >= 0) & (s_true < C.WINDOW); mS = fired & sin
        d['S'] = summ((sp[mS] - s_true[mS]) / C.FS, int(sin.sum()))
    return d


res = json.load(open(OUT)) if (os.path.exists(OUT) and BATAS is None) else {}
dropped = json.load(open(os.path.join(C.RESULTS_DIR, 'metrics.json')))['dropped_flat_windows']
for fam in FAMS:
    if fam == '1ch':
        stead, sz = D.load_stead(); inst, iz = D.load_instance(); ethz, ez = D.load_ethz(); txz = np.load(os.path.join(C.DATA_DIR, 'txed_z.npy'), mmap_mode='r'); win = lambda m, z: D.eval_windows(m, z)
    else:
        stead, sz = D3.load_stead3(); inst, iz = D3.load_instance3(); ethz, ez = D3.load_ethz3(); txz = np.load(os.path.join(C.DATA_DIR, 'txed_enz.npy'), mmap_mode='r'); win = lambda m, z: D3.eval_windows3(m, z, prefilter_input=True)
    txm = pd.read_csv(os.path.join(C.DATA_DIR, 'txed_meta.csv'), low_memory=False)
    SETS = {'stead_test': (stead[stead.split == 'test'], sz), 'instance_test': (inst, iz), 'ethz_test': (ethz, ez), 'txed_test': (txm, txz)}
    for n in SETS_T[:3]:
        m, z = SETS[n]; SETS[n] = (m[~m.trace_name.isin(dropped[n])], z)
    for n in SETS_T:   # one dataset at a time (saves memory for three channels); all models and seeds are run on that window set
        meta, z = SETS[n]
        if BATAS: meta = meta.iloc[:BATAS]
        X, y = win(meta, z); t0 = time.time(); assert 'eval_start' in meta.columns
        for s in SEEDS:
            for key in KEYS:
                slot = res.setdefault(fam, {}).setdefault(key, {}).setdefault(str(s), {'threshold': threshold(fam, key, s)})
                if n in slot: continue
                if (fam, key) in KERAS:
                    mdl = tf.keras.models.load_model(KERAS[(fam, key)].format(s=s), compile=False); pm, pp, sp = curves_keras(mdl, X); del mdl; tf.keras.backend.clear_session()
                else:
                    pm, pp, sp = curves_tflite(TFL[(fam, key)].format(s=s), X)
                slot[n] = avalia(pm, pp, sp, y, meta, slot['threshold'], n)
                json.dump(res, open(OUT, 'w'), indent=1)
                print(f"{fam} {n[:6]:6s} {key:8s} seed {s} P R@0,5 {100 * slot[n]['P']['recall_0p5']:.1f}" + (f" S R@0,5 {100 * slot[n]['S']['recall_0p5']:.1f}" if 'S' in slot[n] else '') + f"  ({time.time() - t0:.0f}s)", flush=True)
        del X; gc.collect()
print('DONE', flush=True)
