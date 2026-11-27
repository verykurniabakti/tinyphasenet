"""
Step 90: evaluation of the ablation with the FINAL RECIPE (to compare with the 10 final seeds 81_/84_). One GROUP per process (saves memory; models are loaded one at a time):
  nofilter : the 10 final models evaluated on windows that are only z-scored (without the 1-45 Hz band-pass), WITHOUT retraining; threshold from the STEAD val without a filter.
  standard : plain Conv1D (76,195 params), seeds 42-44, filtered windows, STEAD val threshold.
  3ch      : input E,N,Z (31,280 params), seeds 42-44, filtered windows; P/S picking on STEAD test and TXED.
Sets: STEAD test, INSTANCE, ETHZ, TXED (flat-window QC the same as 81_). Filtered separable reference: results/final/metrics_per_seed.json.
Guard: stops cleanly if the swap used > SWAP_MAX_MB. Usage: python 90_eval_ablation_final.py <nofilter|standard|3ch|pn|pndw>. Output: results/ablation_final/eval_<group>.json
"""
import os, sys, gc, json, subprocess, importlib
import numpy as np
import pandas as pd
import tensorflow as tf
import config as C
import data as D
import data3ch as D3

ev = importlib.import_module('04_evaluate')
group = sys.argv[1]
SWAP_MAX_MB = 14000
CFG = {'nofilter': ([f'models/svariants/s10_w1_seed{s}.keras' for s in range(42, 52)], 1, False),
       'standard': ([f'models/ablation_final/standard_seed{s}.keras' for s in (42, 43, 44)], 1, True),
       '3ch': ([f'models/ablation_final/3ch_seed{s}.keras' for s in (42, 43, 44)], 3, True),
       'pn': ([f'models/ablation_final/pn_seed{s}.keras' for s in (42, 43, 44)], 3, True),
       'pndw': ([f'models/ablation_final/pndw_seed{s}.keras' for s in (42, 43, 44)], 3, True)}
paths, ch, pf = CFG[group]
paths = [p for p in paths if os.path.exists(p)]   # only the seeds that have been trained (the advisor's question: pilot seed 42, then 43 and 44)
SET_NAMES = ['stead_val', 'stead_test', 'instance_test', 'ethz_test', 'txed_test']


def swap_guard(where):
    used = float(subprocess.check_output(['sysctl', '-n', 'vm.swapusage']).decode().split('used =')[1].split('M')[0])
    if used > SWAP_MAX_MB:
        print(f'[PENJAGA] swap {used:.0f} MB > {SWAP_MAX_MB} MB di {where}; berhenti', flush=True)
        sys.exit(2)


rng = np.random.default_rng(C.SEED)
dropped = json.load(open(os.path.join(C.RESULTS_DIR, 'metrics.json')))['dropped_flat_windows']
if ch == 1:
    stead, sz = D.load_stead(); inst, iz = D.load_instance(); ethz, ez = D.load_ethz(); txz = np.load(os.path.join(C.DATA_DIR, 'txed_z.npy'), mmap_mode='r')
    src = {'stead': (stead, sz), 'instance': (inst, iz), 'ethz': (ethz, ez)}
else:
    stead, sz = D3.load_stead3(); inst, iz = D3.load_instance3(); ethz, ez = D3.load_ethz3(); txz = np.load(os.path.join(C.DATA_DIR, 'txed_enz.npy'), mmap_mode='r')
txm = pd.read_csv(os.path.join(C.DATA_DIR, 'txed_meta.csv'), low_memory=False)
SETS = {'stead_val': (stead[stead.split == 'val'], sz), 'stead_test': (stead[stead.split == 'test'], sz), 'instance_test': (inst, iz), 'ethz_test': (ethz, ez), 'txed_test': (txm, txz)}
for n in SET_NAMES[:4]:
    m, z = SETS[n]; SETS[n] = (m[~m.trace_name.isin(dropped[n])], z)


def windows(n):
    meta, z = SETS[n]
    return (D3.eval_windows3(meta, z, prefilter_input=pf)[0] if ch == 3 else D.eval_windows(meta, z, prefilter_input=pf)[0])


def curves(model, X):
    pr = np.concatenate([model(X[i:i + 256], training=False).numpy() for i in range(0, len(X), 256)])
    return pr[:, :, 0].max(axis=1), pr[:, :, 0].argmax(axis=1), pr[:, :, 1].argmax(axis=1)


raw = [{} for _ in paths]   # [model][set] = (pmax, p_argmax, s_argmax)
if ch == 1:
    # pattern of 81_: the windows of all sets stay in memory (1.2 GB), models are loaded ONE at a time (repeated load/delete per set leaks Metal GPU memory)
    W = {}
    for n in SET_NAMES:
        swap_guard('jendela ' + n); W[n] = windows(n)
    for k, p in enumerate(paths):
        cf = f'results/ablation_final/cache_{group}_{k}.npz'
        if os.path.exists(cf):   # can be resumed: the result per model is saved
            z_ = np.load(cf)
            for n in SET_NAMES:
                raw[k][n] = (z_[n + '|0'], z_[n + '|1'], z_[n + '|2'])
            print('model', k, 'from cache', flush=True); continue
        swap_guard(f'model {k}')
        model = tf.keras.models.load_model(p, compile=False)
        for n in SET_NAMES:
            raw[k][n] = curves(model, W[n])
        np.savez(cf, **{f'{n}|{i}': raw[k][n][i] for n in SET_NAMES for i in range(3)})
        del model; tf.keras.backend.clear_session(); gc.collect()
        print('model', k, 'done', flush=True)
else:
    # 3 channels: large windows (0.7 GB/set) -> per set; 3 models are loaded ONCE and kept
    models = [tf.keras.models.load_model(p, compile=False) for p in paths]
    for n in SET_NAMES:
        swap_guard('set ' + n)
        X = windows(n)
        for k, m in enumerate(models):
            raw[k][n] = curves(m, X)
        del X; gc.collect()
        print('set', n, 'done', flush=True)

out = []
y_of = {n: SETS[n][0].label.to_numpy() for n in SET_NAMES}
for k, r in enumerate(raw):
    thr = ev.pick_threshold(y_of['stead_val'], r['stead_val'][0]); d = {'model': paths[k], 'threshold': float(thr)}
    for n in SET_NAMES[1:]:
        res = ev.with_ci(y_of[n], (r[n][0] >= thr).astype(int), rng)
        d[n] = {kk: float(v) for kk, v in res.items() if isinstance(v, (int, float))}
        if group == '3ch' and n in ('stead_test', 'txed_test'):
            meta = SETS[n][0]; eq = y_of[n] == 1; fired = r[n][0] >= thr
            pt = (meta.p_arrival_sample - meta.eval_start).to_numpy(dtype=float); st = (meta.s_arrival_sample - meta.eval_start).to_numpy(dtype=float)
            sin = eq & np.isfinite(st) & (st >= 0) & (st < C.WINDOW)

            def summ(res_, nt):
                a = np.asarray(res_); return dict(n_total=int(nt), recall_0p1=float((np.abs(a) < 0.1).sum() / nt), recall_0p5=float((np.abs(a) < 0.5).sum() / nt))
            d[n]['pick_P'] = summ((r[n][1][eq & fired] - pt[eq & fired]) / C.FS, eq.sum())
            d[n]['pick_S'] = summ((r[n][2][fired & sin] - st[fired & sin]) / C.FS, sin.sum())
    out.append(d)
    print(group, k, {n: round(100 * d[n]['f1'], 2) for n in SET_NAMES[1:]}, flush=True)
json.dump(out, open(f'results/ablation_final/eval_{group}.json', 'w'), indent=1)
print('DONE', flush=True)
