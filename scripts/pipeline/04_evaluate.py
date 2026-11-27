"""
Step 4: evaluation.
- TinyPhaseNet input: the full trace is causally band-passed at 1-45 Hz (equivalent to the STEAD preprocessing,
  config.PREFILTER_HZ) before the window is cut and z-scored. STA/LTA uses the raw trace
  (no filter) or its own band-pass (config.BANDPASS_HZ, 2-8 Hz, chosen on STEAD val).
- Detection score per window: TinyPhaseNet = maximum P probability in the window;
  STA/LTA = maximum STA/LTA ratio in the window.
- STA/LTA is computed on the FULL TRACE (60 s STEAD, 120 s INSTANCE and ETHZ) so that the LTA is already stable
  before the window; LTA/STA are initialized with the mean energy of the first 10 s of the trace
  (no labels), and only the part inside the window is scored.
- The threshold of each method is chosen on the STEAD 'val' split (maximum F1) and then LOCKED for
  STEAD test, INSTANCE, and ETHZ (cross-domain data are not used to tune anything).
- Test metrics with a 95% confidence interval (bootstrap, 1000 resamples).
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import butter, sosfilt, lfilter
from sklearn.metrics import precision_recall_curve
import config as C
from data import load_stead, load_instance, load_ethz, eval_windows

N_BOOT = 1000


# ---------- TinyPhaseNet ----------
def scores_keras(model, X):
    return model.predict(X, batch_size=256, verbose=0)[:, :, 0].max(axis=1)


def scores_tflite(path, X):
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    in_s, in_zp = inp['quantization']
    out_s, out_zp = out['quantization']
    info = np.iinfo(inp['dtype'])                  # int8 (full INT8) or int16 (16x8)
    s = np.empty(len(X), dtype=np.float32)
    for k in range(len(X)):
        q = np.clip(np.round(X[k:k + 1] / in_s) + in_zp, info.min, info.max).astype(inp['dtype'])
        it.set_tensor(inp['index'], q)
        it.invoke()
        o = (it.get_tensor(out['index']).astype(np.float32) - out_zp) * out_s
        s[k] = o[0, :, 0].max()
    return s


# ---------- STA/LTA ----------
def recursive_energy(a2, n, init):
    c = 1.0 / n
    return lfilter([c], [1.0, -(1.0 - c)], a2, axis=1, zi=(init * (1.0 - c))[:, None])[0]


def scores_stalta(meta, z, bandpass, chunk=2000):
    nsta, nlta = int(C.STA_S * C.FS), int(C.LTA_S * C.FS)
    sos = butter(C.BANDPASS_ORDER, C.BANDPASS_HZ, btype='band', fs=C.FS, output='sos')
    rows = meta.index.to_numpy()
    starts = meta.eval_start.to_numpy()
    out = np.empty(len(rows), dtype=np.float32)
    for a in range(0, len(rows), chunk):
        tr = np.asarray(z[rows[a:a + chunk]], dtype=np.float64)
        tr = tr - tr.mean(axis=1, keepdims=True)
        if bandpass:
            tr = sosfilt(sos, tr, axis=1)          # causal, as on the device
        a2 = tr ** 2
        init = a2[:, :nlta].mean(axis=1)
        sta = recursive_energy(a2, nsta, init)
        lta = recursive_energy(a2, nlta, init)
        cft = sta / np.maximum(lta, 1e-12)
        for k, st in enumerate(starts[a:a + chunk]):
            out[a + k] = cft[k, st:st + C.WINDOW].max()
    return out


# ---------- Metrik ----------
def pick_threshold(y, s):
    """Maximum-F1 threshold on val; decision rule: score >= threshold (the same as precision_recall_curve)."""
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[np.argmax(f1)])


def metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                accuracy=(tp + tn) / len(y), fpr=fp / (fp + tn) if fp + tn else 0.0,
                tp=tp, fp=fp, tn=tn, fn=fn)


def with_ci(y, pred, rng):
    m = metrics(y, pred)
    boot = [metrics(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))]
    for k in ['precision', 'recall', 'f1', 'accuracy', 'fpr']:
        v = np.array([b[k] for b in boot])
        m[k + '_ci95'] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return m


if __name__ == '__main__':
    rng = np.random.default_rng(C.SEED)
    stead, sz = load_stead()
    inst, iz = load_instance()
    ethz, ez = load_ethz()
    sets = {'stead_val': (stead[stead.split == 'val'], sz),
            'stead_test': (stead[stead.split == 'test'], sz),
            'instance_test': (inst, iz),
            'ethz_test': (ethz, ez)}

    # Label-free quality control: drop flat windows (std = 0), e.g. INSTANCE traces
    # 20130204T120446.MN.CUC..HH whose Z channel is all zeros. Applies to all methods.
    dropped = {}
    for name, (meta, z) in sets.items():
        flat = [i for i, st in zip(meta.index, meta.eval_start) if np.std(z[i, st:st + C.WINDOW]) == 0]
        dropped[name] = meta.loc[flat, 'trace_name'].tolist()
        sets[name] = (meta.drop(index=flat), z)
        if flat:
            print(f"[{name}] {len(flat)} flat windows removed: {dropped[name]}")

    keras_model = tf.keras.models.load_model(os.path.join(C.MODEL_DIR, 'tinyphasenet_best.keras'))

    scores, labels = {}, {}
    for name, (meta, z) in sets.items():
        print(f"[{name}] {len(meta)} jendela")
        X, y = eval_windows(meta, z)
        labels[name] = y
        scores[(name, 'TinyPhaseNet Float32')] = scores_keras(keras_model, X)
        scores[(name, 'TinyPhaseNet INT8')] = scores_tflite(os.path.join(C.MODEL_DIR, 'tinyphasenet_int8.tflite'), X)
        scores[(name, 'TinyPhaseNet 16x8')] = scores_tflite(os.path.join(C.MODEL_DIR, 'tinyphasenet_16x8.tflite'), X)
        scores[(name, 'STA/LTA')] = scores_stalta(meta, z, bandpass=False)
        scores[(name, 'STA/LTA + bandpass')] = scores_stalta(meta, z, bandpass=True)

    methods = sorted({m for _, m in scores})
    thresholds = {m: pick_threshold(labels['stead_val'], scores[('stead_val', m)]) for m in methods}

    results = {'thresholds_from_stead_val': thresholds, 'dropped_flat_windows': dropped,
               'n_windows': {name: int(len(labels[name])) for name in labels}, 'test': {}}
    rows = []
    for name in ['stead_val', 'stead_test', 'instance_test', 'ethz_test']:
        for m in methods:
            pred = (scores[(name, m)] >= thresholds[m]).astype(int)
            r = with_ci(labels[name], pred, rng)
            results['test'].setdefault(name, {})[m] = r
            rows.append(dict(set=name, method=m, threshold=thresholds[m],
                             **{k: v for k, v in r.items() if not k.endswith('_ci95')}))

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(C.RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RESULTS_DIR, 'metrics.csv'), index=False)
    np.savez_compressed(os.path.join(C.RESULTS_DIR, 'scores.npz'),
                        **{f"{s}|{m}": v for (s, m), v in scores.items()},
                        **{f"{s}|label": v for s, v in labels.items()})
    pd.set_option('display.width', 200)
    print(df[['set', 'method', 'threshold', 'precision', 'recall', 'f1', 'accuracy', 'fpr']].round(4))
    print("SELESAI.")
