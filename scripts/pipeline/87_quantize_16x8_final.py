"""
Step 87: 16x8 post-training quantization (int16 activations, int8 weights) for the 10 final TinyPhaseNet seeds, a calibration recipe identical to 03_convert_int8.py (300 random windows of the STEAD
train split, seed C.SEED). Each seed: conversion -> models/final_16x8/seed<N>.tflite, then window evaluation (STEAD test, INSTANCE, ETHZ, TXED) with the STEAD val threshold for Float32 and 16x8
respectively (protocol 81_), plus the DECISION agreement (fired Float32 vs 16x8) and the score difference. Usage: python 87_quantize_16x8_final.py [seed ...] (without arguments = 42..51;
seeds already in the JSON are skipped). Output: results/final/quant16x8_per_seed.json
"""
import os, sys, json, importlib
import numpy as np
import pandas as pd
import tensorflow as tf
import config as C
from data import load_stead, load_instance, load_ethz, eval_windows, zscore, random_start

ev = importlib.import_module('04_evaluate')
SEEDS = [int(a) for a in sys.argv[1:]] or list(range(42, 52))
N_CALIB = 300
OUT = 'results/final/quant16x8_per_seed.json'
stead, sz = load_stead(); inst, iz = load_instance(); ethz, ez = load_ethz()
txm = pd.read_csv(os.path.join(C.DATA_DIR, 'txed_meta.csv'), low_memory=False); txz = np.load(os.path.join(C.DATA_DIR, 'txed_z.npy'), mmap_mode='r')
sets = {'stead_val': (stead[stead.split == 'val'], sz), 'stead_test': (stead[stead.split == 'test'], sz), 'instance_test': (inst, iz), 'ethz_test': (ethz, ez), 'txed_test': (txm, txz)}
dropped = json.load(open(os.path.join(C.RESULTS_DIR, 'metrics.json')))['dropped_flat_windows']
for n in ['stead_val', 'stead_test', 'instance_test', 'ethz_test']:
    meta, z = sets[n]; sets[n] = (meta[~meta.trace_name.isin(dropped[n])], z)
W = {n: eval_windows(meta, z) for n, (meta, z) in sets.items()}
train = stead[stead.split == 'train'].sample(n=N_CALIB, random_state=C.SEED)
rng_boot = np.random.default_rng(C.SEED)


def convert(model, path):
    run_model = tf.function(lambda x: model(x))
    concrete = run_model.get_concrete_function(tf.TensorSpec([1, C.WINDOW, 1], tf.float32))
    rng = np.random.default_rng(C.SEED)

    def gen():
        for i, row in train.iterrows():
            st = random_start(rng, row.label, row.p_arrival_sample, C.STEAD_NPTS)
            yield [zscore(sz[i, st:st + C.WINDOW]).reshape(1, C.WINDOW, 1)]

    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    conv.inference_input_type = tf.int16
    conv.inference_output_type = tf.int16
    blob = conv.convert()
    open(path, 'wb').write(blob)
    return len(blob)


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res = {int(k): v for k, v in res.items()}
for s in SEEDS:
    if s in res:
        continue
    model = tf.keras.models.load_model(f'models/svariants/s10_w1_seed{s}.keras', compile=False)
    path = f'models/final_16x8/seed{s}.tflite'
    nbytes = convert(model, path)
    fp = {n: ev.scores_keras(model, X) for n, (X, y) in W.items()}
    q = {n: ev.scores_tflite(path, X) for n, (X, y) in W.items()}
    thr_f, thr_q = ev.pick_threshold(W['stead_val'][1], fp['stead_val']), ev.pick_threshold(W['stead_val'][1], q['stead_val'])
    r = {'bytes': int(nbytes), 'thr_fp32': float(thr_f), 'thr_16x8': float(thr_q)}
    for n in ['stead_test', 'instance_test', 'ethz_test', 'txed_test']:
        y = W[n][1]; ff, fq = fp[n] >= thr_f, q[n] >= thr_q
        mf, mq = ev.metrics(y, ff.astype(int)), ev.metrics(y, fq.astype(int))
        r[n] = dict(fp32={k: float(v) for k, v in mf.items() if isinstance(v, (int, float))}, q16x8={k: float(v) for k, v in mq.items() if isinstance(v, (int, float))},
                    decision_agreement=float((ff == fq).mean()), mean_abs_score_diff=float(np.abs(fp[n] - q[n]).mean()), max_abs_score_diff=float(np.abs(fp[n] - q[n]).max()))
    res[s] = r
    print(f"seed {s}: {nbytes/1024:.1f} KB | " + ' | '.join(f"{n[:6]} F1 {100*r[n]['fp32']['f1']:.2f}->{100*r[n]['q16x8']['f1']:.2f} agree {100*r[n]['decision_agreement']:.2f}" for n in ['stead_test', 'instance_test', 'ethz_test', 'txed_test']), flush=True)
    del model; tf.keras.backend.clear_session()
    json.dump(res, open(OUT, 'w'), indent=2)
print('DONE', flush=True)
