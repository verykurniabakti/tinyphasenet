"""
Step 151 (answer to the advisor, three stages on THREE CHANNELS E,N,Z): 16x8 post-training quantization (int16 activations, int8 weights) for the 3-channel <variant> model, seeds 42-44 (models/ablation_final/<variant>_seed<N>.keras; variant 'pndw' = depthwise separable PhaseNet),
same calibration recipe as 87_quantize_16x8_final.py (300 random windows of the STEAD train split, seed C.SEED; windows cut and normalized as in the training of 89_). Window evaluation (STEAD test, INSTANCE, ETHZ, TXED) with the STEAD val threshold
for Float32 and 16x8 respectively (protocol 87_/90_, filtered windows), decision agreement and score difference. The 3-channel windows are large (0.7 GB per set): one set at a time, the model loaded once.
Usage: python 151_quantize_3ch_16x8.py [variant=pndw] [seed ...]. Output: models/final_16x8_<variant>/seed<N>.tflite, results/final/quant16x8_<variant>_per_seed.json
"""
import os, sys, json, gc, subprocess, importlib
import numpy as np
import pandas as pd
import tensorflow as tf
import config as C
import data as D
import data3ch as D3

ev = importlib.import_module('04_evaluate')
VAR = sys.argv[1] if len(sys.argv) > 1 else 'pndw'; SEEDS = [int(a) for a in sys.argv[2:]] or [42, 43, 44]; N_CALIB = 300; OUT = f'results/final/quant16x8_{VAR}_per_seed.json'; SWAP_MAX_MB = 14000
SET_NAMES = ['stead_val', 'stead_test', 'instance_test', 'ethz_test', 'txed_test']
os.makedirs(f'models/final_16x8_{VAR}', exist_ok=True)


def swap_guard(where):
    used = float(subprocess.check_output(['sysctl', '-n', 'vm.swapusage']).decode().split('used =')[1].split('M')[0])
    if used > SWAP_MAX_MB:
        print(f'[PENJAGA] swap {used:.0f} MB > {SWAP_MAX_MB} MB di {where}; berhenti', flush=True); sys.exit(2)


stead, sw = D3.load_stead3(); inst, iw = D3.load_instance3(); ethz, ew = D3.load_ethz3(); txz = np.load(os.path.join(C.DATA_DIR, 'txed_enz.npy'), mmap_mode='r')
txm = pd.read_csv(os.path.join(C.DATA_DIR, 'txed_meta.csv'), low_memory=False)
dropped = json.load(open(os.path.join(C.RESULTS_DIR, 'metrics.json')))['dropped_flat_windows']
SETS = {'stead_val': (stead[stead.split == 'val'], sw), 'stead_test': (stead[stead.split == 'test'], sw), 'instance_test': (inst, iw), 'ethz_test': (ethz, ew), 'txed_test': (txm, txz)}
for n in SET_NAMES[:4]:
    m, z = SETS[n]; SETS[n] = (m[~m.trace_name.isin(dropped[n])], z)
train = stead[stead.split == 'train'].sample(n=N_CALIB, random_state=C.SEED)


def convert(model, path):
    run_model = tf.function(lambda x: model(x))
    concrete = run_model.get_concrete_function(tf.TensorSpec([1, C.WINDOW, 3], tf.float32))
    rng = np.random.default_rng(C.SEED)

    def gen():
        for i, row in train.iterrows():
            st = D.random_start(rng, row.label, row.p_arrival_sample, C.STEAD_NPTS)
            yield [D3.zscore3(sw[i, st:st + C.WINDOW]).reshape(1, C.WINDOW, 3)]

    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    conv.inference_input_type = tf.int16
    conv.inference_output_type = tf.int16
    blob = conv.convert()
    open(path, 'wb').write(blob)
    return len(blob)


models, paths, nbytes = {}, {}, {}
for s in SEEDS:
    models[s] = tf.keras.models.load_model(f'models/ablation_final/{VAR}_seed{s}.keras', compile=False)
    paths[s] = f'models/final_16x8_{VAR}/seed{s}.tflite'; nbytes[s] = convert(models[s], paths[s])
    print(f'seed {s}: {nbytes[s]/1024:.1f} KB', flush=True)
fp = {s: {} for s in SEEDS}; q = {s: {} for s in SEEDS}; y_of = {n: SETS[n][0].label.to_numpy() for n in SET_NAMES}
for n in SET_NAMES:
    swap_guard('set ' + n)
    meta, z = SETS[n]; X, _ = D3.eval_windows3(meta, z, prefilter_input=True)
    for s in SEEDS:
        fp[s][n] = ev.scores_keras(models[s], X); q[s][n] = ev.scores_tflite(paths[s], X)
    del X; gc.collect(); print('set', n, 'done', flush=True)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res = {int(k): v for k, v in res.items()}
for s in SEEDS:
    thr_f, thr_q = ev.pick_threshold(y_of['stead_val'], fp[s]['stead_val']), ev.pick_threshold(y_of['stead_val'], q[s]['stead_val'])
    r = {'bytes': int(nbytes[s]), 'thr_fp32': float(thr_f), 'thr_16x8': float(thr_q)}
    for n in SET_NAMES[1:]:
        y = y_of[n]; ff, fq = fp[s][n] >= thr_f, q[s][n] >= thr_q
        mf, mq = ev.metrics(y, ff.astype(int)), ev.metrics(y, fq.astype(int))
        r[n] = dict(fp32={k: float(v) for k, v in mf.items() if isinstance(v, (int, float))}, q16x8={k: float(v) for k, v in mq.items() if isinstance(v, (int, float))},
                    decision_agreement=float((ff == fq).mean()), mean_abs_score_diff=float(np.abs(fp[s][n] - q[s][n]).mean()), max_abs_score_diff=float(np.abs(fp[s][n] - q[s][n]).max()))
    res[s] = r
    print(f"seed {s}: {nbytes[s]/1024:.1f} KB | " + ' | '.join(f"{n[:6]} F1 {100*r[n]['fp32']['f1']:.2f}->{100*r[n]['q16x8']['f1']:.2f} agree {100*r[n]['decision_agreement']:.2f}" for n in SET_NAMES[1:]), flush=True)
json.dump(res, open(OUT, 'w'), indent=2)
print('DONE', flush=True)
