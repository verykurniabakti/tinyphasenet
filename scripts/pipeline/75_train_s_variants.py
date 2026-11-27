"""
Step 75: choice of the S label/loss treatment for TinyPhaseNet, chosen ONLY through the STEAD validation (P and S picks on the val earthquake windows; no INSTANCE/ETHZ).
Variants: S label sigma (samples; 10 = 0.1 s currently) and the weight of the S channel in the cross-entropy (1 = currently). The learning-rate recipe is the same as 74_ (cosine, 25 epochs).
Usage: python3 75_train_s_variants.py <sigma_S> <w_S> <seed> [quick_test_steps]
Output: models/svariants/s<sigma>_w<w>_seed<N>.keras, results/svariants/val_s<sigma>_w<w>_seed<N>.json (Recall@0.1/0.5 s and residual sd of P and S on val)
"""
import os, sys, math, json
import numpy as np
import tensorflow as tf
import config as C
import data as D
from data import load_stead, eval_windows, TrainSequence, gaussian, rel
from model import create_tinyphasenet

sigS, wS, seed = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
smoke = int(sys.argv[4]) if len(sys.argv) > 4 else 0
tag = f's{sigS:g}_w{wS:g}_seed{seed}'
OUTM = os.path.join(C.MODEL_DIR, 'svariants'); OUTR = os.path.join(C.RESULTS_DIR, 'svariants')
os.makedirs(OUTM, exist_ok=True); os.makedirs(OUTR, exist_ok=True)


def targets_s(p_rel, s_rel):
    p, s = gaussian(p_rel), gaussian(s_rel, sigma=sigS)
    return np.stack([p, s, np.clip(1.0 - p - s, 0.0, 1.0)], axis=-1)


class Seq(TrainSequence):
    def __getitem__(self, b):
        rows = self.meta.iloc[self.order[b * self.batch_size:(b + 1) * self.batch_size]]
        X = np.empty((len(rows), C.WINDOW, 1), dtype=np.float32); Y = np.empty((len(rows), C.WINDOW, 3), dtype=np.float32)
        for k, row in enumerate(rows.itertuples()):
            st = D.random_start(self.rng, row.label, row.p_arrival_sample, C.STEAD_NPTS)
            X[k, :, 0] = D.zscore(self.z[row.index, st:st + C.WINDOW])
            Y[k] = targets_s(rel(row.p_arrival_sample, st), rel(row.s_arrival_sample, st))
        return X, Y


W = tf.constant([1.0, wS, 1.0])
def weighted_ce(y, p):
    return -tf.reduce_mean(tf.reduce_sum(W * y * tf.math.log(tf.clip_by_value(p, 1e-7, 1.0)), axis=-1))


tf.keras.utils.set_random_seed(seed)
meta, z = load_stead()
seq = Seq(meta[meta.split == 'train'], z, seed=seed)
val_meta = meta[meta.split == 'val']
X_val, _, Y_val = eval_windows(val_meta, z, with_targets=True, prefilter_input=False)
Yv_s = np.stack([Y_val[:, :, 0], np.zeros_like(Y_val[:, :, 0]), Y_val[:, :, 2]], -1)  # nao usado
model = create_tinyphasenet()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=weighted_ce)
EP = 25
def lr_fn(ep):
    if ep == 0: return 1e-4
    t = (ep - 1) / (EP - 2)
    return 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * t))
cbs = [tf.keras.callbacks.LearningRateScheduler(lr_fn, verbose=0),
       tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTM, tag + '.keras'), monitor='val_loss', save_best_only=True, verbose=0),
       tf.keras.callbacks.CSVLogger(os.path.join(OUTR, f'train_history_{tag}.csv'))]
if smoke:
    model.fit(seq, validation_data=(X_val[:256], Y_val[:256]), epochs=1, steps_per_epoch=smoke, verbose=2)
else:
    model.fit(seq, validation_data=(X_val, Y_val), epochs=EP, callbacks=cbs, verbose=2)
    model = tf.keras.models.load_model(os.path.join(OUTM, tag + '.keras'), custom_objects={'weighted_ce': weighted_ce})

# metrica de picking na VALIDACAO STEAD: janelas de terremoto com S dentro da janela; centro verdadeiro = argmax do alvo gaussiano
prob = model.predict(X_val, batch_size=256, verbose=0)
eq = (val_meta.label.to_numpy() == 1)
res = {}
for ch, nm in [(0, 'P'), (1, 'S')]:
    true_c = np.argmax(Y_val[:, :, ch], axis=1); has = Y_val[:, :, ch].max(axis=1) > 0.99
    m = eq & has
    err = np.abs(np.argmax(prob[m, :, ch], axis=1) - true_c[m]) / C.FS
    core = err[err < 0.5]
    res[nm] = dict(n=int(m.sum()), recall_0p1=float((err < 0.1).mean()), recall_0p5=float((err < 0.5).mean()), mae_core=float(core.mean()) if len(core) else None)
res['val_loss_plain'] = float(-np.mean(np.sum(Y_val * np.log(np.clip(prob, 1e-7, 1)), axis=-1)))
print(json.dumps(res), flush=True)
if not smoke:
    json.dump(res, open(os.path.join(OUTR, f'val_{tag}.json'), 'w'), indent=2)
print('DONE', flush=True)
