"""
Step 89: architecture/input ablation with the FINAL TinyPhaseNet RECIPE (cosine 25 epochs, sigma_S = 10, w_S = 1; the same as 75_train_s_variants.py s10_w1) so that it is comparable with the 10 final seeds.
Variants: 'standard' = plain Conv1D (1 channel Z, 76,195 parameters); '3ch' = input E,N,Z (separable architecture, 31,280 parameters); 'pn' = the PhaseNet architecture in Keras (3 channels, plain Conv, 268,443 trained parameters, 269,675 in total with the untrained BN statistics; phasenet_keras.py) and 'pndw' = the same PhaseNet with depthwise separable convolution (108,152 trained, 109,384 in total), both with this final recipe (the advisor's question).
Usage: python3 89_train_ablation_final.py <standard|3ch|pn|pndw> <seed> [quick_test_steps]
Output: models/ablation_final/<variant>_seed<N>.keras, results/ablation_final/train_history_<variant>_seed<N>.csv
"""
import os, sys, math
import numpy as np
import tensorflow as tf
import config as C
import data as D
import data3ch as D3
from data import rel, gaussian
from model import create_tinyphasenet

variant, seed = sys.argv[1], int(sys.argv[2])
smoke = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SIG_S = 10.0
tag = f'{variant}_seed{seed}'
OUTM = os.path.join(C.MODEL_DIR, 'ablation_final'); OUTR = os.path.join(C.RESULTS_DIR, 'ablation_final')
os.makedirs(OUTM, exist_ok=True); os.makedirs(OUTR, exist_ok=True)


def targets_s(p_rel, s_rel):
    p, s = gaussian(p_rel), gaussian(s_rel, sigma=SIG_S)
    return np.stack([p, s, np.clip(1.0 - p - s, 0.0, 1.0)], axis=-1)


three = variant in ('3ch', 'pn', 'pndw')
base = D3.TrainSequence3 if three else D.TrainSequence
norm = D3.zscore3 if three else D.zscore
src_name = 'w' if three else 'z'


class Seq(base):
    def __getitem__(self, b):
        rows = self.meta.iloc[self.order[b * self.batch_size:(b + 1) * self.batch_size]]
        ch = 3 if three else 1
        X = np.empty((len(rows), C.WINDOW, ch), dtype=np.float32); Y = np.empty((len(rows), C.WINDOW, 3), dtype=np.float32)
        arr = getattr(self, src_name)
        for k, row in enumerate(rows.itertuples()):
            st = D.random_start(self.rng, row.label, row.p_arrival_sample, C.STEAD_NPTS)
            seg = arr[row.index, st:st + C.WINDOW]
            if three:
                X[k] = norm(seg)
            else:
                X[k, :, 0] = norm(seg)
            Y[k] = targets_s(rel(row.p_arrival_sample, st), rel(row.s_arrival_sample, st))
        return X, Y


tf.keras.utils.set_random_seed(seed)
if three:
    meta, z = D3.load_stead3(); X_val, _, Y_val = D3.eval_windows3(meta[meta.split == 'val'], z, with_targets=True, prefilter_input=False)
    if variant in ('pn', 'pndw'):
        from phasenet_keras import create_phasenet_keras
        model = create_phasenet_keras(conv='separable' if variant == 'pndw' else 'standard', name='PhaseNet-' + variant)
    else:
        model = create_tinyphasenet(input_channels=3, model_name='TinyPhaseNet-3ch')
else:
    meta, z = D.load_stead(); X_val, _, Y_val = D.eval_windows(meta[meta.split == 'val'], z, with_targets=True, prefilter_input=False)
    model = create_tinyphasenet(conv='standard', model_name='TinyPhaseNet-standard')
seq = Seq(meta[meta.split == 'train'], z, seed=seed)
print(variant, 'parameter', model.count_params(), 'train', len(seq) * C.BATCH_SIZE, 'val', len(X_val), flush=True)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy')
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
print('DONE', flush=True)
