"""
Window cutting, normalization, and labels. Shared by training, INT8 conversion,
evaluation, and firmware input generation.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import butter, sosfilt
import config as C

PREFILTER_SOS = butter(C.PREFILTER_ORDER, C.PREFILTER_HZ, btype='band', fs=C.FS, output='sos')


def load_stead():
    meta = pd.read_csv(os.path.join(C.DATA_DIR, 'stead_meta.csv'), low_memory=False)
    z = np.load(os.path.join(C.DATA_DIR, 'stead_z.npy'), mmap_mode='r')
    return meta, z


def load_instance():
    meta = pd.read_csv(os.path.join(C.DATA_DIR, 'instance_meta.csv'), low_memory=False)
    z = np.load(os.path.join(C.DATA_DIR, 'instance_z.npy'), mmap_mode='r')
    return meta, z


def load_ethz():
    meta = pd.read_csv(os.path.join(C.DATA_DIR, 'ethz_meta.csv'), low_memory=False)
    z = np.load(os.path.join(C.DATA_DIR, 'ethz_z.npy'), mmap_mode='r')
    return meta, z


def zscore(w):
    w = w.astype(np.float32)
    return (w - w.mean()) / (w.std() + 1e-8)


def gaussian(center, n=C.WINDOW, sigma=C.LABEL_SIGMA):
    if center is None or np.isnan(center) or center < 0 or center >= n:
        return np.zeros(n, dtype=np.float32)
    x = np.arange(n, dtype=np.float32)
    return np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


def targets(p_rel, s_rel):
    """Labels [P, S, Noise] per time sample; noise = 1 - P - S (clipped to [0, 1])."""
    p, s = gaussian(p_rel), gaussian(s_rel)
    return np.stack([p, s, np.clip(1.0 - p - s, 0.0, 1.0)], axis=-1)


def random_start(rng, label, p_sample, npts):
    """The same rule as 01_prepare_data.eval_window_start, redrawn every epoch."""
    if label == 1:
        u = rng.integers(C.P_POS_MIN, int(min(C.P_POS_MAX, p_sample)) + 1)
        return int(p_sample) - int(u)
    return int(rng.integers(0, npts - C.WINDOW + 1))


def rel(sample, start):
    return np.nan if pd.isna(sample) else float(sample) - start


def prefilter(trace):
    """Causal 1-45 Hz band-pass on the FULL trace (mean subtracted first), equivalent to the STEAD preprocessing."""
    tr = np.asarray(trace, dtype=np.float64)
    return sosfilt(PREFILTER_SOS, tr - tr.mean())


def eval_windows(meta, z, with_targets=False, prefilter_input=True):
    """Locked evaluation windows (column eval_start) -> X (N, WINDOW, 1), y, [Y].
        prefilter_input=True: the full trace is band-passed at 1-45 Hz before the window is cut and z-scored."""
    X = np.empty((len(meta), C.WINDOW, 1), dtype=np.float32)
    Y = np.empty((len(meta), C.WINDOW, 3), dtype=np.float32) if with_targets else None
    for k, (i, row) in enumerate(meta.iterrows()):
        st = int(row.eval_start)
        src = prefilter(z[i]) if prefilter_input else z[i]
        X[k, :, 0] = zscore(src[st:st + C.WINDOW])
        if with_targets:
            Y[k] = targets(rel(row.p_arrival_sample, st), rel(row.get('s_arrival_sample', np.nan), st))
    y = meta.label.to_numpy()
    return (X, y, Y) if with_targets else (X, y)


class TrainSequence(tf.keras.utils.Sequence):
    """Random training windows (position redrawn at every draw), earthquake + noise."""

    def __init__(self, meta, z, batch_size=C.BATCH_SIZE, seed=C.SEED):
        super().__init__()
        self.meta = meta.reset_index()   # column 'index' = row in z
        self.z = z
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(self.meta))

    def __len__(self):
        return len(self.meta) // self.batch_size

    def __getitem__(self, b):
        rows = self.meta.iloc[self.order[b * self.batch_size:(b + 1) * self.batch_size]]
        X = np.empty((len(rows), C.WINDOW, 1), dtype=np.float32)
        Y = np.empty((len(rows), C.WINDOW, 3), dtype=np.float32)
        for k, row in enumerate(rows.itertuples()):
            st = random_start(self.rng, row.label, row.p_arrival_sample, C.STEAD_NPTS)
            X[k, :, 0] = zscore(self.z[row.index, st:st + C.WINDOW])
            Y[k] = targets(rel(row.p_arrival_sample, st), rel(row.s_arrival_sample, st))
        return X, Y

    def on_epoch_end(self):
        self.order = self.rng.permutation(len(self.meta))
