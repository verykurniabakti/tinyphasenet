"""
Three-channel (E,N,Z) variant of data.py, for the ablation "is the Z component alone enough".
Split, window position, and labels use the SAME metadata as the one-channel model
(stead_meta.csv, instance_meta.csv, ethz_meta.csv) -- a fair head-to-head comparison.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import sosfilt
import config as C
from data import PREFILTER_SOS, targets, rel, random_start


def load_stead3():
    meta = pd.read_csv(os.path.join(C.DATA_DIR, 'stead_meta.csv'), low_memory=False)
    w = np.load(os.path.join(C.DATA_DIR, 'stead_enz.npy'), mmap_mode='r')
    return meta, w


def load_instance3():
    meta = pd.read_csv(os.path.join(C.DATA_DIR, 'instance_meta.csv'), low_memory=False)
    w = np.load(os.path.join(C.DATA_DIR, 'instance_enz.npy'), mmap_mode='r')
    return meta, w


def load_ethz3():
    meta = pd.read_csv(os.path.join(C.DATA_DIR, 'ethz_meta.csv'), low_memory=False)
    w = np.load(os.path.join(C.DATA_DIR, 'ethz_enz.npy'), mmap_mode='r')
    return meta, w


def zscore3(w):
    """Per-channel normalization (zero mean, unit std), equivalent to data.zscore for one channel."""
    w = w.astype(np.float32)
    return (w - w.mean(axis=0, keepdims=True)) / (w.std(axis=0, keepdims=True) + 1e-8)


def prefilter3(trace):
    """Causal 1-45 Hz band-pass per channel on the full trace (mean subtracted first)."""
    tr = np.asarray(trace, dtype=np.float64)
    tr = tr - tr.mean(axis=0, keepdims=True)
    return sosfilt(PREFILTER_SOS, tr, axis=0)


def eval_windows3(meta, w, with_targets=False, prefilter_input=True):
    """Locked evaluation windows (column eval_start, the SAME as the one-channel model) -> X (N,WINDOW,3), y, [Y]."""
    X = np.empty((len(meta), C.WINDOW, 3), dtype=np.float32)
    Y = np.empty((len(meta), C.WINDOW, 3), dtype=np.float32) if with_targets else None
    for k, (i, row) in enumerate(meta.iterrows()):
        st = int(row.eval_start)
        src = prefilter3(w[i]) if prefilter_input else w[i]
        X[k] = zscore3(src[st:st + C.WINDOW])
        if with_targets:
            Y[k] = targets(rel(row.p_arrival_sample, st), rel(row.get('s_arrival_sample', np.nan), st))
    y = meta.label.to_numpy()
    return (X, y, Y) if with_targets else (X, y)


class TrainSequence3(tf.keras.utils.Sequence):
    """Random three-channel training windows (position redrawn at every draw), earthquake + noise."""

    def __init__(self, meta, w, batch_size=C.BATCH_SIZE, seed=C.SEED):
        super().__init__()
        self.meta = meta.reset_index()        # column 'index' = row in w
        self.w = w
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(self.meta))

    def __len__(self):
        return len(self.meta) // self.batch_size

    def __getitem__(self, b):
        rows = self.meta.iloc[self.order[b * self.batch_size:(b + 1) * self.batch_size]]
        X = np.empty((len(rows), C.WINDOW, 3), dtype=np.float32)
        Y = np.empty((len(rows), C.WINDOW, 3), dtype=np.float32)
        for k, row in enumerate(rows.itertuples()):
            st = random_start(self.rng, row.label, row.p_arrival_sample, C.STEAD_NPTS)
            X[k] = zscore3(self.w[row.index, st:st + C.WINDOW])
            Y[k] = targets(rel(row.p_arrival_sample, st), rel(row.s_arrival_sample, st))
        return X, Y

    def on_epoch_end(self):
        self.order = self.rng.permutation(len(self.meta))
