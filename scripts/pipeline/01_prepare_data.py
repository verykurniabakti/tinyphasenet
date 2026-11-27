"""
Step 1: prepare the data.
- STEAD: 100k local earthquakes (manual P & S picks, distance <= 100 km, M 1-5) + 100k noise,
  split 80:10:10 per category (fixed seed). The Z channel of the full trace (6000 samples) is saved
  to data/stead_z.npy; metadata + split + evaluation-window position to data/stead_meta.csv.
- INSTANCE (cross-domain test): 10k events + 10k noise, velocimeters only (HH/EH).
  The Z channel of the full trace (12000 samples) to data/instance_z.npy, metadata to data/instance_meta.csv.

The evaluation-window position (val/test/INSTANCE) is drawn ONCE here and then locked, so that
all methods (TinyPhaseNet Float32/INT8, STA/LTA) are scored on exactly the same windows.
"""
import os
import h5py
import numpy as np
import pandas as pd
import config as C


def eval_window_start(rng, kind, p_sample, npts):
    """Window start: event -> P falls in [P_POS_MIN, P_POS_MAX] from the window start;
        noise -> random position along the trace."""
    if kind == 1:
        hi = int(min(C.P_POS_MAX, p_sample))
        u = rng.integers(C.P_POS_MIN, hi + 1)
        start = int(p_sample) - u
    else:
        start = rng.integers(0, npts - C.WINDOW + 1)
    assert 0 <= start <= npts - C.WINDOW
    return int(start)


def prepare_stead(rng):
    print("[STEAD] membaca metadata...")
    cols = ['trace_name', 'trace_category', 'p_status', 's_status', 'source_distance_km',
            'source_magnitude', 'p_arrival_sample', 's_arrival_sample']
    df = pd.read_csv(C.STEAD_CSV, usecols=cols, low_memory=False)

    eq = df[(df.trace_category == 'earthquake_local') & (df.p_status == 'manual') &
            (df.s_status == 'manual') & (df.source_distance_km <= 100.0) &
            (df.source_magnitude >= 1.0) & (df.source_magnitude <= 5.0) &
            df.p_arrival_sample.notna() & df.s_arrival_sample.notna()]
    noise = df[df.trace_category == 'noise']
    print(f"[STEAD] kandidat: {len(eq)} earthquakes, {len(noise)} noise")
    eq = eq.sample(n=C.N_EQ, random_state=C.SEED)
    noise = noise.sample(n=C.N_NOISE, random_state=C.SEED)

    def assign_split(part):
        n = len(part)
        n_tr, n_va = int(n * C.SPLIT[0]), int(n * C.SPLIT[1])
        labels = np.array(['train'] * n_tr + ['val'] * n_va + ['test'] * (n - n_tr - n_va))
        part = part.copy()
        part['split'] = rng.permutation(labels)
        return part

    meta = pd.concat([assign_split(eq), assign_split(noise)])
    meta['label'] = (meta.trace_category == 'earthquake_local').astype(int)
    meta = meta.sample(frac=1, random_state=C.SEED).reset_index(drop=True)
    meta['eval_start'] = -1
    for i in np.where(meta.split != 'train')[0]:
        meta.at[i, 'eval_start'] = eval_window_start(rng, meta.label[i], meta.p_arrival_sample[i], C.STEAD_NPTS)

    out = np.lib.format.open_memmap(os.path.join(C.DATA_DIR, 'stead_z.npy'), mode='w+',
                                    dtype=np.float32, shape=(len(meta), C.STEAD_NPTS))
    print(f"[STEAD] mengekstrak kanal Z {len(meta)} trace...")
    with h5py.File(C.STEAD_HDF5, 'r') as hf:
        g = hf['data']
        for i, name in enumerate(meta.trace_name):
            w = g[name][:]
            assert w.shape == (C.STEAD_NPTS, 3), (name, w.shape)
            out[i] = w[:, 2]                  # STEAD: columns E, N, Z
            if (i + 1) % 20000 == 0:
                print(f"  {i + 1}/{len(meta)}")
    out.flush()
    meta.to_csv(os.path.join(C.DATA_DIR, 'stead_meta.csv'), index=False)
    print(meta.groupby(['split', 'label']).size())


def instance_z_index(group, df, n_check=50):
    """Make sure of the Z channel index by matching the RMS of each channel to the trace_Z_rms_counts metadata."""
    hits = []
    for _, row in df.head(n_check).iterrows():
        w = group[row.trace_name][:].astype(np.float64)
        rms = np.sqrt(np.mean((w - w.mean(axis=1, keepdims=True)) ** 2, axis=1))
        hits.append(int(np.argmin(np.abs(rms - row.trace_Z_rms_counts))))
    idx = int(np.bincount(hits).argmax())
    print(f"[INSTANCE] indeks kanal Z = {idx} (cocok {hits.count(idx)}/{len(hits)} trace)")
    return idx


def prepare_instance(rng):
    rms_cols = ['trace_E_rms_counts', 'trace_N_rms_counts', 'trace_Z_rms_counts']
    print("[INSTANCE] membaca metadata event...")
    eq = pd.read_csv(C.INSTANCE_EQ_CSV, low_memory=False,
                     usecols=['trace_name', 'station_channels', 'trace_P_arrival_sample'] + rms_cols)
    eq = eq[eq.station_channels.isin(C.VELOCIMETER_CHANNELS) & eq.trace_P_arrival_sample.notna() &
            (eq.trace_P_arrival_sample >= C.P_POS_MAX) &
            (eq.trace_P_arrival_sample - C.P_POS_MIN + C.WINDOW <= C.INSTANCE_NPTS)]
    eq = eq.sample(n=C.N_INSTANCE_EQ, random_state=C.SEED)
    print("[INSTANCE] membaca metadata noise...")
    no = pd.read_csv(C.INSTANCE_NOISE_CSV, low_memory=False,
                     usecols=['trace_name', 'station_channels'] + rms_cols)
    no = no[no.station_channels.isin(C.VELOCIMETER_CHANNELS)]
    no = no.sample(n=C.N_INSTANCE_NOISE, random_state=C.SEED)

    eq = eq.assign(label=1, p_arrival_sample=eq.trace_P_arrival_sample, source='events')
    no = no.assign(label=0, p_arrival_sample=np.nan, source='noise')
    meta = pd.concat([eq, no]).reset_index(drop=True)
    meta['eval_start'] = [eval_window_start(rng, l, p, C.INSTANCE_NPTS)
                          for l, p in zip(meta.label, meta.p_arrival_sample)]

    out = np.lib.format.open_memmap(os.path.join(C.DATA_DIR, 'instance_z.npy'), mode='w+',
                                    dtype=np.float32, shape=(len(meta), C.INSTANCE_NPTS))
    for source, path in [('events', C.INSTANCE_EQ_HDF5), ('noise', C.INSTANCE_NOISE_HDF5)]:
        with h5py.File(path, 'r') as hf:
            g = hf['data'] if 'data' in hf else hf
            part = meta[meta.source == source]
            z = instance_z_index(g, part)
            assert z == 2, "urutan kanal INSTANCE tidak sesuai asumsi E,N,Z"
            print(f"[INSTANCE] mengekstrak {len(part)} trace {source}...")
            for i, name in zip(part.index, part.trace_name):
                w = g[name][:]
                assert w.shape == (3, C.INSTANCE_NPTS), (name, w.shape)
                out[i] = w[z]
    out.flush()
    meta.drop(columns=rms_cols).to_csv(os.path.join(C.DATA_DIR, 'instance_meta.csv'), index=False)
    print(meta.groupby(['label', 'station_channels']).size())


if __name__ == '__main__':
    os.makedirs(C.DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(C.SEED)
    prepare_stead(rng)
    prepare_instance(rng)
    print("SELESAI.")
