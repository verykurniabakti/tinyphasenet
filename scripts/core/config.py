"""
Single configuration of the TinyPhaseNet v2 experiment. Every script reads its constants from here
so that data, training, evaluation, and firmware use the same rules.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')        # large arrays (not committed)
RESULTS_DIR = os.path.join(ROOT, 'results')  # metrics, logs, thresholds
MODEL_DIR = os.path.join(ROOT, 'models')

# --- Data sources ---
STEAD_CSV = os.environ.get('STEAD_CSV', 'data_external/merge.csv')
STEAD_HDF5 = os.environ.get('STEAD_HDF5', 'data_external/merge.hdf5')
INSTANCE_EQ_CSV = os.environ.get('INSTANCE_EQ_CSV', 'data_external/metadata_Instance_events_v3.csv')
INSTANCE_EQ_HDF5 = os.environ.get('INSTANCE_EQ_HDF5', 'data_external/Instance_events_counts.hdf5')
INSTANCE_NOISE_CSV = os.environ.get('INSTANCE_NOISE_CSV', 'data_external/metadata_Instance_noise.csv.bz2')
INSTANCE_NOISE_HDF5 = os.environ.get('INSTANCE_NOISE_HDF5', 'data_external/Instance_noise.hdf5')

SEED = 42

# --- Subset STEAD (gempa lokal kualitas analis + noise) ---
N_EQ = 100_000
N_NOISE = 100_000
SPLIT = (0.8, 0.1, 0.1)                       # train / val / test
STEAD_NPTS = 6000

# --- Uji lintas domain INSTANCE ---
N_INSTANCE_EQ = 10_000
N_INSTANCE_NOISE = 10_000
INSTANCE_NPTS = 12000
VELOCIMETER_CHANNELS = ('HH', 'EH')           # HN/HL/EN = akselerometer, dikeluarkan

# --- Uji lintas domain ETHZ (Swiss, SeisBench) ---
ETHZ_DIR = os.environ.get('ETHZ_DIR', 'data_external/ethz')
N_ETHZ_EQ = 10_000
N_ETHZ_NOISE = 10_000
ETHZ_NPTS = 12000                             # trace ETHZ 120 s (P di detik ke-60), setelah resample
ETHZ_P_PHASES = ('P1', 'Pg', 'Pn', 'PmP', 'P')  # P onset = earliest P-phase pick
ETHZ_NOISE_GAP = 500                          # noise window ends >= 5 s before the P onset

# --- Window & labels (PhaseNet: Zhu & Beroza 2019, GJI 216:261-273) ---
FS = 100.0
WINDOW = 3072                                 # 30,72 s, habis dibagi 4^4
# The P onset in STEAD is always at sample 400-1000 from the start of the trace, so the P position inside the
# window is restricted to this range for ALL sets (train/val/test, STEAD/INSTANCE)
# so that the P position does not become a source of domain shift.
P_POS_MIN = 50
P_POS_MAX = 1000
LABEL_SIGMA = 10                              # 0.1 s at 100 Hz, the same as PhaseNet
# Preprocessing of the TinyPhaseNet input at evaluation: the STEAD waveforms are already band-passed at 1-45 Hz by
# penyusunnya (Mousavi et al. 2020, EQTransformer), sedangkan INSTANCE/ETHZ berupa counts mentah
# dominated by power < 1 Hz. The same filter (causal, on the full trace) is applied to ALL evaluation sets
# before the z-score (see 12_eval_bandpass_1_45.py for the comparison without the filter).
PREFILTER_HZ = (1.0, 45.0)
PREFILTER_ORDER = 4

# --- Model & training ---
BASE_FILTERS = 8
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 5

# --- Baseline STA/LTA ---
STA_S = 1.0
LTA_S = 10.0
BANDPASS_HZ = (2.0, 8.0)                    # chosen on the STEAD validation set from 7 candidates (13_stalta_band_selection.py)
BANDPASS_ORDER = 4                            # causal (sosfilt), as on the MCU
