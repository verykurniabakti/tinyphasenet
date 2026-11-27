"""
Langkah 202 (draf pendamping kompresi): MAC dan FLOPs empat model utama dengan SATU metode yang eksplisit menghitung konvolusi transpos, diperiksa silang dengan dua cara lain.
Model: satu kanal standar (models/ablation_final/standard_seed42.keras) dan separable (models/svariants/s10_w1_seed42.keras); tiga kanal standar (pn_seed42.keras, PhaseNet dilatih ulang) dan separable (pndw_seed42.keras).
Metode utama (graf TFLite Float32): MAC per operator, dari bentuk tensor di graf yang dikonversi:
  CONV_2D: elemen keluaran x kh x kw x C_masuk; DEPTHWISE_CONV_2D: elemen keluaran x kh x kw; TRANSPOSE_CONV: elemen masukan x C_keluaran x kh x kw (tiap elemen masukan dikalikan seluruh kernel).
  FLOPs = 2 x MAC (konvensi: satu perkalian dan satu penjumlahan). Hanya operator konvolusi dihitung; operasi elementwise (MUL dan ADD BatchNorm, ReLU, softmax) tidak dihitung sebagai MAC.
Pemeriksaan silang: (a) rumus analitik dari konfigurasi lapis Keras (Conv1D: panjang keluaran x k x C_masuk x C_keluaran; SeparableConv1D: depthwise + pointwise; Conv1DTranspose: panjang masukan x k x C_masuk x C_keluaran);
(b) angka lama dari profiler TensorFlow (results/final/jawaban_arief_tiga_tahap.json dan jawaban_arief_phasenet.json; skrip 150_ dan 153_), yang hanya menghitung Conv2D dan DepthwiseConv2dNative.
Keluaran: results/final/mac_flops.json dan tabel ringkas di layar. Pemakaian (dari folder ini): python 202_mac_flops.py
"""
import os, json
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import config as C

MODELS = {('1ch', 'standar'): ('models/ablation_final/standard_seed42.keras', 1), ('1ch', 'separable'): ('models/svariants/s10_w1_seed42.keras', 1),
          ('3ch', 'standar'): ('models/ablation_final/pn_seed42.keras', 3), ('3ch', 'separable'): ('models/ablation_final/pndw_seed42.keras', 3)}
OLD = {('1ch', 'standar'): ('jawaban_arief_tiga_tahap.json', 'standard'), ('1ch', 'separable'): ('jawaban_arief_tiga_tahap.json', 'separable'),
       ('3ch', 'standar'): ('jawaban_arief_phasenet.json', 'pn'), ('3ch', 'separable'): ('jawaban_arief_phasenet.json', 'pndw')}
NODEL = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES


def mac_tflite(m, K):
    conc = tf.function(lambda x: m(x)).get_concrete_function(tf.TensorSpec([1, C.WINDOW, K], tf.float32)); blob = tf.lite.TFLiteConverter.from_concrete_functions([conc], m).convert()
    it = tf.lite.Interpreter(model_content=blob, experimental_op_resolver_type=NODEL); it.allocate_tensors(); td = {d['index']: d for d in it.get_tensor_details()}
    sh = lambda i: [int(v) for v in td[i]['shape']]; by = {}; n_other = {}
    for o in it._get_ops_details():
        n = o['op_name']
        if n == 'CONV_2D': f = sh(o['inputs'][1]); mac = int(np.prod(sh(o['outputs'][0]))) * f[1] * f[2] * f[3]
        elif n == 'DEPTHWISE_CONV_2D': f = sh(o['inputs'][1]); mac = int(np.prod(sh(o['outputs'][0]))) * f[1] * f[2]
        elif n == 'TRANSPOSE_CONV': f = sh(o['inputs'][1]); mac = int(np.prod(sh(o['inputs'][2]))) * f[0] * f[1] * f[2]
        else: n_other[n] = n_other.get(n, 0) + 1; continue
        by[n] = by.get(n, 0) + mac
    return sum(by.values()), by, n_other


def mac_keras(m):   # rumus analitik dari konfigurasi lapis
    tot = {'conv': 0, 'depthwise': 0, 'pointwise': 0, 'transpose': 0}
    for l in m.layers:
        t = type(l).__name__
        if t == 'Conv1D': tot['conv'] += int(l.output.shape[1]) * l.kernel_size[0] * int(l.input.shape[-1]) * l.filters
        elif t == 'SeparableConv1D':
            cin = int(l.input.shape[-1]); lo = int(l.output.shape[1]); tot['depthwise'] += lo * l.kernel_size[0] * cin * l.depth_multiplier; tot['pointwise'] += lo * cin * l.depth_multiplier * l.filters
        elif t == 'Conv1DTranspose': tot['transpose'] += int(l.input.shape[1]) * l.kernel_size[0] * int(l.input.shape[-1]) * l.filters
    return sum(tot.values()), tot


res = {}
for (fam, conv), (path, K) in MODELS.items():
    m = tf.keras.models.load_model(path, compile=False); mt, by, other = mac_tflite(m, K); mk, bk = mac_keras(m); fn, key = OLD[(fam, conv)]; old = json.load(open('results/final/' + fn))['seeds']['42'][key]
    res[f'{fam}_{conv}'] = dict(params=int(m.count_params()), macs_tflite=mt, flops=2 * mt, por_operator=by, operasi_lain=other, macs_keras_analitik=mk, rincian_keras=bk, macs_profiler_lama=old['macs'])
    print(f'{fam} {conv:9s} params {m.count_params():7d}  MAC TFLite {mt:11,d}  FLOPs {2 * mt:12,d}  Keras analitik {mk:11,d}  old profiler {old["macs"]:11,.0f}   {by}')
for fam in ('1ch', '3ch'):
    a, b = res[f'{fam}_standar'], res[f'{fam}_separable']
    res[f'{fam}_rasio'] = dict(params=a['params'] / b['params'], macs=a['macs_tflite'] / b['macs_tflite']); print(fam, 'rasio standar / separable: parameter %.2f, MAC %.2f' % (res[f'{fam}_rasio']['params'], res[f'{fam}_rasio']['macs']))
json.dump(res, open('results/final/mac_flops.json', 'w'), indent=1)
