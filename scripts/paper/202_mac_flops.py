"""
Step 202 (companion compression manuscript): MACs and FLOPs of the four main models with ONE method that explicitly counts the transposed convolution, cross-checked in two other ways.
Models: one-channel standard (models/ablation_final/standard_seed42.keras) and separable (models/svariants/s10_w1_seed42.keras); three-channel standard (pn_seed42.keras, retrained PhaseNet) and separable (pndw_seed42.keras).
Main method (Float32 TFLite graph): MACs per operator, from the tensor shapes in the converted graph:
  CONV_2D: output elements x kh x kw x C_in; DEPTHWISE_CONV_2D: output elements x kh x kw; TRANSPOSE_CONV: input elements x C_out x kh x kw (every input element is multiplied by the whole kernel).
  FLOPs = 2 x MAC (convention: one multiplication and one addition). Only convolution operators are counted; elementwise operations (BatchNorm MUL and ADD, ReLU, softmax) are not counted as MACs.
Cross-checks: (a) analytic formula from the Keras layer configuration (Conv1D: output length x k x C_in x C_out; SeparableConv1D: depthwise + pointwise; Conv1DTranspose: input length x k x C_in x C_out);
(b) the earlier numbers of the TensorFlow profiler (results/final/jawaban_arief_tiga_tahap.json and jawaban_arief_phasenet.json; scripts 150_ and 153_), which only count Conv2D and DepthwiseConv2dNative.
Output: results/final/mac_flops.json and a compact table on screen. Usage (from this folder): python 202_mac_flops.py
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


def mac_keras(m):   # analytic formula from the layer configuration
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
