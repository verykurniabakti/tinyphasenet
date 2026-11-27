"""
Step 173 (answer to the advisor, Table 2): REAL 4-bit TFLite files. The per-channel W4A8 models of the main 'pcmse' scheme (models/quant_spectrum/<fam>_w4a8_pcmse_seed<sd>.tflite; 159_/161_) are used as the source.
Those files are of type INT8 (the emulated 4-bit weights are stored one weight per byte). Here the weights of all convolution kernels (CONV_2D, DEPTHWISE_CONV_2D, TRANSPOSE_CONV; filter = 2nd input) are restored to levels -7..7 (q = round(w_int8 * 7 / 127), exact because the INT8 conversion maps the per-channel scale to 127 out of 7 levels), packed two weights per byte (low nibble first, two's complement), the tensor type is changed to INT4, and the per-channel scale is multiplied by 127/7 so that the real value equals q * the 4-bit scale. Biases, other scales, and the graph structure are unchanged.
Verification in the script: the unpacked nibbles are exactly equal to q; the relative error (with respect to the largest weight) between the original INT8 value and the 4-bit value is less than 0.5/127.
NOTE: this file is valid as a storage file but CANNOT be run by the TFLite interpreter or by TFLM (there is no convolution kernel with INT4 weights: 'transpose_conv.cc: weights->type != input->type'). The 4-bit accuracy in the manuscript still comes from the emulated INT8 model.
Output: models/int4_nyata/<fam>_w4a8_pcmse_seed<sd>_int4.tflite and results/final/int4_nyata.json
Usage (from this folder): python 173_make_int4_nyata.py
"""
import os, json
import numpy as np
import flatbuffers
from tensorflow.lite.python import schema_py_generated as S

SEEDS = [42, 43, 44]; OUTD = 'models/int4_nyata'; os.makedirs(OUTD, exist_ok=True)
CODES = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith('_')}


def to_int4(src, dst):
    raw = open(src, 'rb').read(); m = S.ModelT.InitFromObj(S.Model.GetRootAsModel(bytearray(raw), 0)); sg = m.subgraphs[0]; wt = set()
    for op in sg.operators:
        if CODES[m.operatorCodes[op.opcodeIndex].builtinCode] in ('CONV_2D', 'DEPTHWISE_CONV_2D', 'TRANSPOSE_CONV'): wt.add(int(op.inputs[1]))
    tot = 0; maxerr = 0.0
    for ti in sorted(wt):
        t = sg.tensors[ti]; assert t.type == S.TensorType.INT8
        buf = m.buffers[t.buffer]; w8 = np.frombuffer(bytes(buf.data), dtype=np.int8).astype(np.int32); q = np.rint(w8 * 7 / 127).astype(np.int32); assert q.min() >= -7 and q.max() <= 7
        sc = np.array(t.quantization.scale, dtype=np.float64); qd = int(t.quantization.quantizedDimension); shp = list(t.shape); bs = [1] * len(shp); bs[qd] = -1; scb = sc.reshape(bs) if len(sc) > 1 else sc
        real8 = w8.reshape(shp) * scb; real4 = q.reshape(shp) * scb * 127 / 7; maxerr = max(maxerr, float(np.max(np.abs(real8 - real4)) / (np.max(np.abs(real8)) + 1e-12)))
        nib = (q & 0xF).astype(np.uint8); nib = np.append(nib, 0) if nib.size % 2 else nib; packed = (nib[0::2] | (nib[1::2] << 4)).astype(np.uint8)
        lo = (packed & 0xF).astype(np.int8); lo = np.where(lo > 7, lo - 16, lo); hi = (packed >> 4).astype(np.int8); hi = np.where(hi > 7, hi - 16, hi)
        un = np.empty(packed.size * 2, dtype=np.int8); un[0::2] = lo; un[1::2] = hi; assert np.array_equal(un[:w8.size], q), 'pembongkaran nibble gagal'
        buf.data = list(packed.tobytes()); t.type = S.TensorType.INT4; t.quantization.scale = (sc * 127 / 7).astype(np.float32).tolist(); tot += w8.size
    assert maxerr < 0.5 / 127 + 1e-9, maxerr
    b = flatbuffers.Builder(1024); b.Finish(m.Pack(b), b'TFL3'); out = bytes(b.Output()); open(dst, 'wb').write(out)
    return dict(n_tensor_kernel=len(wt), n_bobot_kernel=int(tot), bytes_int8=len(raw), bytes_int4=len(out), bytes_teoretis=len(raw) - tot // 2, maks_galat_relatif=maxerr)


R = {}
for fam in ('1ch', '3ch'):
    for sd in SEEDS:
        r = to_int4(f'models/quant_spectrum/{fam}_w4a8_pcmse_seed{sd}.tflite', f'{OUTD}/{fam}_w4a8_pcmse_seed{sd}_int4.tflite'); R.setdefault(fam, {})[str(sd)] = r; print(fam, sd, r, flush=True)
json.dump(R, open('results/final/int4_nyata.json', 'w'), indent=1)
