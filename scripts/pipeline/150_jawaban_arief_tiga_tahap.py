"""
Step 150 (answer to the advisor's question, Phase 1): three-stage comparison of the model on the SAME COMPUTER:
  stage 1 = standard convolution (Float32), stage 2 = depthwise separable TinyPhaseNet (Float32), stage 3 = TinyPhaseNet quantized 16x8.
Seeds 42, 43, 44 (the standard model only exists for those three seeds; the same final recipe, 89_train_ablation_final.py).
Contents: (a) parameters, MACs, TFLite file size; (b) latency per window (TFLite, one thread, without and with the built-in delegate); (c) quality from results already measured on this computer
(results/ablation_final/eval_standard.json, results/final/quant16x8_per_seed.json; threshold from the STEAD validation of each model, protocol 81_/87_/90_).
Output: results/final/jawaban_arief_tiga_tahap.json; ../jawaban_prof_arief_fase1/{numbers_jawaban.tex, tabel/tab_arsitektur.tex, tab_kualitas.tex, tab_waktu.tex}.
Usage: python 150_jawaban_arief_tiga_tahap.py [ukur]   (without an argument it only builds the tables from the existing JSON; 'ukur' runs the latency and size measurements)
"""
import os, sys, json, re, time, platform, statistics as st
import numpy as np

OUT_JSON = 'results/final/jawaban_arief_tiga_tahap.json'; OUT = '../jawaban_prof_arief_fase1'; SEEDS = [42, 43, 44]
STAGES = [('standard', '1 Standar'), ('separable', '2 \\emph{Separable}'), ('q16x8', '3 \\emph{Separable} 16x8')]
N_WIN, N_WARM, N_REP = 50, 20, 300


def measure():
    import tensorflow as tf
    import config as C
    from data import load_stead, eval_windows
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    stead, sz = load_stead(); te = stead[stead.split == 'test'].sample(n=N_WIN, random_state=C.SEED); X, _ = eval_windows(te, sz); del sz

    def macs(model):
        conc = tf.function(lambda x: model(x)).get_concrete_function(tf.TensorSpec([1, C.WINDOW, 1], tf.float32))
        _, gd = convert_variables_to_constants_v2_as_graph(conc)
        with tf.Graph().as_default() as g:
            tf.graph_util.import_graph_def(gd, name='')
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation(); opts['account_type_regexes'] = ['Conv2D.*', 'DepthwiseConv2dNative']   # convolutions only, the same as the manuscript's depthwise table
            fl = tf.compat.v1.profiler.profile(graph=g, run_meta=tf.compat.v1.RunMetadata(), cmd='op', options=opts)
        return fl.total_float_ops / 2.0

    def tflite_float(model):
        conc = tf.function(lambda x: model(x)).get_concrete_function(tf.TensorSpec([1, C.WINDOW, 1], tf.float32))
        return tf.lite.TFLiteConverter.from_concrete_functions([conc], model).convert()

    def latency(blob, delegates):
        kw = dict(model_content=blob, num_threads=1)
        if not delegates: kw['experimental_op_resolver_type'] = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        it = tf.lite.Interpreter(**kw); it.allocate_tensors(); inp, out = it.get_input_details()[0], it.get_output_details()[0]
        s, zp = inp['quantization']; info = np.iinfo(inp['dtype']) if np.issubdtype(inp['dtype'], np.integer) else None
        Q = [np.clip(np.round(X[k:k + 1] / s) + zp, info.min, info.max).astype(inp['dtype']) if info is not None else X[k:k + 1] for k in range(len(X))]
        t = []
        for r in range(N_WARM + N_REP):
            it.set_tensor(inp['index'], Q[r % len(Q)]); t0 = time.perf_counter_ns(); it.invoke(); dt = (time.perf_counter_ns() - t0) / 1e6
            if r >= N_WARM: t.append(dt)
        return dict(median=float(np.median(t)), q1=float(np.percentile(t, 25)), q3=float(np.percentile(t, 75)))

    res = dict(maquina=dict(cpu=os.popen('sysctl -n machdep.cpu.brand_string').read().strip(), os=platform.platform(), tf=tf.__version__, threads=1, n_jendela=N_WIN, n_pemanasan=N_WARM, n_ulang=N_REP), seeds={})
    for sd in SEEDS:
        r = {}
        mods = {'standard': f'models/ablation_final/standard_seed{sd}.keras', 'separable': f'models/svariants/s10_w1_seed{sd}.keras'}
        for name, path in mods.items():
            m = tf.keras.models.load_model(path, compile=False); blob = tflite_float(m)
            r[name] = dict(params=int(m.count_params()), macs=float(macs(m)), bytes=len(blob), lat_ref=latency(blob, False), lat_def=latency(blob, True))
            del m; tf.keras.backend.clear_session()
        blob = open(f'models/final_16x8/seed{sd}.tflite', 'rb').read()
        r['q16x8'] = dict(params=r['separable']['params'], macs=r['separable']['macs'], bytes=len(blob), lat_ref=latency(blob, False), lat_def=latency(blob, True))
        res['seeds'][sd] = r; print('seed', sd, {k: (v['params'], f"{v['macs']/1e6:.2f}M", v['bytes'], f"{v['lat_ref']['median']:.2f}/{v['lat_def']['median']:.2f} ms") for k, v in r.items()}, flush=True)
    json.dump(res, open(OUT_JSON, 'w'), indent=1)


def fixdec(t):
    return re.sub(r'(?<=\d)\.(?=\d)', ',', t).replace('_', '.')   # decimal comma; '_' is the thousands separator


def tables():
    res = json.load(open(OUT_JSON)); S = {int(k): v for k, v in res['seeds'].items()}
    std = json.load(open('results/ablation_final/eval_standard.json')); q = {int(k): v for k, v in json.load(open('results/final/quant16x8_per_seed.json')).items()}
    dsets = [('stead_test', 'STEAD'), ('instance_test', 'INSTANCE'), ('ethz_test', 'ETHZ'), ('txed_test', 'TXED')]
    qual = {}
    for s in STAGES: qual[s[0]] = {d: {m: [] for m in ('recall', 'f1', 'fpr')} for d, _ in dsets}
    for i, sd in enumerate(SEEDS):
        assert std[i]['model'].endswith(f'standard_seed{sd}.keras')
        for d, _ in dsets:
            for m in ('recall', 'f1', 'fpr'):
                qual['standard'][d][m].append(100 * std[i][d][m]); qual['separable'][d][m].append(100 * q[sd][d]['fp32'][m]); qual['q16x8'][d][m].append(100 * q[sd][d]['q16x8'][m])
    ms = lambda v, p=2: f"{st.mean(v):.{p}f} $\\pm$ {st.stdev(v):.{p}f}"
    mac = {}; mm = lambda k, v: mac.__setitem__(k, v)
    # architecture and size
    rows = ''
    for key, lab in STAGES:
        v = [S[sd][key] for sd in SEEDS]; wbits = {'standard': ('32', '32'), 'separable': ('32', '32'), 'q16x8': ('8', '16')}[key]
        rows += f"{lab} & {v[0]['params']:,} & {v[0]['macs']/1e6:.2f} & {wbits[0]} / {wbits[1]} & {st.mean(x['bytes'] for x in v)/1024:.1f} \\\\\n".replace(',', '_')
        mm(('sz' + key).replace('q16x8', 'Qq'), f"{st.mean(x['bytes'] for x in v)/1024:.1f}"); mm(('pr' + key).replace('q16x8', 'Qq'), f"{v[0]['params']:,}".replace(',', '_')); mm(('mc' + key).replace('q16x8', 'Qq'), f"{v[0]['macs']/1e6:.2f}")
    # kualitas
    rq = ''
    for d, dl in dsets:
        for j, (key, lab) in enumerate(STAGES):
            c = qual[key][d]; rq += (f"\\multirow{{3}}{{*}}{{{dl}}} & " if j == 0 else " & ") + f"{lab} & {ms(c['recall'])} & {ms(c['f1'])} & {ms(c['fpr'])} \\\\\n"
        rq += '\\midrule\n' if d != 'txed_test' else ''
    # time
    rt = ''
    for key, lab in STAGES:
        a = [S[sd][key]['lat_ref']['median'] for sd in SEEDS]; b = [S[sd][key]['lat_def']['median'] for sd in SEEDS]
        iq = [S[sd][key]['lat_ref']['q3'] - S[sd][key]['lat_ref']['q1'] for sd in SEEDS]
        rt += f"{lab} & {st.mean(a):.2f} $\\pm$ {st.stdev(a):.2f} & {st.mean(b):.2f} $\\pm$ {st.stdev(b):.2f} & {1000/st.mean(b):.0f} \\\\\n"
        mm(('lr' + key).replace('q16x8', 'Qq'), f"{st.mean(a):.2f}"); mm(('ld' + key).replace('q16x8', 'Qq'), f"{st.mean(b):.2f}")
    mm('lRatioStdSep', f"{st.mean(S[sd]['standard']['lat_ref']['median'] for sd in SEEDS)/st.mean(S[sd]['separable']['lat_ref']['median'] for sd in SEEDS):.2f}")
    mm('szRatioStdQ', f"{st.mean(S[sd]['standard']['bytes'] for sd in SEEDS)/st.mean(S[sd]['q16x8']['bytes'] for sd in SEEDS):.1f}")
    mm('prRatio', f"{S[42]['standard']['params']/S[42]['separable']['params']:.2f}"); mm('mcRatio', f"{S[42]['standard']['macs']/S[42]['separable']['macs']:.2f}")
    for k in ('stead', 'instance', 'ethz', 'txed'):
        d = f'{k}_test'
        for key, tag in (('standard', 'Std'), ('separable', 'Sep'), ('q16x8', 'Qq')): mm(f'f{k.capitalize()}{tag}'.replace('Stead', 'St').replace('Instance', 'In').replace('Ethz', 'Et').replace('Txed', 'Tx'), f"{st.mean(qual[key][d]['f1']):.2f}")
    sz = {k: st.mean(S[sd][k]['bytes'] for sd in SEEDS) / 1024 for k, _ in STAGES}; mx = max(sz.values())
    mm('redPr', f"{100*(1-S[42]['separable']['params']/S[42]['standard']['params']):.1f}"); mm('redMc', f"{100*(1-S[42]['separable']['macs']/S[42]['standard']['macs']):.1f}")
    mm('redSzSepQ', f"{100*(1-sz['q16x8']/sz['separable']):.1f}"); mm('redSzStdSep', f"{100*(1-sz['separable']/sz['standard']):.1f}"); mm('redSzStdQ', f"{100*(1-sz['q16x8']/sz['standard']):.1f}")
    mm('lRatioQSep', f"{st.mean(S[sd]['q16x8']['lat_ref']['median'] for sd in SEEDS)/st.mean(S[sd]['separable']['lat_ref']['median'] for sd in SEEDS):.1f}")
    dF = {k: st.mean(qual[k]['stead_test']['f1']) for k, _ in STAGES}; mm('dFStdSep', f"{dF['standard']-dF['separable']:.1f}"); mm('dFSepQMax', f"{max(abs(st.mean(qual['q16x8'][d]['f1'])-st.mean(qual['separable'][d]['f1'])) for d, _ in dsets):.2f}")
    raw = {'cpuName': res['maquina']['cpu'], 'tfVer': res['maquina']['tf'], 'nRep': str(res['maquina']['n_ulang']), 'nWin': str(res['maquina']['n_jendela']), 'nWarm': str(res['maquina']['n_pemanasan'])}
    raw.update({'barstandard': f"{4.0*sz['standard']/mx:.3f}", 'barseparable': f"{4.0*sz['separable']/mx:.3f}", 'barQq': f"{4.0*sz['q16x8']/mx:.3f}"})   # bar width (cm) in the figure, decimal point for TikZ
    os.makedirs(f'{OUT}/tabel', exist_ok=True)
    for nm, body in (('tab_arsitektur', rows), ('tab_kualitas', rq), ('tab_waktu', rt)):
        txt = fixdec(body)
        open(f'{OUT}/tabel/{nm}.tex', 'w').write(txt + '\\bottomrule\n')
    num = fixdec(''.join(f'\\newcommand{{\\j{k}}}{{{v}}}\n' for k, v in mac.items()))
    num += ''.join(f'\\newcommand{{\\j{k}}}{{{v}}}\n' for k, v in raw.items())
    open(f'{OUT}/numbers_jawaban.tex', 'w').write('% Dibangkitkan oleh 150_jawaban_arief_tiga_tahap.py\n' + num); print('ok', len(mac) + len(raw), 'makro')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'ukur': measure()
    tables()
