"""
Step 153 (answer to the advisor's question, Phase 1, THREE CHANNELS): original PhaseNet -> depthwise separable PhaseNet -> 16x8 quantization, on the SAME COMPUTER. Four rows:
  official = the official PhaseNet (checkpoint 190703-214543 transferred to Keras, phasenet_keras.py; quality from results/phasenet_original and results/final/phasenet_txed.json, original preprocessing, trained on NCEDC by Zhu & Beroza),
  pn       = the same PhaseNet architecture retrained on STEAD with our final recipe (89_ variant pn; quality results/ablation_final/eval_pn.json from 90_),
  pndw     = PhaseNet with depthwise separable convolutions, the same recipe (89_ variant pndw; quality from results/final/quant16x8_pndw_per_seed.json, fp32 part, from 151_),
  q        = pndw quantized 16x8 (151_; models/final_16x8_pndw/seed<N>.tflite).
Time and size are measured again here (TFLite one thread, without and with the built-in delegate, 300 invocations on 50 three-channel STEAD windows). Seeds = those available in all files (arguments after 'ukur', default 42 43 44).
Usage: python 153_jawaban_arief_phasenet.py [ukur] [seed ...]. Output: results/final/jawaban_arief_phasenet.json; ../jawaban_prof_arief_fase1/{numbers_pn.tex, tabel_pn/tab_arsitektur.tex, tab_kualitas.tex, tab_waktu.tex}.
"""
import os, sys, json, re, time, platform, statistics as st
import numpy as np

OUT_JSON = 'results/final/jawaban_arief_phasenet.json'; OUT = '../jawaban_prof_arief_fase1'; TAB = 'tabel_pn'; PRE = 'p'
args = [a for a in sys.argv[1:] if a != 'ukur']; DO_MEASURE = 'ukur' in sys.argv[1:]; SEEDS = [int(a) for a in args] or [42, 43, 44]
STAGES = [('resmi', '1a \\emph{PhaseNet} resmi'), ('pn', '1b \\emph{PhaseNet} dilatih ulang'), ('pndw', '2 \\emph{PhaseNet depthwise}'), ('q', '3 \\emph{PhaseNet depthwise} 16x8')]
N_WIN, N_WARM, N_REP = 50, 20, 300
DSETS = [('stead_test', 'STEAD'), ('instance_test', 'INSTANCE'), ('ethz_test', 'ETHZ'), ('txed_test', 'TXED')]


def measure():
    import tensorflow as tf
    import config as C
    import data3ch as D3
    from phasenet_keras import create_phasenet_keras, load_official_weights

    stead, sw = D3.load_stead3(); te = stead[stead.split == 'test'].sample(n=N_WIN, random_state=C.SEED); X, _ = D3.eval_windows3(te, sw); del sw

    def macs(model):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
        conc = tf.function(lambda x: model(x)).get_concrete_function(tf.TensorSpec([1, C.WINDOW, 3], tf.float32))
        _, gd = convert_variables_to_constants_v2_as_graph(conc)
        with tf.Graph().as_default() as g:
            tf.graph_util.import_graph_def(gd, name='')
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation(); opts['account_type_regexes'] = ['Conv2D.*', 'DepthwiseConv2dNative']   # convolutions only (the same as the manuscript's depthwise table)
            fl = tf.compat.v1.profiler.profile(graph=g, run_meta=tf.compat.v1.RunMetadata(), cmd='op', options=opts)
        return fl.total_float_ops / 2.0

    def tflite_float(model):
        conc = tf.function(lambda x: model(x)).get_concrete_function(tf.TensorSpec([1, C.WINDOW, 3], tf.float32))
        return tf.lite.TFLiteConverter.from_concrete_functions([conc], model).convert()

    def latency(blob, delegates):
        kw = dict(model_content=blob, num_threads=1)
        if not delegates: kw['experimental_op_resolver_type'] = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        it = tf.lite.Interpreter(**kw); it.allocate_tensors(); inp = it.get_input_details()[0]
        s, zp = inp['quantization']; info = np.iinfo(inp['dtype']) if np.issubdtype(inp['dtype'], np.integer) else None
        Q = [np.clip(np.round(X[k:k + 1] / s) + zp, info.min, info.max).astype(inp['dtype']) if info is not None else X[k:k + 1] for k in range(len(X))]
        t = []
        for r in range(N_WARM + N_REP):
            it.set_tensor(inp['index'], Q[r % len(Q)]); t0 = time.perf_counter_ns(); it.invoke(); dt = (time.perf_counter_ns() - t0) / 1e6
            if r >= N_WARM: t.append(dt)
        return dict(median=float(np.median(t)), q1=float(np.percentile(t, 25)), q3=float(np.percentile(t, 75)))

    def row(model, blob):
        return dict(params=int(model.count_params()), macs=float(macs(model)), bytes=len(blob), lat_ref=latency(blob, False), lat_def=latency(blob, True))

    res = dict(maquina=dict(cpu=os.popen('sysctl -n machdep.cpu.brand_string').read().strip(), os=platform.platform(), tf=tf.__version__, threads=1, n_jendela=N_WIN, n_pemanasan=N_WARM, n_ulang=N_REP), seeds={}, resmi=None)
    m = load_official_weights(create_phasenet_keras(conv='standard', name='PhaseNet-resmi')); res['resmi'] = row(m, tflite_float(m)); del m; tf.keras.backend.clear_session()
    print('resmi', res['resmi']['params'], f"{res['resmi']['macs']/1e6:.2f}M", res['resmi']['bytes'], flush=True)
    for sd in SEEDS:
        r = {}
        for name in ('pn', 'pndw'):
            m = tf.keras.models.load_model(f'models/ablation_final/{name}_seed{sd}.keras', compile=False); r[name] = row(m, tflite_float(m)); del m; tf.keras.backend.clear_session()
        m = tf.keras.models.load_model(f'models/ablation_final/pndw_seed{sd}.keras', compile=False); mc = float(macs(m)); pc = int(m.count_params()); del m; tf.keras.backend.clear_session()
        blob = open(f'models/final_16x8_pndw/seed{sd}.tflite', 'rb').read()
        r['q'] = dict(params=pc, macs=mc, bytes=len(blob), lat_ref=latency(blob, False), lat_def=latency(blob, True))
        res['seeds'][sd] = r; print('seed', sd, {k: (v['params'], f"{v['macs']/1e6:.2f}M", v['bytes'], f"{v['lat_ref']['median']:.2f}/{v['lat_def']['median']:.2f} ms") for k, v in r.items()}, flush=True)
    json.dump(res, open(OUT_JSON, 'w'), indent=1)


def fixdec(t):
    return re.sub(r'(?<=\d)\.(?=\d)', ',', t).replace('_', '.')   # decimal comma; '_' is the thousands separator


def tables():
    res = json.load(open(OUT_JSON)); S = {int(k): v for k, v in res['seeds'].items()}; seeds = sorted(S)
    off = json.load(open('results/phasenet_original/metrics.json'))['test']; offt = json.load(open('results/final/phasenet_txed.json'))['detection']
    pn = {int(e['model'].split('seed')[1].split('.')[0]): e for e in json.load(open('results/ablation_final/eval_pn.json'))}
    q = {int(k): v for k, v in json.load(open('results/final/quant16x8_pndw_per_seed.json')).items()}
    qual = {k: {d: {m: [] for m in ('recall', 'f1', 'fpr')} for d, _ in DSETS} for k, _ in STAGES}
    for d, _ in DSETS:
        for m in ('recall', 'f1', 'fpr'):
            qual['resmi'][d][m] = [100 * (offt[m] if d == 'txed_test' else off[d][m])]
            for sd in seeds:
                qual['pn'][d][m].append(100 * pn[sd][d][m]); qual['pndw'][d][m].append(100 * q[sd][d]['fp32'][m]); qual['q'][d][m].append(100 * q[sd][d]['q16x8'][m])
    ms = lambda v, p=2: f"{st.mean(v):.{p}f}" + (f" $\\pm$ {st.stdev(v):.{p}f}" if len(v) > 1 else '')
    mac = {}; mm = lambda k, v: mac.__setitem__(k, v)
    def stage_rows(key):   # list of values per seed (official: one value)
        return [res['resmi']] if key == 'resmi' else [S[sd][key] for sd in seeds]
    rows = ''
    for key, lab in STAGES:
        v = stage_rows(key); wbits = '8 / 16' if key == 'q' else '32 / 32'
        rows += f"{lab} & {v[0]['params']:,} & {v[0]['macs']/1e6:.2f} & {wbits} & {st.mean(x['bytes'] for x in v)/1024:.1f} \\\\\n".replace(',', '_')
        mm('sz' + key, f"{st.mean(x['bytes'] for x in v)/1024:.1f}"); mm('pr' + key, f"{v[0]['params']:,}".replace(',', '_')); mm('mc' + key, f"{v[0]['macs']/1e6:.2f}")
    rq = ''
    for d, dl in DSETS:
        for j, (key, lab) in enumerate(STAGES):
            c = qual[key][d]; rq += (f"\\multirow{{4}}{{*}}{{{dl}}} & " if j == 0 else " & ") + f"{lab} & {ms(c['recall'])} & {ms(c['f1'])} & {ms(c['fpr'])} \\\\\n"
        rq += '\\midrule\n' if d != 'txed_test' else ''
    rt = ''
    for key, lab in STAGES:
        v = stage_rows(key); a = [x['lat_ref']['median'] for x in v]; b = [x['lat_def']['median'] for x in v]
        sd_ = lambda z: (f" $\\pm$ {st.stdev(z):.2f}" if len(z) > 1 else '')
        rt += f"{lab} & {st.mean(a):.2f}{sd_(a)} & {st.mean(b):.2f}{sd_(b)} & {1000/st.mean(b):.0f} \\\\\n"
        mm('lr' + key, f"{st.mean(a):.2f}"); mm('ld' + key, f"{st.mean(b):.2f}")
    sz = {k: st.mean(x['bytes'] for x in stage_rows(k)) / 1024 for k, _ in STAGES}; mx = max(sz.values())
    mm('redPr', f"{100*(1-S[seeds[0]]['pndw']['params']/res['resmi']['params']):.1f}"); mm('redMc', f"{100*(1-S[seeds[0]]['pndw']['macs']/res['resmi']['macs']):.1f}")
    mm('redSzDQ', f"{100*(1-sz['q']/sz['pndw']):.1f}"); mm('redSzRQ', f"{100*(1-sz['q']/sz['resmi']):.1f}"); mm('szRatioRQ', f"{sz['resmi']/sz['q']:.1f}")
    mm('lRatioQDW', f"{st.mean(x['lat_ref']['median'] for x in stage_rows('q'))/st.mean(x['lat_ref']['median'] for x in stage_rows('pndw')):.1f}")
    mm('lRatioNDW', f"{st.mean(x['lat_ref']['median'] for x in stage_rows('pn'))/st.mean(x['lat_ref']['median'] for x in stage_rows('pndw')):.2f}")
    for k in ('stead', 'instance', 'ethz', 'txed'):
        d = f'{k}_test'; tag = {'stead': 'St', 'instance': 'In', 'ethz': 'Et', 'txed': 'Tx'}[k]
        for key, t2 in (('resmi', 'Res'), ('pn', 'Pn'), ('pndw', 'Dw'), ('q', 'Qq')): mm(f'f{tag}{t2}', f"{st.mean(qual[key][d]['f1']):.2f}")
    mm('dFQMax', f"{max(abs(st.mean(qual['q'][d]['f1']) - st.mean(qual['pndw'][d]['f1'])) for d, _ in DSETS):.2f}")
    mm('dFPnDwSt', f"{st.mean(qual['pn']['stead_test']['f1']) - st.mean(qual['pndw']['stead_test']['f1']):.1f}")
    mm('nSeeds', str(len(seeds)))
    os.makedirs(f'{OUT}/{TAB}', exist_ok=True)
    for nm, body in (('tab_arsitektur', rows), ('tab_kualitas', rq), ('tab_waktu', rt)):
        open(f'{OUT}/{TAB}/{nm}.tex', 'w').write(fixdec(body) + '\\bottomrule\n')
    raw = {'cpuName': res['maquina']['cpu'], 'tfVer': res['maquina']['tf'], 'nRep': str(res['maquina']['n_ulang']), 'nWin': str(res['maquina']['n_jendela']), 'nWarm': str(res['maquina']['n_pemanasan'])}
    raw.update({f'bar{k}': f"{4.0*sz[k]/mx:.3f}" for k, _ in STAGES})
    res_teapz = json.load(open('results/final/jawaban_arief_tiga_tahap.json'))
    sz_tp = {
        'std': st.mean(res_teapz['seeds'][str(sd)]['standard']['bytes'] for sd in seeds) / 1024,
        'sep': st.mean(res_teapz['seeds'][str(sd)]['separable']['bytes'] for sd in seeds) / 1024,
        'q': st.mean(res_teapz['seeds'][str(sd)]['q16x8']['bytes'] for sd in seeds) / 1024,
    }
    raw.update({
        'bgabpn': f"{5.0 * sz['pn'] / mx:.3f}",
        'bgabpndw': f"{5.0 * sz['pndw'] / mx:.3f}",
        'bgabq': f"{5.0 * sz['q'] / mx:.3f}",
        'bgabtpstd': f"{5.0 * sz_tp['std'] / mx:.3f}",
        'bgabtpsep': f"{5.0 * sz_tp['sep'] / mx:.3f}",
        'bgabtpq': f"{5.0 * sz_tp['q'] / mx:.3f}",
    })
    num = fixdec(''.join(f'\\newcommand{{\\{PRE}{k}}}{{{v}}}\n' for k, v in mac.items())) + ''.join(f'\\newcommand{{\\{PRE}{k}}}{{{v}}}\n' for k, v in raw.items())
    open(f'{OUT}/numbers_pn.tex', 'w').write('% Dibangkitkan oleh 153_jawaban_arief_phasenet.py\n' + num); print('ok', len(mac) + len(raw), 'makro; seed', seeds)


if __name__ == '__main__':
    if DO_MEASURE: measure()
    tables()
