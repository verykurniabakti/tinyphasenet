"""
Step 187 (device manuscript, Q): combined table of standard convolution, depthwise separable, and quantized formats (Float32, 16x8, INT8, 4/8).
The number of parameters is computed from the seed-42 Keras models (not typed): models/ablation_final/standard_seed42.keras (1 channel, plain Conv1D), models/svariants/s10_w1_seed42.keras (1 channel, separable = TinyPhaseNet),
models/ablation_final/pn_seed42.keras (3 channels, PhaseNet with plain Conv1D), models/ablation_final/pndw_seed42.keras (3 channels, separable). The Float32 file size, the activation-memory lower bound, and the F1-score (STEAD and mean of four datasets, mean of three seeds)
are read from results/final/pareto_kandidat.json (164_). The training recipe of the four models is the same (89_train_ablation_final.py and 75_train_s_variants.py).
Output: results/final/parametros_modelos.json; ../draft_publikasi_1/tabel_dev/{tab_steps,numbers_dw}{,_id}.tex
Usage (from this folder): python 187_tabel_depthwise.py   (tab_steps = combined table of standard convolution, depthwise separable, and quantized formats)
"""
import json, os, statistics as st
import tensorflow as tf

OUT = '../draft_publikasi_1/tabel_dev/'; os.makedirs(OUT, exist_ok=True)
MODELS = {('1ch', 'standard'): 'models/ablation_final/standard_seed42.keras', ('1ch', 'fp32'): 'models/svariants/s10_w1_seed42.keras',
          ('3ch', 'pn'): 'models/ablation_final/pn_seed42.keras', ('3ch', 'fp32'): 'models/ablation_final/pndw_seed42.keras'}
PAR = {f'{f}_{k}': tf.keras.models.load_model(p, compile=False).count_params() for (f, k), p in MODELS.items()}
json.dump(PAR, open('results/final/parametros_modelos.json', 'w'), indent=1)
P = {(c['fam'], c['key']): c for c in json.load(open('results/final/pareto_kandidat.json'))['kandidat']}
# F1 per dataset (mean of three seeds 42-44), from the per-seed result files (the same source as 164_)
SEEDS = [42, 43, 44]; DS = ['stead_test', 'instance_test', 'ethz_test', 'txed_test']
_S = json.load(open('results/final/spektrum_kuantisasi_3seed.json')); _W = json.load(open('results/final/w4a8_perkanal_3seed.json'))
_pn = {int(e['model'].split('seed')[1].split('.')[0]): e for e in json.load(open('results/ablation_final/eval_pn.json'))}
_std = {int(e['model'].split('seed')[1].split('.')[0]): e for e in json.load(open('results/ablation_final/eval_standard.json'))}
_q3 = {int(k): v for k, v in json.load(open('results/final/quant16x8_pndw_per_seed.json')).items()}; _q1 = {int(k): v for k, v in json.load(open('results/final/quant16x8_per_seed.json')).items()}


def f1_ds(fam, key, d):   # F1 (percent), mean of three seeds
    if key == 'standard': xs = [100 * _std[s][d]['f1'] for s in SEEDS]
    elif key == 'pn': xs = [100 * _pn[s][d]['f1'] for s in SEEDS]
    elif key in ('fp32', 'q16x8'): xs = [100 * (_q3 if fam == '3ch' else _q1)[s][d][key]['f1'] for s in SEEDS]
    elif key == 'int8': xs = [100 * _S[fam][str(s)]['int8']['kualitas'][d]['f1'] for s in SEEDS]
    else: xs = [100 * _W[fam][str(s)]['pcmse']['kualitas'][d]['f1'] for s in SEEDS]
    return st.mean(xs)

FAMS = [('1ch', '1ch', 'standard'), ('3ch', '3ch', 'pn')]   # (family, key of the standard-convolution candidate)
FMTS = ['standard', 'fp32', 'q16x8', 'int8', 'w4a8']
BITS = {'standard': '32 / 32', 'fp32': '32 / 32', 'q16x8': '8 / 16', 'int8': '8 / 8', 'w4a8': '4 / 8'}
NAME = {'en': {'1ch': 'Single-channel', '3ch': 'Three-channel', 'standard': 'standard', 'sep': 'separable'}, 'id': {'1ch': '1 kanal', '3ch': '3 kanal', 'standard': 'standar', 'sep': '\\textit{separable}'}}
FMTLAB = {'fp32': 'Float32', 'q16x8': '16x8', 'int8': 'INT8', 'w4a8': '4/8'}


def num(v, d, lang):
    s = f'{v:,.{d}f}'
    return s if lang == 'en' else s.replace(',', '§').replace('.', ',').replace('§', '.')


def write(lang):
    sfx = '' if lang == 'en' else '_id'; N = NAME[lang]
    rows = ''
    for fi, (fam, _, stdkey) in enumerate(FAMS):
        for fmt in FMTS:
            c = P[(fam, stdkey if fmt == 'standard' else fmt)]; par = PAR[f'{fam}_{stdkey if fmt == "standard" else "fp32"}']
            conv = N['standard'] if fmt == 'standard' else N['sep']
            name = f"{N[fam]} {conv}{' (PhaseNet)' if (fmt == 'standard' and fam == '3ch') else ''} {FMTLAB.get(fmt, 'Float32')}"
            ck = stdkey if fmt == 'standard' else fmt
            rows += (f"{name} & {num(par, 0, lang)} & {BITS[fmt]} & {num(c['flash_kb'], 1, lang)} & {num(c['aktivasi_lb_kb'], 0, lang)} & "
                     + ' & '.join(num(f1_ds(fam, ck, d), 1, lang) for d in DS) + f" & {num(c['f1_mean4'], 1, lang)} \\\\\n")
        if fi == 0:
            rows += '\\midrule\n'
    open(OUT + f'tab_steps{sfx}.tex', 'w').write(rows + '\\bottomrule\n')
    s1, d1, s3, d3 = P[('1ch', 'standard')], P[('1ch', 'fp32')], P[('3ch', 'pn')], P[('3ch', 'fp32')]
    M = {'dvDwParXOne': PAR['1ch_standard'] / PAR['1ch_fp32'], 'dvDwParXThr': PAR['3ch_pn'] / PAR['3ch_fp32'],
         'dvDwFileXOne': s1['flash_kb'] / d1['flash_kb'], 'dvDwFileXThr': s3['flash_kb'] / d3['flash_kb'],
         'dvDwDSteadOne': s1['f1_stead'] - d1['f1_stead'], 'dvDwDSteadThr': s3['f1_stead'] - d3['f1_stead'],
         'dvDwDMeanOne': s1['f1_mean4'] - d1['f1_mean4'], 'dvDwDMeanThr': s3['f1_mean4'] - d3['f1_mean4']}
    open(OUT + f'numbers_dw{sfx}.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{num(v, 1, lang)}}}\n' for k, v in M.items()))
    return M


for lg in ('en', 'id'):
    M = write(lg)
print('ok', PAR, {k: round(v, 2) for k, v in M.items()})
