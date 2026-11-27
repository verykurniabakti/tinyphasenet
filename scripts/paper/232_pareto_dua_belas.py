"""
Step 232 (companion and device summaries): Pareto analysis of TWELVE configurations measured on the ESP32-S3: TinyPhaseNet (separable) and PhaseNet (standard), one channel and three channels, each in Float32, 16x8, INT8.
Extends 193_ (eight configurations; standard only in Float32) with the quantized standard models (231_). F1 = mean of four datasets (STEAD, INSTANCE, ETHZ, TXED), mean of three seeds; latency and energy per window from hw_tiga_seed.json, hw_padrao_tiga_seed.json, hw_padrao_kuant_tiga_seed.json.
Dominance: A dominates B if A is better or equal in F1 (high), latency (low), and energy (low), and better in at least one. Output: results/final/pareto_dua_belas.json (the same schema as pareto_delapan.json for 206_) and a summary on screen.
Usage (from this folder): python 232_pareto_dua_belas.py
"""
import json, itertools
import numpy as np

SEEDS = [42, 43, 44]; DS = ['stead_test', 'instance_test', 'ethz_test', 'txed_test']
S = json.load(open('results/final/spektrum_kuantisasi_3seed.json'))
q3 = {int(k): v for k, v in json.load(open('results/final/quant16x8_pndw_per_seed.json')).items()}
q1 = {int(k): v for k, v in json.load(open('results/final/quant16x8_per_seed.json')).items()}
H = json.load(open('results/final/hw_tiga_seed.json')); PD = json.load(open('results/final/hw_padrao_tiga_seed.json')); PQ = json.load(open('results/final/hw_padrao_kuant_tiga_seed.json'))
PT = json.load(open('results/final/ptq_padrao_3seed.json'))
std1 = {int(e['model'].split('seed')[1].split('.')[0]): e for e in json.load(open('results/ablation_final/eval_standard.json'))}
pn = {int(e['model'].split('seed')[1].split('.')[0]): e for e in json.load(open('results/ablation_final/eval_pn.json'))}
V6 = ['1ch_fp32', '1ch_16x8', '1ch_int8', '3ch_fp32', '3ch_16x8', '3ch_int8']; V12 = V6 + ['1ch_std_fp32', '1ch_std_16x8', '1ch_std_int8', '3ch_std_fp32', '3ch_std_16x8', '3ch_std_int8']
FMT = {'fp32': 'fp32', '16x8': 'q16x8', 'int8': 'int8'}


def f1_seed(v, d, s):   # F1 (persen)
    p = v.split('_')
    if 'std' in p:
        fam, fmt = p[0], p[2]
        if fmt == 'fp32': return 100 * (std1 if fam == '1ch' else pn)[s][d]['f1']
        return 100 * PT[fam][FMT[fmt]][str(s)]['kualitas'][d]['f1']
    fam, fmt = p; k = FMT[fmt]
    if k == 'int8': return 100 * S[fam][str(s)]['int8']['kualitas'][d]['f1']
    return 100 * (q3 if fam == '3ch' else q1)[s][d][k]['f1']


def cost(v, s):
    p = v.split('_')
    if 'std' in p:
        fam, fmt = p[0], p[2]; h = PD[fam][str(s)] if fmt == 'fp32' else PQ[f'{fam}_{fmt}'][str(s)]
    else: h = H[v][str(s)]
    return h['latency_ms'], h['energy_mj']


val = {}
for v in V12:
    for s in SEEDS:
        lat, en = cost(v, s)
        val[(v, s)] = dict(f1=float(np.mean([f1_seed(v, d, s) for d in DS])), f1_stead=f1_seed(v, 'stead_test', s), f1_lintas=float(np.mean([f1_seed(v, d, s) for d in DS[1:]])), lat=lat, en=en)
mean = {v: {k: float(np.mean([val[(v, s)][k] for s in SEEDS])) for k in ('f1', 'f1_stead', 'f1_lintas', 'lat', 'en')} for v in V12}


def nondominated(pts):
    return {a for a, (fa, la, ea) in pts.items() if not any((fb >= fa and lb <= la and eb <= ea) and (fb > fa or lb < la or eb < ea) for b, (fb, lb, eb) in pts.items() if b != a)}


tup = lambda d: (d['f1'], d['lat'], d['en'])
front = nondominated({v: tup(mean[v]) for v in V12}); front8 = nondominated({v: tup(mean[v]) for v in V6 + ['1ch_std_fp32', '3ch_std_fp32']})
cnt = {v: 0 for v in V12}
for s in SEEDS:
    for v in nondominated({v: tup(val[(v, s)]) for v in V12}): cnt[v] += 1
n = 3 ** len(V12); cnt_all = {v: 0 for v in V12}
for combo in itertools.product(SEEDS, repeat=len(V12)):
    for v in nondominated({V12[i]: tup(val[(V12[i], combo[i])]) for i in range(len(V12))}): cnt_all[v] += 1
NAME = {v: v for v in V12}
out = dict(nama=NAME, rerata=mean, per_seed={f'{v}_{s}': val[(v, s)] for (v, s) in val}, batas_12_rerata=sorted(front, key=V12.index), batas_8_dari_12_rerata=sorted(front8, key=V12.index),
           tak_terdominasi_seed_sama=cnt, n_kombinasi=n, proporsi_tak_terdominasi={v: cnt_all[v] / n for v in V12})
json.dump(out, open('results/final/pareto_dua_belas.json', 'w'), indent=1)
print('mean (F1 averaged over four datasets; STEAD F1; cross-domain F1; latency ms; energy mJ):')
for v in V12: m = mean[v]; print(f"  {v:13s} {m['f1']:6.2f} {m['f1_stead']:6.2f} {m['f1_lintas']:6.2f} {m['lat']:8.1f} {m['en']:8.1f}")
print('front (12) on the mean:', out['batas_12_rerata']); print('front (8, without quantized standard) on the mean:', out['batas_8_dari_12_rerata'])
print('non-dominated on same-seed (of 3):', cnt); print(f'proportion of {n} kombinasi (%):', {v: round(100 * cnt_all[v] / n, 1) for v in V12})

# ---- Pareto table for the summary (Indonesian format): ../ringkasan_hasil_pendamping_perangkat/tabel/tab_pareto.tex; ordered by latency like Figure 12; the dominance column = the dominating configuration with the highest F1 (then lowest latency), the same as 206_
idn = lambda x, nd: f'{x:,.{nd}f}'.replace(',', '§').replace('.', ',').replace('§', '.')
LAB = {'1ch_fp32': 'TinyPhaseNet satu kanal Float32', '1ch_16x8': 'TinyPhaseNet satu kanal 16x8', '1ch_int8': 'TinyPhaseNet satu kanal INT8', '3ch_fp32': 'TinyPhaseNet tiga kanal Float32', '3ch_16x8': 'TinyPhaseNet tiga kanal 16x8',
       '3ch_int8': 'TinyPhaseNet tiga kanal INT8', '1ch_std_fp32': 'PhaseNet satu kanal Float32', '1ch_std_16x8': 'PhaseNet satu kanal 16x8', '1ch_std_int8': 'PhaseNet satu kanal INT8',
       '3ch_std_fp32': 'PhaseNet tiga kanal Float32', '3ch_std_16x8': 'PhaseNet tiga kanal 16x8', '3ch_std_int8': 'PhaseNet tiga kanal INT8'}
dom = lambda a, b: (a['f1'] >= b['f1'] and a['lat'] <= b['lat'] and a['en'] <= b['en']) and (a['f1'] > b['f1'] or a['lat'] < b['lat'] or a['en'] < b['en'])
rows = []
for v in sorted(V12, key=lambda v: mean[v]['lat']):
    ds = sorted((o for o in V12 if o != v and dom(mean[o], mean[v])), key=lambda o: (-mean[o]['f1'], mean[o]['lat']))
    m = mean[v]; status = '\\textbf{tidak didominasi}' if not ds else LAB[ds[0]]
    pp = v.split('_'); hh = (PD[pp[0]] if pp[2:] == ['fp32'] else PQ[f'{pp[0]}_{pp[2]}']) if 'std' in pp else H[v]
    rows.append(f"{LAB[v]} & {idn(m['f1'], 1)} & {idn(m['lat'], 1)} & {idn(m['en'], 1)} & {idn(hh['rerata']['arena_kb'], 1)} ({'SRAM' if v.startswith('1ch') else 'PSRAM'}) & {status} \\\\")
import os
open(os.path.join('..', 'ringkasan_hasil_pendamping_perangkat', 'tabel', 'tab_pareto.tex'), 'w').write('\n'.join(rows) + '\n\\bottomrule\n')
print('\n'.join(rows))
