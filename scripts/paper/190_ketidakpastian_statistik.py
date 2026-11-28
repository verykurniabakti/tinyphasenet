"""
Step 190 (device manuscript, Q, methodology improvement stage 2): statistical uncertainty from the existing data, with no new experiment.
(a) On-device F1 (100 STEAD windows: 50 earthquake + 50 noise) per configuration and seed, with a 95% confidence interval from a stratified bootstrap
    (TP from the 50 earthquakes, FP from the 50 noise windows, each stratum resampled binomially from its proportion; B = 10000). The interval of the mean of three seeds
    uses a separate bootstrap per seed and then averages (the same windows are used by the three seeds, so this independence assumption
    probably does not add width; recorded as an approximation).
(b) Robustness of the Pareto analysis to the seed: for the six measured configurations (one channel and three channels; Float32, 16x8, INT8) each configuration
    takes one seed (42, 43, or 44) for the F1 averaged over four datasets, the latency, and the energy; all 3^6 = 729 combinations are computed, and the
    proportion of combinations in which each configuration is not dominated is reported. Also computed for the three same-seed combinations and for the mean of three seeds.
Sources: results/final/{spektrum_kuantisasi_3seed,w4a8_perkanal_3seed,quant16x8_per_seed,quant16x8_pndw_per_seed,hw_tiga_seed}.json, results/ablation_final/eval_*.json,
firmware/tinyphasenet_profiling/results_hw/*.log (TP/TN/FP/FN). Output: results/final/ketidakpastian_statistik.json, ../draft_publikasi_1/tabel_dev/numbers_unc{,_id}.tex (number macros), and a summary on screen.
Usage (from this folder): python 190_ketidakpastian_statistik.py
"""
import json, re, itertools
import numpy as np

SEEDS = [42, 43, 44]; DS = ['stead_test', 'instance_test', 'ethz_test', 'txed_test']
S = json.load(open('results/final/spektrum_kuantisasi_3seed.json'))
q3 = {int(k): v for k, v in json.load(open('results/final/quant16x8_pndw_per_seed.json')).items()}
q1 = {int(k): v for k, v in json.load(open('results/final/quant16x8_per_seed.json')).items()}
H = json.load(open('results/final/hw_tiga_seed.json'))
RH = 'firmware/tinyphasenet_profiling/results_hw/'
V = ['1ch_fp32', '1ch_16x8', '1ch_int8', '3ch_fp32', '3ch_16x8', '3ch_int8']


def f1_seed(fam, key, d, s):   # F1 (percent) of one seed, one dataset
    if key == 'fp32': return 100 * (q3 if fam == '3ch' else q1)[s][d]['fp32']['f1']
    if key == 'q16x8': return 100 * (q3 if fam == '3ch' else q1)[s][d]['q16x8']['f1']
    return 100 * S[fam][str(s)]['int8']['kualitas'][d]['f1']


def logpath(v, s):
    if v == '1ch_fp32': return RH + f'sesi_07nov_seed{s}_1ch_fp32_sram_stead.log'
    if s == 42: return RH + (f'sesi_07nov_float32_fastpath_{v[:3]}_stead.log' if v.endswith('fp32') else f'sesi_06nov_kernel_baru_{v}_stead.log')
    return RH + f'sesi_07nov_seed{s}_{v}_stead.log'


def confusion(path):
    t = open(path, errors='ignore').read().replace('\r', '')
    m = re.search(r'TP:?\s*(\d+)\D+TN:?\s*(\d+)\D+FP:?\s*(\d+)\D+FN:?\s*(\d+)', t)
    return tuple(int(x) for x in m.groups())


# ---------------- (a) confidence interval of the on-device F1
rng = np.random.default_rng(42); B = 10000
out = {'f1_perangkat': {}, 'pareto_seed': {}}
print('On-device F1, 100 windows (95% interval, stratified bootstrap):')
for v in V:
    boots, per = [], {}
    for s in SEEDS:
        tp, tn, fp, fn = confusion(logpath(v, s)); npos, nneg = tp + fn, tn + fp
        a = rng.binomial(npos, tp / npos, B); b = rng.binomial(nneg, fp / nneg, B)
        f = 100 * 2 * a / (2 * a + b + (npos - a)); boots.append(f)
        per[s] = dict(tp=tp, tn=tn, fp=fp, fn=fn, f1=100 * 2 * tp / (2 * tp + fp + fn), lo=float(np.percentile(f, 2.5)), hi=float(np.percentile(f, 97.5)))
    m = np.mean(boots, axis=0); out['f1_perangkat'][v] = dict(per_seed=per, rerata=float(np.mean([per[s]['f1'] for s in SEEDS])), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)))
    r = out['f1_perangkat'][v]
    print(f"  {v:9s} seed 42/43/44: " + ' / '.join(f"{per[s]['f1']:.1f} [{per[s]['lo']:.1f}, {per[s]['hi']:.1f}]" for s in SEEDS) + f"   mean {r['rerata']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}]")

# ---------------- (b) ketahanan Pareto terhadap seed
KEY = {'fp32': 'fp32', '16x8': 'q16x8', 'int8': 'int8'}
val = {}
for v in V:
    fam, fmt = v.split('_'); k = KEY[fmt]
    for s in SEEDS:
        val[(v, s)] = (np.mean([f1_seed(fam, k, d, s) for d in DS]), H[v][str(s)]['latency_ms'], H[v][str(s)]['energy_mj'])


def nondominated(pts):   # pts: {nama: (f1, lat, en)}
    nd = set()
    for a, (fa, la, ea) in pts.items():
        dom = any((fb >= fa and lb <= la and eb <= ea) and (fb > fa or lb < la or eb < ea) for b, (fb, lb, eb) in pts.items() if b != a)
        if not dom: nd.add(a)
    return nd


mean_pts = {v: tuple(np.mean([val[(v, s)][i] for s in SEEDS]) for i in range(3)) for v in V}
nd_mean = nondominated(mean_pts)
cnt = {v: 0 for v in V}; n = 0
for combo in itertools.product(SEEDS, repeat=len(V)):
    for v in nondominated({V[i]: val[(V[i], combo[i])] for i in range(len(V))}): cnt[v] += 1
    n += 1
cnt_same = {v: 0 for v in V}
for s in SEEDS:
    for v in nondominated({v: val[(v, s)] for v in V}): cnt_same[v] += 1
out['pareto_seed'] = dict(n_kombinasi=n, nondominasi_rerata=sorted(nd_mean), proporsi_729={v: cnt[v] / n for v in V}, tak_terdominasi_seed_sama={v: cnt_same[v] for v in V},
                          rerata_f1_mean4={v: mean_pts[v][0] for v in V})
print(f'\nPareto (F1 averaged over four datasets, latency, energy) -- non-dominated on the mean of three seeds: {sorted(nd_mean)}')
print(f'proportion of {n} seed-per-configuration combinations (one seed chosen for each configuration) in which each configuration is not dominated:')
for v in V: print(f'  {v:9s} {100 * cnt[v] / n:5.1f}%   same-seed (of 3): {cnt_same[v]}   F1 mean {mean_pts[v][0]:.2f}')
json.dump(out, open('results/final/ketidakpastian_statistik.json', 'w'), indent=1)

# ---------------- number macros for the manuscript (English: decimal point; Indonesian: decimal comma)
F = out['f1_perangkat']; ex = F['1ch_fp32']['per_seed'][42]
M = {'UncNComb': (n, 0), 'UncShareOneQq': (100 * cnt['1ch_16x8'] / n, 0), 'UncShareThrQq': (100 * cnt['3ch_16x8'] / n, 0), 'UncShareThrInt': (100 * cnt['3ch_int8'] / n, 0),
     'CiSeedF': (ex['f1'], 1), 'CiSeedLo': (ex['lo'], 1), 'CiSeedHi': (ex['hi'], 1),
     'CiOneFpLo': (F['1ch_fp32']['lo'], 1), 'CiOneFpHi': (F['1ch_fp32']['hi'], 1), 'CiOneIntLo': (F['1ch_int8']['lo'], 1), 'CiOneIntHi': (F['1ch_int8']['hi'], 1)}
for sfx, dec in (('', '.'), ('_id', ',')):
    open(f'../draft_publikasi_1/tabel_dev/numbers_unc{sfx}.tex', 'w').write(''.join(f'\\newcommand{{\\dv{k}}}{{{v:.{d}f}}}\n'.replace('.', dec) if False else f'\\newcommand{{\\dv{k}}}{{{(f"{v:.{d}f}").replace(".", dec)}}}\n' for k, (v, d) in M.items()))
print('\nok -> results/final/ketidakpastian_statistik.json')
