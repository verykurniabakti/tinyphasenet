"""
Step 234 (device manuscript, Q): synchronizes the results of the TWELVE ESP32-S3 configurations into draft_perangkat_q.tex and _id.tex.
Writes ../draft_publikasi_1/tabel_dev/{tab_device,numbers_dev12}{,_id}.tex and fig_dev_pareto{,_id}.pdf (12 configurations), and computes the selection rules R1-R3 on the 12 configurations (193_ for eight).
Sources: results/final/pareto_dua_belas.json (232_), hw_tiga_seed.json, hw_padrao_tiga_seed.json, hw_padrao_kuant_tiga_seed.json. Usage (from this folder): python 234_sinkron_draf_perangkat.py
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = 'results/final/'; OUT = '../draft_publikasi_1/tabel_dev/'
P = json.load(open(R + 'pareto_dua_belas.json')); H = json.load(open(R + 'hw_tiga_seed.json')); PD = json.load(open(R + 'hw_padrao_tiga_seed.json')); PQ = json.load(open(R + 'hw_padrao_kuant_tiga_seed.json'))
mean = P['rerata']; V6 = ['1ch_fp32', '1ch_16x8', '1ch_int8', '3ch_fp32', '3ch_16x8', '3ch_int8']; V12 = V6 + [f'{a}_std_{b}' for a in ('1ch', '3ch') for b in ('fp32', '16x8', 'int8')]; NDOM = set(P['batas_12_rerata'])
FM = {'fp32': 'Float32', '16x8': '16x8', 'int8': 'INT8'}
LANG = {'en': dict(name={'1ch': 'Single-channel', '3ch': 'Three-channel'}, std='standard', dec='.', thou=','), 'id': dict(name={'1ch': 'Satu kanal', '3ch': 'Tiga kanal'}, std='standar', dec=',', thou='.')}
num = lambda v, nd, L: f'{v:,.{nd}f}'.replace(',', '\u0001').replace('.', L['dec']).replace('\u0001', L['thou'])


def hw(v):
    p = v.split('_')
    if 'std' in p: return PD[p[0]] if p[2] == 'fp32' else PQ[f'{p[0]}_{p[2]}']
    return H[v]


def name(v, L):
    p = v.split('_'); fam, fmt = p[0], p[-1]
    return f"{L['name'][fam]} {L['std'] + ' ' if 'std' in p else ''}{FM[fmt]}"


# selection rules on 12 configurations (the same as 193_)
fb = sorted(NDOM, key=lambda v: mean[v]['en']); R1 = {}
for th in (90, 95, 97, 98):
    feas = [v for v in V12 if mean[v]['f1_stead'] >= th and mean[v]['f1_lintas'] >= 90]; R1[th] = min(feas, key=lambda v: mean[v]['en']) if feas else None
e = np.array([mean[v]['en'] for v in fb]); f = np.array([mean[v]['f1'] for v in fb]); en_ = (e - e.min()) / (e.max() - e.min()); fn_ = (f - f.min()) / (f.max() - f.min())
p0, p1 = np.array([en_[0], fn_[0]]), np.array([en_[-1], fn_[-1]]); d = [abs(np.cross(p1 - p0, np.array([x, y]) - p0)) / np.linalg.norm(p1 - p0) for x, y in zip(en_, fn_)]; R2 = fb[int(np.argmax(d))]
e = np.array([mean[v]['en'] for v in V12]); f = np.array([mean[v]['f1'] for v in V12]); en_ = (e - e.min()) / (e.max() - e.min()); fn_ = (f - f.min()) / (f.max() - f.min())
win = [(round(float(w), 2), V12[int(np.argmax(w * fn_ - (1 - w) * en_))]) for w in np.linspace(0, 1, 21)]; iv = []
for w, nm_ in win:
    if iv and iv[-1][2] == nm_: iv[-1][1] = w
    else: iv.append([w, w, nm_])
print('front', fb); print('R1', R1); print('R2', R2, [round(float(x), 3) for x in d]); print('R3', iv)

for lg, L in LANG.items():
    sfx = '' if lg == 'en' else '_id'; n = lambda v, nd=1: num(v, nd, L); rows = ''
    for i, v in enumerate(V12):
        h = hw(v); r, s = h['rerata'], h['sd']; loc = 'SRAM' if v.startswith('1ch') else 'PSRAM'
        rows += f"{name(v, L)} & {n(r['arena_kb'])} ({loc}) & {n(r['latency_ms'])} $\\pm$ {n(s['latency_ms'])} & {n(r['energy_mj'])} $\\pm$ {n(s['energy_mj'])} & {n(100 * r['latency_ms'] / 30720)} & {n(100 * r['f1_100'])} $\\pm$ {n(100 * s['f1_100'])} \\\\\n"
        if i in (5,): rows += '\\midrule\n'
    open(OUT + f'tab_device{sfx}.tex', 'w').write(rows + '\\bottomrule\n')
    X = lambda a, b: n(hw(a)['rerata']['latency_ms'] / hw(b)['rerata']['latency_ms']); XE = lambda a, b: n(hw(a)['rerata']['energy_mj'] / hw(b)['rerata']['energy_mj'])
    M = {}
    for v in V12[6:]:
        t = {'1ch': 'One', '3ch': 'Thr'}[v[:3]] + {'fp32': 'Fp', '16x8': 'Qq', 'int8': 'Int'}[v[-4:] if v.endswith('fp32') or v.endswith('int8') else '16x8']
        r, s = hw(v)['rerata'], hw(v)['sd']
        M.update({f'dvPn{t}Lat': n(r['latency_ms']), f'dvPn{t}En': n(r['energy_mj']), f'dvPn{t}Ar': n(r['arena_kb']), f'dvPn{t}Duty': n(100 * r['latency_ms'] / 30720), f'dvPn{t}Ef': n(100 * r['f1_100']), f'dvPn{t}Fm': n(mean[v]['f1'])})
    M.update({'dvPnIntFpOneX': X('1ch_std_fp32', '1ch_std_int8'), 'dvPnIntFpThrX': X('3ch_std_fp32', '3ch_std_int8'), 'dvPnQqFpThrX': X('3ch_std_16x8', '3ch_std_fp32'), 'dvPnQqFpOneX': X('1ch_std_16x8', '1ch_std_fp32'),
              'dvPnXIntOne': X('1ch_std_int8', '1ch_int8'), 'dvPnXIntThr': X('3ch_std_int8', '3ch_int8'), 'dvPnXFpOne': X('1ch_std_fp32', '1ch_fp32'), 'dvPnXFpThr': X('3ch_std_fp32', '3ch_fp32'),
              'dvPnXQqOne': X('1ch_std_16x8', '1ch_16x8'), 'dvPnXQqThr': X('3ch_std_16x8', '3ch_16x8'), 'dvTinyIntFpOneX': X('1ch_int8', '1ch_fp32'),
              'dvPnEfMin': n(100 * min(hw(v)['rerata']['f1_100'] for v in V12[6:])), 'dvPnDutyMax': n(max(100 * hw(v)['rerata']['latency_ms'] / 30720 for v in V12)), 'dvPnArThrQqMin': n(min(PQ['3ch_16x8'][s]['arena_kb'] for s in ('42', '43', '44')), 0), 'dvPnArThrQqMax': n(max(PQ['3ch_16x8'][s]['arena_kb'] for s in ('42', '43', '44')), 0),
              'dvPwMin': n(min(mean[v]['en'] / mean[v]['lat'] * 1000 for v in V12), 0), 'dvPwMax': n(max(mean[v]['en'] / mean[v]['lat'] * 1000 for v in V12), 0)})
    M.update({'dvParNCombTwelve': num(P['n_kombinasi'], 0, L), 'dvPropPnOneFp': n(100 * P['proporsi_tak_terdominasi']['1ch_std_fp32'], 0), 'dvPropPnThrFp': n(100 * P['proporsi_tak_terdominasi']['3ch_std_fp32'], 0)})
    open(OUT + f'numbers_dev12{sfx}.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in M.items()))

plt.rcParams['font.family'] = 'sans-serif'; plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
INK = '#0f172a'; C1, C3 = '#0f766e', '#c2410c'
TXT = {'en': dict(lat='Latency (ms per window)', en='Energy (mJ per window)', f1='Mean F1-score, four datasets (%)', one='Single-channel', thr='Three-channel', nd='Non-dominated', dom='Dominated', std='Std '),
       'id': dict(lat='Latensi (ms per jendela)', en='Energi (mJ per jendela)', f1='F1-score rerata empat dataset (%)', one='Satu kanal', thr='Tiga kanal', nd='Tidak terdominasi', dom='Terdominasi', std='Std ')}
MK = {'fp32': 'o', '16x8': 's', 'int8': '^'}
for lg, T in TXT.items():
    sfx = '' if lg == 'en' else '_id'
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), dpi=300)
    for ax, (key, xl) in zip(axes, (('en', T['en']), ('lat', T['lat']))):
        for v in V12:
            p = v.split('_'); fam, fmt = p[0], p[-1]; col = C1 if fam == '1ch' else C3; std = 'std' in p; filled = v in NDOM
            ax.scatter(mean[v][key], mean[v]['f1'], marker=MK[fmt], s=44, facecolor=col if filled else 'white', edgecolor=col, linewidth=1.6 if std else 1.0, zorder=3)
            dx, dy, ha_ = {'1ch_std_int8': (-4, -9, 'right')}.get(v, (4, 3, 'left'))
            ax.annotate((T['std'] if std else '') + FM[fmt], (mean[v][key], mean[v]['f1']), textcoords='offset points', xytext=(dx, dy), ha=ha_, fontsize=5.6, color=INK)
        ax.set_xscale('log'); ax.set_xlabel(xl, fontsize=7.6); ax.set_ylabel(T['f1'] if ax is axes[0] else '', fontsize=7.6); ax.tick_params(labelsize=6.8); ax.grid(linestyle='--', alpha=0.35, zorder=0)
        for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
    hs = [Line2D([0], [0], marker='o', color='w', markerfacecolor=C1, markeredgecolor=C1, markersize=6, label=T['one']), Line2D([0], [0], marker='o', color='w', markerfacecolor=C3, markeredgecolor=C3, markersize=6, label=T['thr']),
          Line2D([0], [0], marker='o', color='w', markerfacecolor=INK, markeredgecolor=INK, markersize=6, label=T['nd']), Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor=INK, markersize=6, label=T['dom'])]
    fig.legend(handles=hs, loc='upper center', ncol=4, fontsize=7, frameon=False); plt.tight_layout(rect=[0, 0, 1, 0.91]); fig.savefig(OUT + f'fig_dev_pareto{sfx}.pdf', bbox_inches='tight'); plt.close(fig)
print('ok')
