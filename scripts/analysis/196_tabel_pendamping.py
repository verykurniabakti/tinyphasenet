"""
Langkah 196 (draf pendamping kompresi): tabel dan figur perbandingan proses depthwise separable dan kuantisasi pascapelatihan untuk model satu kanal dan tiga kanal, dari hasil tersimpan (tanpa percobaan baru).
Semua selisih BERPASANGAN per seed (seed 42, 43, 44) lalu dirata-ratakan (rerata +- simpangan baku sampel):
  - langkah 1: depthwise separable Float32 dikurangi konvolusi standar Float32 (model standar: eval_standard.json satu kanal, eval_pn.json tiga kanal);
  - langkah 2: format terkuantisasi (8/16, 8/8, 4/8 emulasi dengan penjepitan MSE per kanal) dikurangi Float32 dari model depthwise separable yang sama.
F1 dalam persen, selisih dalam poin persentase. Dataset: STEAD, INSTANCE, ETHZ, TXED (uji).
Sumber: results/final/{spektrum_kuantisasi_3seed,w4a8_perkanal_3seed,quant16x8_per_seed,quant16x8_pndw_per_seed}.json, results/ablation_final/eval_{standard,pn}.json.
Keluaran: ../draft_publikasi_1/tabel_pend/{tab_pend_step1,tab_pend_step2}{,_id}.tex, numbers_pend{,_id}.tex, fig_pend_delta{,_id}.pdf, results/final/pendamping_selisih.json.
Pemakaian (dari folder ini): python 196_tabel_pendamping.py
"""
import json, os, statistics as st
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '../draft_publikasi_1/tabel_pend/'; os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 43, 44]; DS = ['stead_test', 'instance_test', 'ethz_test', 'txed_test']; DSN = ['STEAD', 'INSTANCE', 'ETHZ', 'TXED']
S = json.load(open('results/final/spektrum_kuantisasi_3seed.json')); W = json.load(open('results/final/w4a8_perkanal_3seed.json'))
q1 = {int(k): v for k, v in json.load(open('results/final/quant16x8_per_seed.json')).items()}; q3 = {int(k): v for k, v in json.load(open('results/final/quant16x8_pndw_per_seed.json')).items()}
std = {int(e['model'].split('seed')[1].split('.')[0]): e for e in json.load(open('results/ablation_final/eval_standard.json'))}
pn = {int(e['model'].split('seed')[1].split('.')[0]): e for e in json.load(open('results/ablation_final/eval_pn.json'))}
FAM = [('1ch', 'Single-channel', q1, std), ('3ch', 'Three-channel', q3, pn)]


def f1(fam, key, d, s):
    q = q3 if fam == '3ch' else q1
    if key == 'standard': return 100 * (pn if fam == '3ch' else std)[s][d]['f1']
    if key == 'fp32': return 100 * q[s][d]['fp32']['f1']
    if key == 'q16x8': return 100 * q[s][d]['q16x8']['f1']
    if key == 'int8': return 100 * S[fam][str(s)]['int8']['kualitas'][d]['f1']
    return 100 * W[fam][str(s)]['pcmse']['kualitas'][d]['f1']


def delta(fam, a, b, d=None):   # selisih berpasangan per seed: F1(a) - F1(b); d=None -> rerata empat dataset
    if d is None: return [st.mean(f1(fam, a, dd, s) - f1(fam, b, dd, s) for dd in DS) for s in SEEDS]
    return [f1(fam, a, d, s) - f1(fam, b, d, s) for s in SEEDS]


def pm(xs, nd=1, dec='.'):
    m = st.mean(xs); txt = '0.0' if abs(m) < 0.05 else f"{m:+.{nd}f}".replace('-', '$-$')   # tanpa tanda pada nol hasil pembulatan
    return f"{txt} $\\pm$ {st.stdev(xs):.{nd}f}".replace('.', dec)


res = {}
NMID = {'1ch': 'Kanal tunggal', '3ch': 'Tiga kanal'}
for fam, nm, _, _ in FAM:
    res[f'{fam}_passo1'] = {d: delta(fam, 'fp32', 'standard', d) for d in DS}; res[f'{fam}_passo1']['mean4'] = delta(fam, 'fp32', 'standard')
for sfx, dec, NM in (('', '.', {f: n for f, n, _, _ in FAM}), ('_id', ',', NMID)):
    r1 = ''
    for fam, _, _, _ in FAM:
        r1 += f"{NM[fam]} & " + ' & '.join([pm(delta(fam, 'fp32', 'standard', d), 1, dec) for d in DS] + [pm(delta(fam, 'fp32', 'standard'), 1, dec)]) + ' \\\\\n'
    open(OUT + f'tab_pend_step1{sfx}.tex', 'w').write(r1 + '\\bottomrule\n')
FMT = [('q16x8', '8/16'), ('int8', '8/8'), ('w4a8', '4/8 (emulated)')]
for fam, nm, _, _ in FAM:
    for key, fn in FMT:
        res[f'{fam}_{key}'] = {d: delta(fam, key, 'fp32', d) for d in DS}; res[f'{fam}_{key}']['mean4'] = delta(fam, key, 'fp32')
for sfx, dec, NM, FN in (('', '.', {f: n for f, n, _, _ in FAM}, {'q16x8': '8/16', 'int8': '8/8', 'w4a8': '4/8 (emulated)'}), ('_id', ',', NMID, {'q16x8': '8/16', 'int8': '8/8', 'w4a8': '4/8 (emulasi)'})):
    r2 = ''
    for fam, _, _, _ in FAM:
        for key, _ in FMT:
            r2 += f"{NM[fam]} {FN[key]} & " + ' & '.join([pm(delta(fam, key, 'fp32', d), 1, dec) for d in DS] + [pm(delta(fam, key, 'fp32'), 1, dec)]) + ' \\\\\n'
        if fam == '1ch': r2 += '\\midrule\n'
    open(OUT + f'tab_pend_step2{sfx}.tex', 'w').write(r2 + '\\bottomrule\n')

# makro angka untuk teks
mx = lambda fam, key: [abs(st.mean(res[f'{fam}_{key}'][d])) for d in DS]
M = {'PdQqMax': max(max(mx(f, 'q16x8')) for f in ('1ch', '3ch'))}
for f, t in (('1ch', 'One'), ('3ch', 'Thr')):
    for key, kt in (('int8', 'Int'), ('w4a8', 'Four')):
        v = mx(f, key); M[f'Pd{kt}{t}Min'] = min(v); M[f'Pd{kt}{t}Max'] = max(v)
    M[f'PdShift{t}'] = max(st.mean(f1(f, 'fp32', 'stead_test', s) - f1(f, 'fp32', 'instance_test', s) for s in SEEDS), st.mean(f1(f, 'fp32', 'stead_test', s) - f1(f, 'fp32', 'txed_test', s) for s in SEEDS))
for f, t in (('1ch', 'One'), ('3ch', 'Thr')):   # simpangan baku selisih per dataset (INT8 dan 4/8): rentang
    sds = [st.stdev(res[f'{f}_{k}'][d]) for k in ('int8', 'w4a8') for d in DS]; M[f'PdSd{t}Min'] = min(sds); M[f'PdSd{t}Max'] = max(sds)
for sfx, dec in (('', '.'), ('_id', ',')): open(OUT + f'numbers_pend{sfx}.tex', 'w').write(''.join(f'\\newcommand{{\\dv{k}}}{{{v:.1f}}}\n'.replace('.', dec) if dec == ',' and False else f'\\newcommand{{\\dv{k}}}{{{(f"{v:.1f}").replace(".", dec)}}}\n' for k, v in M.items()))
res['makros'] = M; json.dump(res, open('results/final/pendamping_selisih.json', 'w'), indent=1)

# figur: perubahan F1 rerata empat dataset pada tiap langkah, satu kanal dan tiga kanal, rerata +- SD atas tiga seed (Inggris dan Indonesia)
C1, C3, INK = '#0f766e', '#c2410c', '#0f172a'
LANG = {'': dict(cats=[('Step 1\nstandard $\\rightarrow$ separable', 'passo1'), ('Step 2\n8/16', 'q16x8'), ('Step 2\n8/8', 'int8'), ('Step 2\n4/8 (emulated)', 'w4a8')], y='Change in mean F1-score,\nfour datasets (points)', fam={'1ch': 'Single-channel', '3ch': 'Three-channel'}, dec='.'),
        '_id': dict(cats=[('Langkah 1\nstandar $\\rightarrow$ separable', 'passo1'), ('Langkah 2\n8/16', 'q16x8'), ('Langkah 2\n8/8', 'int8'), ('Langkah 2\n4/8 (emulasi)', 'w4a8')], y='Perubahan F1-score rerata,\nempat dataset (poin)', fam={'1ch': 'Kanal tunggal', '3ch': 'Tiga kanal'}, dec=',')}
for sfx, L in LANG.items():
    cats = L['cats']; fig, ax = plt.subplots(figsize=(7.4, 3.3), dpi=300); x = np.arange(len(cats)); w = 0.36
    for off, (fam, col) in ((-w / 2, ('1ch', C1)), (w / 2, ('3ch', C3))):
        mu = [st.mean(res[f'{fam}_{k}']['mean4']) for _, k in cats]; sd = [st.stdev(res[f'{fam}_{k}']['mean4']) for _, k in cats]
        ax.bar(x + off, mu, w, yerr=sd, color=col, edgecolor=INK, linewidth=0.5, capsize=2.2, error_kw=dict(lw=0.8, ecolor=INK), label=L['fam'][fam], zorder=3)
        for xi, m, sdv in zip(x + off, mu, sd): ax.text(xi, 0.5 if m < 0 else m + sdv + 0.3, f"{m:+.1f}".replace('-', '\u2212').replace('.', L['dec']), ha='center', va='bottom', fontsize=6.8, color=INK)   # nilai di atas garis nol agar tidak menimpa batang galat
    ax.axhline(0, color=INK, linewidth=0.8); ax.set_xticks(x); ax.set_xticklabels([c for c, _ in cats], fontsize=7)
    ax.set_ylabel(L['y'], fontsize=7.6); ax.tick_params(axis='y', labelsize=7); ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc='lower left', bbox_to_anchor=(0.0, 0.02)); ax.set_ylim(-29, 3.6)
    plt.tight_layout(); fig.savefig(OUT + f'fig_pend_delta{sfx}.pdf', bbox_inches='tight'); plt.close(fig)
print(open(OUT + 'tab_pend_step1.tex').read()); print(open(OUT + 'tab_pend_step2.tex').read()); print(open(OUT + 'numbers_pend.tex').read())
