"""
Step 188 (device manuscript, Q): compression-trace figure, English and Indonesian.
For each model family (1 channel and 3 channels): F1 averaged over four datasets against file size, following the two compression steps:
standard-convolution Float32 -> depthwise separable Float32 (step 1, trained from scratch) -> 16x8, full INT8, 4/8 (step 2, post-training quantization; 4/8 = accuracy emulation).
Source: results/final/pareto_kandidat.json (164_; mean of three seeds; no hand-typed numbers).
Output: ../draft_publikasi_1/tabel_dev/fig_dev_dw.pdf and fig_dev_dw_id.pdf
Usage (from this folder): python 188_gambar_depthwise.py
"""
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = '../draft_publikasi_1/tabel_dev/'; os.makedirs(OUT, exist_ok=True)
P = {(c['fam'], c['key']): c for c in json.load(open('results/final/pareto_kandidat.json'))['kandidat']}
INK, GREY, BLUE = '#0f172a', '#475569', '#1d4ed8'   # step 1 = grey, step 2 = blue (the same in both panels)
ORDER = [('standard', 'D'), ('fp32', 'o'), ('q16x8', 's'), ('int8', '^'), ('w4a8', 'v')]
KEY = {'standard': 'standard', 'fp32': 'fp32', 'q16x8': 'q16x8', 'int8': 'int8', 'w4a8': 'w4a8'}
LANG = {'en': dict(f1='Mean F1-score, four datasets (%)', x='File size (KB, log scale)', one='(a) Single-channel', thr='(b) Three-channel',
                   lab={'standard': 'Standard conv.', 'fp32': 'Depthwise sep.\nFloat32', 'q16x8': '16x8', 'int8': 'INT8', 'w4a8': '4/8 (emulated)'},
                   s1='step 1: depthwise separable', s2='step 2: post-training quantization', ph="Single-channel", pt="Three-channel"),
        'id': dict(f1='F1-score rerata, empat dataset (%)', x='Ukuran berkas (KB, skala log)', one='(a) Kanal tunggal', thr='(b) Tiga kanal',
                   lab={'standard': 'Konv. standar', 'fp32': 'Depthwise sep.\nFloat32', 'q16x8': '16x8', 'int8': 'INT8', 'w4a8': '4/8 (emulasi)'},
                   s1='langkah 1: depthwise separable', s2='langkah 2: kuantisasi pascapelatihan', ph='1 kanal', pt='3 kanal')}
FAM_KEY = {'1ch': {'standard': 'standard'}, '3ch': {'standard': 'pn'}}   # standard-convolution counterpart of each family


def pt(fam, name):
    return P[(fam, FAM_KEY[fam].get(name, name))]


for lg in ('en', 'id'):
    L = LANG[lg]; sfx = '' if lg == 'en' else '_id'
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), dpi=300)
    for ax, fam, title, col, ticks in zip(axes, ('1ch', '3ch'), (L['one'], L['thr']), (BLUE, BLUE), ([60, 100, 200, 300], [100, 200, 500, 1000])):
        xs = [pt(fam, n)['flash_kb'] for n, _ in ORDER]; ys = [pt(fam, n)['f1_mean4'] for n, _ in ORDER]
        ax.plot(xs[:2], ys[:2], color=GREY, linewidth=1.2, zorder=2)                 # step 1
        ax.plot(xs[1:], ys[1:], color=col, linewidth=1.2, zorder=2)                  # step 2
        for (n, mk), x, y in zip(ORDER, xs, ys):
            fill = (GREY if n == 'standard' else col) if n != 'w4a8' else 'white'
            ax.scatter(x, y, marker=mk, s=46, facecolor=fill, edgecolor=GREY if n == 'standard' else col, linewidth=1.3, zorder=3)
        ymin, ymax = min(ys), max(ys); pad = (ymax - ymin) * 0.16
        ax.set_ylim(ymin - pad * 1.6, ymax + pad * 1.4)
        for (n, _), x, y in zip(ORDER, xs, ys):
            dx, dy, ha, va = {'standard': (-4, 6, 'right', 'baseline'), 'fp32': (6, -6, 'left', 'top'), 'q16x8': (-5, -13, 'right', 'baseline'),
                              'int8': (6, 4, 'left', 'baseline'), 'w4a8': (7, -4, 'left', 'top')}[n]   # fp32 lower right: the line does not cross the text
            ax.annotate(L['lab'][n], (x, y), textcoords='offset points', xytext=(dx, dy), fontsize=6.3, color=INK, ha=ha, va=va)
        ax.set_xscale('log'); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off(); ax.set_xlabel(L['x'], fontsize=7.6); ax.set_ylabel(L['f1'] if fam == '1ch' else '', fontsize=7.6)
        ax.set_title(title, fontsize=8.2, fontweight='bold', pad=4); ax.tick_params(labelsize=6.8); ax.grid(linestyle='--', alpha=0.35, zorder=0)
        for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
    hs = [Line2D([0], [0], color=GREY, linewidth=1.4, label=L['s1']), Line2D([0], [0], color=BLUE, linewidth=1.4, label=L['s2'])]
    fig.legend(handles=hs, loc='upper center', ncol=2, fontsize=7, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(OUT + f'fig_dev_dw{sfx}.pdf', bbox_inches='tight'); plt.close(fig)
print('ok')
