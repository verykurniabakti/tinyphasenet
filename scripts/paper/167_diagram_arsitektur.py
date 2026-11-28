"""
Step 167 (answer to the advisor, Question 1): architecture diagram. (a) replacement of the standard convolution block with depthwise separable; (b) PhaseNet/depthwise PhaseNet (three channels); (c) one-channel model.
The tensor shapes are read from the final Keras models (s10_w1_seed42.keras and pndw_seed42.keras), not typed by hand. Output: ../jawaban_prof_arief_fase1/grafik_arsitektur.pdf/png
Usage: python 167_diagram_arsitektur.py            (Indonesian labels -> ../jawaban_prof_arief_fase1/grafik_arsitektur.pdf/png)
       python 167_diagram_arsitektur.py en | id    (manuscript -> ../draft_publikasi_1/tabel_dev/fig_dev_arch{,_id}.pdf)
"""
import os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import tensorflow as tf

MODE = sys.argv[1] if len(sys.argv) > 1 else 'jawaban'
OUT = '../jawaban_prof_arief_fase1/' if MODE == 'jawaban' else '../draft_publikasi_1/tabel_dev/'
STEM = 'grafik_arsitektur' if MODE == 'jawaban' else ('fig_dev_arch' if MODE == 'en' else 'fig_dev_arch_id')
TX = {'en': dict(ta='(a) Replacement of the convolution block (kernel 7; Cin = input channels, Cout = output channels)', std='Standard convolution\n(original PhaseNet)\nCin × Cout × 7 weights', rep='replaced by', dw='Depthwise 7×1\nCin × 7 weights', pw='Pointwise 1×1\nCin × Cout weights',
              note='Transposed-convolution layers (decoder path) and the 1×1 output convolution are not replaced.', tb='(b) 3-channel U-Net (5 levels)', tc='(c) TinyPhaseNet (1 channel, 4 levels)', inp='Input', out='Output', bott='bottleneck',
              l1='stride-4 convolution (down)', l2='stride-4 transposed convolution (up)', l3='skip connection (concatenated)'),
      'id': dict(ta='(a) Penggantian blok konvolusi (kernel 7; Cin = kanal masuk, Cout = kanal keluar)', std='Konvolusi standar\n(PhaseNet asli)\nCin × Cout × 7 bobot', rep='diganti', dw='Depthwise 7×1\nCin × 7 bobot', pw='Pointwise 1×1\nCin × Cout bobot',
              note='Lapisan konvolusi transpos (jalur naik) dan konvolusi keluaran 1×1 tidak diganti.', tb='(b) U-Net tiga kanal (5 tingkat)', tc='(c) TinyPhaseNet (1 kanal, 4 tingkat)', inp='Masukan', out='Keluaran', bott='dasar',
              l1='konvolusi langkah 4 (turun)', l2='konvolusi transpos langkah 4 (naik)', l3='sambungan lompatan (digabung)')}
T = TX['en' if MODE == 'en' else 'id']
if MODE == 'jawaban': T = dict(T, tb='(b) PhaseNet / PhaseNet depthwise (3 kanal, 5 tingkat)', tc='(c) Model 1-Kanal (4 tingkat)')
def shapes(path):
    m = tf.keras.models.load_model(path, compile=False); enc, bott, dec = [], None, []
    seq = [(type(l).__name__, tuple(l.output.shape[1:]), l.get_config().get('strides', (1,))[0] if hasattr(l, 'strides') else 1, l.name) for l in m.layers if type(l).__name__ in ('SeparableConv1D', 'Conv1DTranspose', 'Concatenate', 'Conv1D', 'InputLayer')]
    return m, seq
m1, s1 = shapes('models/svariants/s10_w1_seed42.keras'); m3, s3 = shapes('models/ablation_final/pndw_seed42.keras')
# encoder level = output of the stride-1 convolution followed by the strided convolution, or the strided convolution in the one-channel model
def levels_1ch():
    enc = [s[1] for s in s1 if s[0] == 'SeparableConv1D' and s[2] == 4]          # 768x8,192x16,48x32,12x64
    bott = [s[1] for s in s1 if s[0] == 'SeparableConv1D' and s[2] == 1][0]       # 12x64
    dec = [s[1] for s in s1 if s[0] == 'SeparableConv1D' and s[2] == 1][1:]       # 48x32,192x16,768x8
    last = [s[1] for s in s1 if s[0] == 'Conv1DTranspose'][-1]                     # 3072x8
    return [e for e in enc[:-1]], enc[-1], dec, last, s1[0][1], [s for s in s1 if s[0] == 'Conv1D'][-1][1]
def levels_3ch():
    d1 = [s[1] for s in s3 if s[3].startswith('down_conv1_')]                      # 3072x8 ... 12x128
    dec = [s[1] for s in s3 if s[3].startswith('up_conv1_')]                       # the layer order is from the deepest: 48x64 ... 3072x8
    return d1[:-1], d1[-1], dec, s3[0][1], [s for s in s3 if s[0] == 'Conv1D'][-1][1]

C_STD, C_SEP, C_TR, C_BN = '#e2e8f0', '#99f6e4', '#fde68a', '#fecaca'
def box(ax, x, y, w, h, txt, fc, fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec='#0f172a', lw=0.8, zorder=3))
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', zorder=4)
def arr(ax, a, b, style='-|>', ls='-', color='#0f172a', rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=7, lw=0.8, ls=ls, color=color, zorder=2, connectionstyle=f'arc3,rad={rad}'))
fmt = lambda s: f'{s[0]}×{s[1]}'

def unet(ax, title, inp, enc, bott, dec, out, last=None, notes=None):
    n = len(enc); W = 2.1; H = 0.5; HIO = 0.82; dx = 1.35; YT = 1.95   # HIO: height of the input/output boxes (two lines of text)
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=3)
    box(ax, 0, YT, W, HIO, f"{T['inp']}\n{fmt(inp)}", '#ffffff', 8.2)
    xs_e = [i * dx for i in range(n)]; ys = [-i * 0.78 for i in range(n)]
    for i in range(n):
        box(ax, xs_e[i], ys[i], W, H, fmt(enc[i]), C_SEP)
        if i == 0: arr(ax, (xs_e[0], YT - HIO / 2), (xs_e[0], ys[0] + H / 2))
        if i < n - 1: arr(ax, (xs_e[i], ys[i] - H / 2), (xs_e[i + 1] - 0.25, ys[i + 1] + H / 2), rad=0.0)
    xb, yb = n * dx, -n * 0.78
    box(ax, xb, yb, W, H, fmt(bott), C_SEP, bold=True); ax.text(xb, yb - H / 2 - 0.17, T['bott'], fontsize=8, ha='center', va='center', color='#475569')
    arr(ax, (xs_e[-1], ys[-1] - H / 2), (xb - 0.25, yb + H / 2))
    # up path: dec[0] is the deepest
    xs_d = [xb + (j + 1) * dx for j in range(len(dec))]; ysd = [-(n - 1 - j) * 0.78 for j in range(len(dec))]
    prev = (xb, yb)
    for j in range(len(dec)):
        box(ax, xs_d[j], ysd[j], W, H, fmt(dec[j]), C_SEP)
        arr(ax, (prev[0] + 0.25, prev[1] + H / 2), (xs_d[j] - 0.25, ysd[j] - H / 2), color='#b45309')
        lvl = n - 1 - j
        arr(ax, (xs_e[lvl] + W / 2, ys[lvl]), (xs_d[j] - W / 2, ysd[j]), ls='--', color='#64748b')
        prev = (xs_d[j], ysd[j])
    xe = xs_d[-1] + dx if last is None else xs_d[-1] + dx
    if last is not None:
        yl = 0.78; box(ax, xe, yl, W, H, fmt(last), C_TR); arr(ax, (prev[0] + 0.25, prev[1] + H / 2), (xe - 0.25, yl - H / 2), color='#b45309'); prev = (xe, yl); xe += dx
    box(ax, prev[0], YT, W, HIO, f"{T['out']}\n{fmt(out)}", C_BN, 8.0)
    arr(ax, (prev[0], prev[1] + H / 2), (prev[0], YT - HIO / 2))
    ax.set_xlim(-W / 2 - 0.1, prev[0] + W / 2 + 0.1); ax.set_ylim(yb - 0.65, YT + 0.55); ax.axis('off')

fig = plt.figure(figsize=(9.4, 5.3), dpi=300); gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.5], hspace=0.14, wspace=0.05)
# (a) blok
a = fig.add_subplot(gs[0, :]); a.axis('off'); a.set_xlim(0, 10); a.set_ylim(0, 2)
a.set_title(T['ta'], fontsize=10.5, fontweight='bold', pad=2)
box(a, 1.55, 1.0, 2.7, 0.95, T['std'], C_STD, 9.5)
arr(a, (2.95, 1.0), (4.25, 1.0)); a.text(3.6, 1.18, T['rep'], fontsize=9, ha='center', color='#475569')
box(a, 5.6, 1.0, 2.6, 0.95, T['dw'], C_SEP, 9.5)
arr(a, (6.9, 1.0), (7.45, 1.0)); box(a, 8.65, 1.0, 2.2, 0.95, T['pw'], C_SEP, 9.5)
a.text(5.0, 0.2, T['note'], fontsize=9, ha='center', color='#475569', style='italic')
b = fig.add_subplot(gs[1, 0]); c = fig.add_subplot(gs[1, 1])
e3, b3, d3, i3, o3 = levels_3ch()
unet(b, T['tb'], i3, e3, b3, d3, o3)
e1, b1, d1, l1, i1, o1 = levels_1ch()
unet(c, T['tc'], i1, e1, b1, d1, o1, last=l1)
from matplotlib.lines import Line2D
h = [Line2D([0], [0], color='#0f172a', lw=1, label=T['l1']), Line2D([0], [0], color='#b45309', lw=1, label=T['l2']), Line2D([0], [0], color='#64748b', lw=1, ls='--', label=T['l3'])]
fig.legend(handles=h, loc='lower center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.savefig(OUT + STEM + '.pdf', bbox_inches='tight'); fig.savefig(OUT + STEM + '.png', bbox_inches='tight'); print('ok')
