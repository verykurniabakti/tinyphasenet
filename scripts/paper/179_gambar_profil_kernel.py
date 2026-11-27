"""
Step 179 (follow-up manuscript with the device): two figures, English and Indonesian.
 (1) fig_dev_ops{,_id}.pdf : time per operator group (TRANSPOSE_CONV, CONV_2D, DEPTHWISE_CONV_2D, others) before and after the replacement kernel, six configurations; (a) milliseconds, (b) percent.
     Source: operator-profile logs in firmware/tinyphasenet_profiling/results_hw/ (lines 'OPPROF <index> <operator> <microseconds>'; seed 42): sesi_06nov_kernel_transpos_{ref,cepat}_profil_<model>.log and, for Float32,
     sesi_07nov_float32_profil_{ref,fastpath}_<1ch|3ch>.log (Float32 'reference' = the built-in TFLM kernel). The file names are the same as those used by 169_.
 (2) fig_dev_scatter_gather{,_id}.pdf : scheme of the access pattern of the reference (scatter) and replacement (gather) kernels. A scheme (not data); filter size 7, stride 4, padding 'same' are read from the Keras model (167_/178_ are unchanged),
     and the number of contributions per output sample is computed with the same formula as the kernel (transpose_conv_act_fix.cc).
Output: ../draft_publikasi_1/tabel_dev/. Usage (from this folder): python 179_gambar_profil_kernel.py
"""
import re, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

D = 'firmware/tinyphasenet_profiling/results_hw/'; OUT = '../draft_publikasi_1/tabel_dev/'
V = ['1ch_fp32', '1ch_16x8', '1ch_int8', '3ch_fp32', '3ch_16x8', '3ch_int8']
PREF = {v: (f'sesi_07nov_float32_profil_ref_{v[:3]}.log' if v.endswith('fp32') else f'sesi_06nov_kernel_transpos_ref_profil_{v}.log') for v in V}
PCEP = {v: (f'sesi_07nov_float32_profil_fastpath_{v[:3]}.log' if v.endswith('fp32') else f'sesi_06nov_kernel_transpos_cepat_profil_{v}.log') for v in V}
GRP = lambda op: 'tc' if op == 'TRANSPOSE_CONV' else ('conv' if op == 'CONV_2D' else ('dw' if op == 'DEPTHWISE_CONV_2D' else 'other'))
FM = {'fp32': 'Float32', '16x8': '16x8', 'int8': 'INT8'}


def prof(path):
    t = open(D + path, errors='ignore').read().replace('\r', ''); g = dict(tc=0.0, conv=0.0, dw=0.0, other=0.0)
    for m in re.finditer(r'^OPPROF (\d+) (\S+) ([\d.]+)', t, re.M): g[GRP(m[2])] += float(m[3]) / 1000
    return g


P = {v: dict(ref=prof(PREF[v]), opt=prof(PCEP[v])) for v in V}
K = json.load(open('results/final/kernel_transpos_esp32.json'))
for v in V:   # consistency with 169_: TRANSPOSE_CONV time from the same logs
    assert abs(P[v]['ref']['tc'] - K[v]['ref']['tc_ms']) < 0.2 and abs(P[v]['opt']['tc'] - K[v]['cepat']['tc_ms']) < 0.2, v

TX = {'en': dict(name={'1ch': 'Single-channel', '3ch': 'Three-channel'}, ref='Reference', opt='Replacement', tc='TRANSPOSE_CONV', conv='CONV_2D', dw='DEPTHWISE_CONV_2D', other='Other operators', ms='Time per window (ms)', pct='Share of time (%)', dec='.', th=',',
              s_in='Input samples (each with all input channels)', s_scr='Scratch buffer (output size, int32/int64)', s_out='Output (after bias and requantization)', s_a='(a) Reference kernel: scatter', s_b='(b) Replacement kernel: gather',
              s_cnt='contributions per scratch cell', s_dot='dot product over input channels\n+ bias, requantize\n(no scratch buffer)', s_note='Filter length 7, stride 4, padding same: each output sample receives 1 or 2 contributions.', s_in2='Input samples', s_out2='Output samples'),
      'id': dict(name={'1ch': 'Satu kanal', '3ch': 'Tiga kanal'}, ref='Referensi', opt='Pengganti', tc='TRANSPOSE_CONV', conv='CONV_2D', dw='DEPTHWISE_CONV_2D', other='Operator lain', ms='Waktu per jendela (ms)', pct='Porsi waktu (%)', dec=',', th='.',
              s_in='Sampel masukan (masing-masing dengan semua kanal masukan)', s_scr='Scratch buffer (seukuran keluaran, int32/int64)', s_out='Keluaran (setelah bias dan kuantisasi ulang)', s_a='(a) Reference kernel: menyebarkan (scatter)', s_b='(b) Replacement kernel: mengumpulkan (gather)',
              s_cnt='kontribusi per sel scratch buffer', s_dot='perkalian titik atas kanal masukan\n+ bias, kuantisasi ulang\n(tanpa scratch buffer)', s_note='Panjang filter 7, stride 4, padding same: setiap sampel keluaran menerima 1 atau 2 kontribusi.', s_in2='Sampel masukan', s_out2='Sampel keluaran')}
INK = '#0f172a'; COL = {'tc': '#c2410c', 'conv': '#0f766e', 'dw': '#1d4ed8', 'other': '#cbd5e1'}
plt.rcParams['font.family'] = 'sans-serif'; plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']


def fnum(v, nd, T): return f'{v:,.{nd}f}'.replace(',', '\u0001').replace('.', T['dec']).replace('\u0001', T['th'])


for lg, T in TX.items():
    sfx = '' if lg == 'en' else '_id'
    # ---- figure 1: time per operator group
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.1), dpi=300); y = {}; yt = []; yl = []
    for i, v in enumerate(V):
        base = (len(V) - 1 - i) * 3.0; y[(v, 'ref')] = base + 1; y[(v, 'opt')] = base + 0.0
        yt += [base + 1, base]; yl += [f"{T['name'][v[:3]]} {FM[v[4:]]}  {T['ref'][:3].lower() if False else ''}".strip(), '']
    for ax, mode in zip(axes, ('ms', 'pct')):
        for v in V:
            for k in ('ref', 'opt'):
                g = P[v][k]; tot = sum(g.values()); left = 0.0
                for key in ('tc', 'conv', 'dw', 'other'):
                    w = g[key] if mode == 'ms' else 100 * g[key] / tot
                    ax.barh(y[(v, k)], w, left=left, height=0.8, color=COL[key], edgecolor=INK, linewidth=0.4, zorder=3); left += w
                if mode == 'ms': ax.text(left + 40, y[(v, k)], fnum(tot, 0, T), va='center', fontsize=5.8, color='#475569')
                else: ax.text(101, y[(v, k)], T['ref'] if k == 'ref' else T['opt'], va='center', fontsize=5.6, color=INK)
        ax.set_yticks([(y[(v, 'ref')] + y[(v, 'opt')]) / 2 for v in V]); ax.set_yticklabels([f"{T['name'][v[:3]]} {FM[v[4:]]}" for v in V] if mode == 'ms' else [], fontsize=7)
        ax.set_xlabel(T['ms'] if mode == 'ms' else T['pct'], fontsize=7.4); ax.tick_params(axis='x', labelsize=6.6); ax.grid(axis='x', linestyle='--', alpha=0.35, zorder=0); ax.set_axisbelow(True)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        ax.set_xlim(0, max(sum(P[v]['ref'].values()) for v in V) * 1.16 if mode == 'ms' else 118)
        ax.set_ylim(-0.7, 3 * len(V) - 1.2)
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fc=COL[k], ec=INK) for k in ('tc', 'conv', 'dw', 'other')], labels=[T[k] for k in ('tc', 'conv', 'dw', 'other')], loc='upper center', ncol=4, fontsize=6.8, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.93], w_pad=1.0); fig.savefig(OUT + f'fig_dev_ops{sfx}.pdf', bbox_inches='tight'); plt.close(fig)

    # ---- figure 2: scatter and gather scheme (computed with the kernel formula)
    in_w, fw, st = 4, 7, 4; out_w = in_w * st; pad = ((in_w - 1) * st + fw - out_w) // 2
    contrib = lambda ox: [ix for ix in range(in_w) if 0 <= ox + pad - ix * st < fw]
    cnt = [len(contrib(ox)) for ox in range(out_w)]; assert set(cnt) <= {1, 2}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), dpi=300); CI = ['#0f766e', '#1d4ed8', '#7c3aed', '#b45309']
    for ax, mode in zip(axes, ('scatter', 'gather')):
        ax.set_xlim(-1.3, out_w + 0.3); ax.set_ylim(-0.6, 5.3); ax.axis('off'); ax.set_title(T['s_a'] if mode == 'scatter' else T['s_b'], fontsize=8, fontweight='bold', pad=2)
        xin = lambda ix: st * ix + 2.5   # centre of the output cell that is the centre of the ix-th input filter: 4*ix - pad + 3
        yin, yscr, yout = 4.0, 2.2, 0.6
        for ix in range(in_w):
            ax.add_patch(Rectangle((xin(ix) - 0.45, yin - 0.3), 0.9, 0.6, fc=CI[ix], ec=INK, lw=0.6, alpha=0.85, zorder=3)); ax.text(xin(ix), yin, f'x{ix}', ha='center', va='center', fontsize=6.4, color='white', zorder=4)
        ax.text(out_w / 2, yin + 0.58, T['s_in'], ha='center', fontsize=5.8, color='#475569')
        if mode == 'scatter':
            for ox in range(out_w):
                ax.add_patch(Rectangle((ox + 0.04, yscr - 0.28), 0.92, 0.56, fc='#fde68a' if cnt[ox] == 2 else '#fef3c7', ec=INK, lw=0.5, zorder=2)); ax.text(ox + 0.5, yscr, str(cnt[ox]), ha='center', va='center', fontsize=5.8, zorder=4)
                ax.add_patch(Rectangle((ox + 0.04, yout - 0.25), 0.92, 0.5, fc='#fecaca', ec=INK, lw=0.5, zorder=2))
            for ix in range(in_w):
                for fx in range(fw):
                    ox = ix * st - pad + fx
                    if 0 <= ox < out_w: ax.add_patch(FancyArrowPatch((xin(ix), yin - 0.3), (ox + 0.5, yscr + 0.28), arrowstyle='-|>', mutation_scale=3.5, lw=0.45, color=CI[ix], alpha=0.65, zorder=1))
            for ox in range(out_w): ax.add_patch(FancyArrowPatch((ox + 0.5, yscr - 0.28), (ox + 0.5, yout + 0.25), arrowstyle='-|>', mutation_scale=3.5, lw=0.4, color='#64748b', zorder=1))
            ax.text(out_w / 2, 1.4, T['s_scr'], ha='center', va='center', fontsize=5.8, color='#92400e', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#fde68a', lw=0.6), zorder=5); ax.text(out_w / 2, yout - 0.5, T['s_out'], ha='center', fontsize=5.8, color='#475569')
        else:
            for ox in range(out_w): ax.add_patch(Rectangle((ox + 0.04, yout - 0.25), 0.92, 0.5, fc='#fecaca', ec=INK, lw=0.5, zorder=2))
            sel = 7; ax.add_patch(Rectangle((sel + 0.04, yout - 0.25), 0.92, 0.5, fc='#fca5a5', ec=INK, lw=1.4, zorder=3))
            for ix in contrib(sel): ax.add_patch(FancyArrowPatch((xin(ix), yin - 0.3), (sel + 0.5, yout + 0.25), arrowstyle='-|>', mutation_scale=6, lw=1.1, color=CI[ix], zorder=2))
            ax.text(12.6, 2.3, T['s_dot'], ha='center', va='center', fontsize=6.0, color=INK, bbox=dict(boxstyle='round,pad=0.3', fc='#f8fafc', ec='#cbd5e1', lw=0.6))
            ax.text(out_w / 2, yout - 0.5, T['s_out'], ha='center', fontsize=5.8, color='#475569')
    fig.text(0.5, 0.0, T['s_note'], ha='center', fontsize=6.4, style='italic', color='#475569')
    plt.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig(OUT + f'fig_dev_scatter_gather{sfx}.pdf', bbox_inches='tight'); plt.close(fig)
print('ok; kontribusi per sampel keluaran:', cnt)
