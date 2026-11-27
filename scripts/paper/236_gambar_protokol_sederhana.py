"""
Step 236 (companion compression manuscript): simplified Figure 2. Panel (a) data is replaced by light-background boxes in the style of panel (b) and Figure 1 (no dark bars, no legend, no small notes); panel (b) and the protocol text are the same as in 201_.
Only writes ../draft_publikasi_1/tabel_pend/fig_pend_protokol{,_id}.pdf (201_ is not rerun so that the other tables do not change). Usage (from this folder): python 236_gambar_protokol_sederhana.py
"""
import textwrap
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = '../draft_publikasi_1/tabel_pend/'; INK = '#0f172a'; GR = '#475569'; EC = '#94a3b8'; C_TR, C_VA, C_TE = '#e2e8f0', '#fef3c7', '#dbeafe'
PT = {'en': dict(a='(a) Data', b='(b) Processing of one window', row1=('STEAD', '100,000 earthquake + 100,000 noise'), tv=('Training\n160,000', 'Validation\n20,000 (threshold chosen)', 'Test\n20,000'),
                 row2=('Other datasets', 'test only: 10,000 earthquake\n+ 10,000 noise each'), ds=(('INSTANCE', 'velocimeters', '19,999'), ('ETHZ', 'Swiss networks', '19,999'), ('TXED', 'Texas', '20,000')), win='windows',
                 steps=('Vertical trace\n100 Hz', 'Band-pass\n1–45 Hz', 'Window\n3072 samples\n(30.72 s)', 'Zero mean,\nunit variance', 'TinyPhaseNet\n(any format)', 'P, S and noise\nprobability\ncurves', 'Max P probability\n≥ threshold?', 'Pick = sample of\nmaximum\nprobability'),
                 trn='Training: Gaussian labels (\u03c3 = 0.1 s) at the analyst P and S picks; 25 epochs, Adam, batch size 64, cosine learning rate 1e-3 \u2192 1e-5.'),
      'id': dict(a='(a) Data', b='(b) Pemrosesan satu jendela', row1=('STEAD', '100.000 gempa + 100.000 derau'), tv=('Latih\n160.000', 'Validasi\n20.000 (ambang dipilih)', 'Uji\n20.000'),
                 row2=('Dataset lain', 'hanya uji: 10.000 gempa\n+ 10.000 derau tiap dataset'), ds=(('INSTANCE', 'velosimeter', '19.999'), ('ETHZ', 'jaringan Swiss', '19.999'), ('TXED', 'Texas', '20.000')), win='jendela',
                 steps=('Jejak vertikal\n100 Hz', 'Band-pass\n1–45 Hz', 'Jendela\n3072 sampel\n(30,72 s)', 'Rerata nol,\nvarians satu', 'TinyPhaseNet\n(format apa pun)', 'Kurva probabilitas\nP, S, dan derau', 'Probabilitas P maks.\n≥ ambang?', 'Waktu tiba = sampel\nprobabilitas\nmaksimum'),
                 trn='Pelatihan: label Gaussian (\u03c3 = 0,1 s) pada \u0070ick P dan S analis; 25 epoch, Adam, ukuran batch 64, laju belajar cosine 1e-3 \u2192 1e-5.')}
box = lambda ax, x, y, w, h, fc, txt, fs=5.4, bold=False: (ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1,rounding_size=0.8', fc=fc, ec=EC, lw=0.8)), ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold' if bold else 'normal'))
for lg, T in PT.items():
    sfx = '' if lg == 'en' else '_id'
    fig = plt.figure(figsize=(7.4, 3.45), dpi=300); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(-9, 47); ax.axis('off')
    ax.text(1, 45.4, T['a'], fontsize=8.4, fontweight='bold', color=INK, va='center'); X0, W_, g = 24, 74, 1.7; w3 = (W_ - 2 * g) / 3
    ax.text(1, 40.0, T['row1'][0], fontsize=7.4, fontweight='bold', color=INK, va='center'); ax.text(1, 36.9, T['row1'][1], fontsize=5.5, color=GR, va='center')
    for i, (txt, col) in enumerate(zip(T['tv'], (C_TR, C_VA, C_TE))): box(ax, X0 + i * (w3 + g), 34.2, w3, 7.4, col, txt, 5.8)
    ax.text(1, 30.4, T['row2'][0], fontsize=7.4, fontweight='bold', color=INK, va='center'); ax.text(1, 26.4, T['row2'][1], fontsize=5.5, color=GR, va='center', linespacing=1.3)
    for i, (nm, note, n) in enumerate(T['ds']): box(ax, X0 + i * (w3 + g), 24.0, w3, 7.4, C_TE, f'{nm} ({note})\n{n} {T["win"]}', 5.8)
    ax.text(1, 20.2, T['b'], fontsize=8.4, fontweight='bold', color=INK, va='center'); bw, gap, bh = 21.4, 3.7, 7.6; xs_ = [1 + i * (bw + gap) for i in range(4)]; fcs = ['#e2e8f0'] * 4 + ['#dbeafe', '#e2e8f0', '#fef3c7', '#e2e8f0']; rowy = [10.2, -1.4]
    for n, (txt, col) in enumerate(zip(T['steps'], fcs)):
        r, c = divmod(n, 4); box(ax, xs_[c], rowy[r], bw, bh, col, txt, 6.0)
        if c < 3: ax.annotate('', xy=(xs_[c + 1] - 0.1, rowy[r] + bh / 2), xytext=(xs_[c] + bw + 0.1, rowy[r] + bh / 2), arrowprops=dict(arrowstyle='->', color=GR, lw=0.9))
    xe = xs_[3] + bw / 2; xb = xs_[0] + bw / 2; ym = (rowy[0] + rowy[1] + bh) / 2   # connector from box 4 (top row) to box 5 (bottom row)
    ax.plot([xe, xe, xb], [rowy[0] - 0.1, ym, ym], color=GR, lw=0.9, solid_capstyle='butt'); ax.annotate('', xy=(xb, rowy[1] + bh + 0.1), xytext=(xb, ym), arrowprops=dict(arrowstyle='->', color=GR, lw=0.9))
    yy = -5.4
    for sx in (T['trn'],):
        for ln in textwrap.wrap(sx, 150): ax.text(1, yy, ln, fontsize=5.7, color=GR, va='center'); yy -= 2.0
        yy -= 0.6
    fig.savefig(OUT + f'fig_pend_protokol{sfx}.pdf'); plt.close(fig)
print('ok')
