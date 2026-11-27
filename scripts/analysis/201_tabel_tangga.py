"""
Langkah 201 (draf pendamping kompresi, tahap 2): tangga bit bobot dan aktivasi untuk model depthwise separable DAN konvolusi standar, satu kanal dan tiga kanal (seed 42-44), dari hasil tersimpan (tanpa percobaan baru).
Format (nama menurut bit bobot/aktivasi): Float32 (32/32), 8/16 (disebut 16x8 oleh TFLite), 4/16 (bobot 4-bit pcmse, aktivasi 16-bit; emulasi akurasi), 8/8 (disebut INT8 oleh TFLite), 4/8 (bobot 4-bit pcmse, aktivasi 8-bit; emulasi akurasi).
Sumber: depthwise: results/final/{quant16x8_per_seed,quant16x8_pndw_per_seed,spektrum_kuantisasi_3seed,w4a8_perkanal_3seed}.json; standar: results/ablation_final/eval_{standard,pn}.json (Float32) dan results/final/ptq_padrao_3seed.json (199_).
Semua selisih BERPASANGAN per seed terhadap Float32 model yang SAMA, rerata empat dataset (STEAD, INSTANCE, ETHZ, TXED uji), lalu rerata +- simpangan baku sampel atas tiga seed.
Ukuran berkas: byte berkas TFLite; format 4-bit = byte berkas dikurangi separuh jumlah bobot kernel (konvolusi, depthwise, konvolusi transpos; dua bobot per byte), sama untuk semua model (cocok dengan berkas 4-bit nyata dari 173_ dalam 1 byte).
Memori aktivasi: batas bawah arena (arena_lower_bound, 159_); 4/8 memakai nilai INT8, 4/16 memakai nilai 16x8. Dominasi: konfigurasi A mendominasi B bila F1 rerata empat dataset A >= B, berkas A <= B, memori aktivasi A <= B, dan sedikitnya satu lebih baik.
Keluaran: ../draft_publikasi_1/tabel_pend/{tab_ladder,tab_steps_full}{,_id}.tex, numbers_ladder{,_id}.tex, fig_pend_tradeoff{,_id}.pdf, results/final/pendamping_tangga.json
Pemakaian (dari folder ini): python 201_tabel_tangga.py
"""
import os, json, statistics as st
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = '../draft_publikasi_1/tabel_pend/'; os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 43, 44]; DS = ['stead_test', 'instance_test', 'ethz_test', 'txed_test']
S = json.load(open('results/final/spektrum_kuantisasi_3seed.json')); W = json.load(open('results/final/w4a8_perkanal_3seed.json')); P = json.load(open('results/final/ptq_padrao_3seed.json'))
Q = {'1ch': {int(k): v for k, v in json.load(open('results/final/quant16x8_per_seed.json')).items()}, '3ch': {int(k): v for k, v in json.load(open('results/final/quant16x8_pndw_per_seed.json')).items()}}
STD = {'1ch': {int(e['model'].split('seed')[1].split('.')[0]): e for e in json.load(open('results/ablation_final/eval_standard.json'))}, '3ch': {int(e['model'].split('seed')[1].split('.')[0]): e for e in json.load(open('results/ablation_final/eval_pn.json'))}}
PARAMS = json.load(open('results/final/parametros_modelos.json'))   # 187_
MAC = json.load(open('results/final/mac_flops.json'))   # 202_: MAC dan FLOPs (graf TFLite Float32, tiga cara identik); tidak bergantung pada format kuantisasi
FMTS = ['fp32', 'q16x8', 'w4a16', 'int8', 'w4a8']; QF = FMTS[1:]
BITS = {'fp32': '32 / 32', 'q16x8': '8 / 16', 'w4a16': '4 / 16', 'int8': '8 / 8', 'w4a8': '4 / 8'}
SEPKEY = {'fp32': 'fp32', 'q16x8': 'q16x8', 'int8': 'int8', 'w4a8': 'pcmse', 'w4a16': 'pcmse_a16'}


def f1(fam, conv, fmt, s, d):   # F1 (persen)
    if conv == 'standar':
        return 100 * (STD[fam][s][d]['f1'] if fmt == 'fp32' else P[fam][fmt][str(s)]['kualitas'][d]['f1'])
    if fmt in ('fp32', 'q16x8'): return 100 * Q[fam][s][d][fmt]['f1']
    if fmt == 'int8': return 100 * S[fam][str(s)]['int8']['kualitas'][d]['f1']
    return 100 * W[fam][str(s)][SEPKEY[fmt]]['kualitas'][d]['f1']


def tflite_path(fam, conv, fmt, s):
    if conv == 'standar': return f'models/quant_padrao/{fam}_standard_{fmt}_seed{s}.tflite'
    return {'int8': f'models/quant_spectrum/{fam}_int8_seed{s}.tflite', 'w4a8': f'models/quant_spectrum/{fam}_w4a8_pcmse_seed{s}.tflite', 'w4a16': f'models/quant_spectrum/{fam}_w4a8_pcmse_a16_seed{s}.tflite',
            'q16x8': f"models/final_16x8{'' if fam == '1ch' else '_pndw'}/seed{s}.tflite"}[fmt]


def n_kernel(path):   # jumlah bobot filter pada CONV_2D, DEPTHWISE_CONV_2D, TRANSPOSE_CONV (filter = masukan ke-2)
    it = tf.lite.Interpreter(model_path=path); it.allocate_tensors(); td = {d['index']: d for d in it.get_tensor_details()}
    return int(sum(np.prod(td[o['inputs'][1]]['shape']) for o in it._get_ops_details() if o['op_name'] in ('CONV_2D', 'DEPTHWISE_CONV_2D', 'TRANSPOSE_CONV')))


def size_arena(fam, conv, fmt):   # (berkas KB, memori aktivasi KB), rerata tiga seed
    fk = 'standard' if fam == '1ch' else 'pn'; fl, ar = [], []
    for s in SEEDS:
        if fmt == 'fp32':
            e = S[fam][str(s)][fk if conv == 'standar' else 'fp32']; b, a = e['bytes'], e['arena_lb']
        else:
            src = (P[fam][fmt][str(s)] if conv == 'standar' else (S[fam][str(s)][fmt] if fmt in ('q16x8', 'int8') else None))
            if src is not None: b, a = src['bytes'], src['arena_lb']
            else:   # depthwise 4/8 / 4/16: berkas dan arena dari berkas TFLite hasil 161_
                blob = open(tflite_path(fam, conv, fmt, s), 'rb').read(); b = len(blob); a = S[fam][str(s)]['int8' if fmt == 'w4a8' else 'q16x8']['arena_lb']
        if fmt in ('w4a8', 'w4a16'): b -= n_kernel(tflite_path(fam, conv, fmt, s)) // 2
        if conv == 'standar' and fmt in ('w4a8', 'w4a16'): a = P[fam][fmt][str(s)]['arena_lb']
        fl.append(b / 1024); ar.append(a / 1024)
    return st.mean(fl), st.mean(ar)


R = {}   # R[fam][conv][fmt] = dict(f1_ds, f1_mean4, delta_mean4, delta_ds, file, act)
for fam in ('1ch', '3ch'):
    for conv in ('separable', 'standar'):
        R.setdefault(fam, {}).setdefault(conv, {})
        for fmt in FMTS:
            fd = {d: [f1(fam, conv, fmt, s, d) for s in SEEDS] for d in DS}; m4 = [st.mean(fd[d][i] for d in DS) for i in range(3)]
            fl, ar = size_arena(fam, conv, fmt); R[fam][conv][fmt] = dict(f1_ds={d: st.mean(v) for d, v in fd.items()}, f1_seeds=m4, f1_mean4=st.mean(m4), file_kb=fl, act_kb=ar)
        for fmt in QF:   # selisih berpasangan terhadap Float32 model yang sama
            R[fam][conv][fmt]['delta_seeds'] = [R[fam][conv][fmt]['f1_seeds'][i] - R[fam][conv]['fp32']['f1_seeds'][i] for i in range(3)]
            R[fam][conv][fmt]['delta_ds'] = {d: [f1(fam, conv, fmt, s, d) - f1(fam, conv, 'fp32', s, d) for s in SEEDS] for d in DS}

# ---------- Pareto 20 konfigurasi (F1 rerata empat dataset, berkas, memori aktivasi)
CFG = {(fam, conv, fmt): (R[fam][conv][fmt]['f1_mean4'], R[fam][conv][fmt]['file_kb'], R[fam][conv][fmt]['act_kb']) for fam in R for conv in R[fam] for fmt in FMTS}
dom = lambda a, b: a[0] >= b[0] and a[1] <= b[1] and a[2] <= b[2] and (a[0] > b[0] or a[1] < b[1] or a[2] < b[2])
ND = {k: not any(dom(w, v) for j, w in CFG.items() if j != k) for k, v in CFG.items()}
json.dump(dict(R={f: {c: {m: {k: v for k, v in x.items()} for m, x in d.items()} for c, d in cc.items()} for f, cc in R.items()}, nondominated={'|'.join(k): v for k, v in ND.items()}), open('results/final/pendamping_tangga.json', 'w'), indent=1)
for k, v in CFG.items(): print(k, 'F1 %.2f berkas %.1f KB memori %.1f KB' % v, 'TAK TERDOMINASI' if ND[k] else '')

# ---------- teks tabel
LG = {'en': dict(dec='.', fam={'1ch': 'Single-channel', '3ch': 'Three-channel'}, conv={'separable': 'separable', 'standar': 'standard'}, fmt={'fp32': 'Float32', 'q16x8': '8/16', 'w4a16': '4/16', 'int8': '8/8', 'w4a8': '4/8'}),
      'id': dict(dec=',', fam={'1ch': 'Kanal tunggal', '3ch': 'Tiga kanal'}, conv={'separable': '\\textit{separable}', 'standar': 'standar'}, fmt={'fp32': 'Float32', 'q16x8': '8/16', 'w4a16': '4/16', 'int8': '8/8', 'w4a8': '4/8'})}


def n(v, d, dec, sign=False):
    t = f'{v:+.{d}f}' if sign else f'{v:,.{d}f}'
    if abs(v) < 0.5 * 10 ** -d and sign: t = f'{0:.{d}f}'
    return t.replace('-', '$-$').replace(',', '§').replace('.', dec).replace('§', '.' if dec == ',' else ',')


def pm(xs, dec): return f"{n(st.mean(xs), 1, dec, True)} $\\pm$ {n(st.stdev(xs), 1, dec)}"


for lg, L in LG.items():
    sfx = '' if lg == 'en' else '_id'; dec = L['dec']
    # tangga: selisih F1 terhadap Float32 model sama; baris = format, kolom = keluarga x jenis konvolusi
    rows = ''
    for fmt in QF: rows += f"{L['fmt'][fmt].replace('/', ' & ')} & " + ' & '.join(pm(R[fam][conv][fmt]['delta_seeds'], dec) for fam in ('1ch', '3ch') for conv in ('separable', 'standar')) + ' \\\\\n'
    open(OUT + f'tab_ladder{sfx}.tex', 'w').write(rows + '\\bottomrule\n')
    # tabel lengkap
    rows = ''
    for fi, fam in enumerate(('1ch', '3ch')):
        for conv in ('standar', 'separable'):
            key = ('standard' if fam == '1ch' else 'pn') if conv == 'standar' else 'fp32'
            par = PARAMS[f"{fam}_{key}"]
            for fmt in FMTS:
                r = R[fam][conv][fmt]; nm = f"{L['fam'][fam]} {L['conv'][conv]}{' (PhaseNet)' if (fam == '3ch' and conv == 'standar') else ''}"
                mc = MAC[f'{fam}_{conv}']['macs_tflite']; c3 = (n(par, 0, dec), n(mc / 1e6, 2, dec), n(2 * mc / 1e6, 2, dec)) if fmt == 'fp32' else ('', '', '')   # parameter, MAC, FLOPs hanya pada baris Float32 tiap model (sama pada format lain)
                rows += f"{nm} & {c3[0]} & {c3[1]} & {c3[2]} & {BITS[fmt]} & {n(r['file_kb'], 1, dec)} & {n(r['act_kb'], 0, dec)} & " + ' & '.join(n(r['f1_ds'][d], 1, dec) for d in DS) + f" & {n(r['f1_mean4'], 1, dec)} \\\\\n"
        if fi == 0: rows += '\\midrule\n'
    open(OUT + f'tab_steps_full{sfx}.tex', 'w').write(rows + '\\bottomrule\n')

# ---------- makro angka (kehilangan = Float32 dikurangi format, poin; rerata tiga seed)
M = {}
for fam, t in (('1ch', 'One'), ('3ch', 'Thr')):
    for conv, c in (('separable', 'Dw'), ('standar', 'Sd')):
        for fmt, f in (('q16x8', 'Qq'), ('w4a16', 'WaS'), ('int8', 'Int'), ('w4a8', 'Four')):
            M[f'Ld{c}{f}{t}'] = -st.mean(R[fam][conv][fmt]['delta_seeds'])
        M[f'LdMax{c}{t}'] = max(-st.mean(R[fam][conv][f]['delta_seeds']) for f in QF)
        M[f'LdAbs{c}{t}'] = max(abs(st.mean(R[fam][conv][f]['delta_seeds'])) for f in QF)   # selisih mutlak terbesar atas empat format terkuantisasi
        # efek aktivasi (16 -> 8 bit, bobot 8 bit) dan efek bobot (8 -> 4 bit, aktivasi 16 bit)
        M[f'LdAct{c}{t}'] = R[fam][conv]['q16x8']['f1_mean4'] - R[fam][conv]['int8']['f1_mean4']; M[f'LdWgt{c}{t}'] = R[fam][conv]['q16x8']['f1_mean4'] - R[fam][conv]['w4a16']['f1_mean4']
        M[f'FmFp{c}{t}'] = R[fam][conv]['fp32']['f1_mean4']; M[f'FileFp{c}{t}'] = R[fam][conv]['fp32']['file_kb']
        for fmt, f in (('q16x8', 'Qq'), ('int8', 'Int'), ('w4a8', 'Four'), ('w4a16', 'WaS')): M[f'File{f}{c}{t}'] = R[fam][conv][fmt]['file_kb']; M[f'Act{f}{c}{t}'] = R[fam][conv][fmt]['act_kb']; M[f'Fm{f}{c}{t}'] = R[fam][conv][fmt]['f1_mean4']
    M[f'FileRatQq{t}'] = R[fam]['standar']['q16x8']['file_kb'] / R[fam]['separable']['q16x8']['file_kb']; M[f'FileRatInt{t}'] = R[fam]['standar']['int8']['file_kb'] / R[fam]['separable']['int8']['file_kb']
    M[f'FileRatFp{t}'] = R[fam]['standar']['fp32']['file_kb'] / R[fam]['separable']['fp32']['file_kb']
    M[f'GapQq{t}'] = R[fam]['standar']['q16x8']['f1_mean4'] - R[fam]['separable']['q16x8']['f1_mean4']; M[f'GapInt{t}'] = R[fam]['standar']['int8']['f1_mean4'] - R[fam]['separable']['int8']['f1_mean4']
for fam, t in (('1ch', 'One'), ('3ch', 'Thr')):   # MAC (juta), rasio standar/separable, porsi konvolusi transpos pada model separable
    a, b = MAC[f'{fam}_standar'], MAC[f'{fam}_separable']; M[f'MacStd{t}'] = a['macs_tflite'] / 1e6; M[f'MacSep{t}'] = b['macs_tflite'] / 1e6; M[f'MacRat{t}'] = a['macs_tflite'] / b['macs_tflite']; M[f'MacTrSh{t}'] = 100 * b['por_operator']['TRANSPOSE_CONV'] / b['macs_tflite']
M['LdQqDsMax'] = max(abs(st.mean(R[f][c]['q16x8']['delta_ds'][d])) for f in R for c in R[f] for d in DS)   # 16x8: maks selisih mutlak per dataset, dua keluarga dan dua jenis konvolusi
M['ParetoN'] = float(len(CFG)); M['ParetoND'] = float(sum(ND.values()))
CFGR = {k: v for k, v in CFG.items() if k[2] in ('fp32', 'q16x8', 'int8')}   # hanya format yang dapat dijalankan (tanpa W4, yang hanya emulasi akurasi)
NDR = {k: not any(dom(w, v) for j, w in CFGR.items() if j != k) for k, v in CFGR.items()}; M['ParetoRunN'] = float(len(CFGR)); M['ParetoRunND'] = float(sum(NDR.values()))
for lg, dec in (('en', '.'), ('id', ',')):
    sfx = '' if lg == 'en' else '_id'
    open(OUT + f'numbers_ladder{sfx}.tex', 'w').write(''.join(f"\\newcommand{{\\dv{k}}}{{{(f'{v:.0f}' if k.startswith('Pareto') else f'{v:.1f}').replace('.', dec)}}}\n" for k, v in M.items()))

# ---------- gambar: satu panel, 20 konfigurasi. x = ukuran berkas (log), y = F1 rerata empat dataset, luas penanda = memori aktivasi; terisi = tidak terdominasi (dominasi gabungan semua 20 konfigurasi atas ketiga ukuran)
COL = {'separable': '#1d4ed8', 'standar': '#b45309'}; INK = '#0f172a'; MK = {'fp32': 'o', 'q16x8': 's', 'w4a16': 'P', 'int8': '^', 'w4a8': 'v'}
TXT = {'en': dict(x='File size (KB, log scale)', y='Mean F1-score, four datasets (%)', conv={'separable': 'Depthwise separable', 'standar': 'Standard convolution'}, ct='convolution (colour)', ft='bits, weights/activations (marker)', at='activation memory (area)', better='better', ch='-ch',
                  nd='filled: not dominated in F1-score, file size and activation memory (%d configurations); open: dominated'),
       'id': dict(x='Ukuran berkas (KB, skala log)', y='F1-score rerata, empat dataset (%)', conv={'separable': 'Depthwise separable', 'standar': 'Konvolusi standar'}, ct='konvolusi (warna)', ft='bit bobot/aktivasi (penanda)', at='memori aktivasi (luas)', better='lebih baik', ch=' kanal',
                  nd='terisi: tidak terdominasi dalam F1-score, ukuran berkas, dan memori aktivasi (%d konfigurasi); kosong: terdominasi')}
OFFS = {('1ch', 'separable', 'w4a16'): (-7, 0, 'right'), ('1ch', 'separable', 'w4a8'): (8, 0, 'left'), ('1ch', 'standar', 'q16x8'): (6, 9, 'left'), ('1ch', 'standar', 'w4a16'): (-3, 11, 'right'),
        ('1ch', 'standar', 'int8'): (2, -12, 'center'), ('1ch', 'standar', 'w4a8'): (-8, -9, 'right'), ('3ch', 'standar', 'w4a8'): (0, 11, 'center')}   # label hanya pada titik tak terdominasi; geseran dalam titik
LAB = {'fp32': 'Float32', 'q16x8': '8/16', 'w4a16': '4/16', 'int8': '8/8', 'w4a8': '4/8'}; AREA = lambda a: 14 + a * 0.55
for lg in ('en', 'id'):
    T = TXT[lg]; sfx = '' if lg == 'en' else '_id'; fig, ax = plt.subplots(figsize=(7.4, 4.5), dpi=300)
    for (fam, conv, f), (y, x, a) in CFG.items():
        nd = ND[(fam, conv, f)]; ax.scatter(x, y, marker=MK[f], s=AREA(a), facecolor=COL[conv] if nd else 'white', edgecolor=COL[conv], linewidth=1.4, alpha=0.95, zorder=3)
        if nd and (fam, conv, f) in OFFS:
            dx, dy, ha = OFFS[(fam, conv, f)]; ax.annotate(f"{fam[0]}{T['ch']} {LAB[f]}", (x, y), textcoords='offset points', xytext=(dx, dy), fontsize=6.4, ha=ha, va='center', color=INK, zorder=4)
    ax.set_xscale('log'); ax.set_xlim(45, 1400); ax.set_xticks([50, 100, 200, 500, 1000]); ax.set_xticklabels(['50', '100', '200', '500', '1000']); ax.minorticks_off()
    ax.set_xlabel(T['x'], fontsize=8); ax.set_ylabel(T['y'], fontsize=8); ax.tick_params(labelsize=7); ax.grid(linestyle='--', alpha=0.35, zorder=0)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.annotate(T['better'], xy=(0.55, 0.66), xycoords='axes fraction', xytext=(0.64, 0.56), textcoords='axes fraction', fontsize=7.5, color='#475569', ha='center', arrowprops=dict(arrowstyle='->', color='#475569', lw=1.0))
    h1 = [Line2D([0], [0], color=COL[c], marker='o', linestyle='', markersize=6, label=T['conv'][c]) for c in COL]
    h2 = [Line2D([0], [0], color='#475569', marker=MK[f], linestyle='', markersize=5.5, markerfacecolor='none', label=LAB[f]) for f in LAB]
    h3 = [Line2D([0], [0], color='#475569', marker='o', linestyle='', markersize=AREA(a) ** 0.5, markerfacecolor='none', label=f'{a} KB') for a in (48, 96, 192, 384)]
    l1 = ax.legend(handles=h1, loc='lower right', fontsize=6.6, frameon=False, title=T['ct'], title_fontsize=6.6, bbox_to_anchor=(1.0, 0.02)); ax.add_artist(l1)
    l2 = ax.legend(handles=h2, loc='lower right', fontsize=6.6, frameon=False, title=T['ft'], title_fontsize=6.6, bbox_to_anchor=(0.78, 0.02)); ax.add_artist(l2)
    ax.legend(handles=h3, loc='lower right', fontsize=6.6, frameon=False, title=T['at'], title_fontsize=6.6, bbox_to_anchor=(0.56, 0.02))
    fig.text(0.5, 0.005, T['nd'] % len(CFG), ha='center', fontsize=6.8, color=INK); plt.tight_layout(rect=[0, 0.03, 1, 1]); fig.savefig(OUT + f'fig_pend_tradeoff{sfx}.pdf', bbox_inches='tight'); plt.close(fig)

# ---------- gambar utama: diagram titik Cleveland, tiga panel (F1, berkas, memori aktivasi), baris diurutkan menurut berkas; baris tak terdominasi diarsir; baris lain menyebut satu konfigurasi yang mendominasinya (F1 tertinggi)
DT = {'en': dict(f1='Mean F1-score (%)', fl='File size (KB, log)', ac='Activation memory (KB, log)', by='dominated by', conv=('standard', 'separable'), ch='-ch ', c1='Depthwise separable', c2='Standard convolution', nd='filled, shaded row: not dominated', dd='open: dominated'),
      'id': dict(f1='F1-score rerata (%)', fl='Ukuran berkas (KB, log)', ac='Memori aktivasi (KB, log)', by='terdominasi oleh', conv=('standar', 'separable'), ch=' kanal ', c1='Depthwise separable', c2='Konvolusi standar', nd='terisi, baris diarsir: tak terdominasi', dd='kosong: terdominasi')}
for lg in ('en', 'id'):
    T = DT[lg]; sfx = '' if lg == 'en' else '_id'; lab = lambda k: f"{k[0][0]}{T['ch']}{T['conv'][0] if k[1] == 'standar' else T['conv'][1]} {LAB[k[2]]}"
    rows = sorted(({'k': k, 'lab': lab(k), 'conv': k[1], 'y': v[0], 'f': v[1], 'a': v[2], 'nd': ND[k]} for k, v in CFG.items()), key=lambda r: r['f'])
    for r in rows:
        ds = sorted((o for o in rows if o is not r and dom((o['y'], o['f'], o['a']), (r['y'], r['f'], r['a']))), key=lambda o: (-o['y'], o['f'])); r['by'] = ds[0]['lab'] if ds else None
    n_ = len(rows); yp = list(range(n_)); fig, axes = plt.subplots(1, 3, figsize=(7.8, 6.3), dpi=300, sharey=True, gridspec_kw=dict(width_ratios=[1.1, 1, 0.8], wspace=0.07))
    for ax, (key, ttl, lim, lgx, fm_) in zip(axes, (('y', T['f1'], (74, 100.8), False, '{:.1f}'), ('f', T['fl'], (45, 1700), True, '{:,.0f}'), ('a', T['ac'], (36, 560), True, '{:.0f}'))):
        for r, y_ in zip(rows, yp):
            if r['nd']: ax.axhspan(y_ - 0.5, y_ + 0.5, color='#fef3c7', zorder=0)
        for r, y_ in zip(rows, yp):
            ax.hlines(y_, lim[0], r[key], color='#cbd5e1', linewidth=0.7, zorder=1)
            ax.scatter(r[key], y_, s=34, marker='o', facecolor=COL[r['conv']] if r['nd'] else 'white', edgecolor=COL[r['conv']], linewidth=1.5, zorder=3)
            ax.text(r[key] * (1.13 if lgx else 1) + (0 if lgx else 0.8), y_, fm_.format(r[key]), va='center', fontsize=5.6, color=INK, fontweight='bold' if r['nd'] else 'normal', zorder=4)
        if lgx: ax.set_xscale('log'); ax.minorticks_off()
        ax.set_xlim(*lim); ax.set_title(ttl, fontsize=7.4, fontweight='bold', pad=4); ax.tick_params(labelsize=6.2); ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    axes[1].set_xticks([50, 100, 200, 500, 1000]); axes[1].set_xticklabels(['50', '100', '200', '500', '1000']); axes[2].set_xticks([48, 96, 192, 384]); axes[2].set_xticklabels(['48', '96', '192', '384'])
    axes[0].set_yticks(yp); axes[0].set_yticklabels([r['lab'] for r in rows], fontsize=6.2); axes[0].invert_yaxis()
    for t, r in zip(axes[0].get_yticklabels(), rows):
        if r['nd']: t.set_fontweight('bold')
    for r, y_ in zip(rows, yp): axes[2].text(1.06, y_, '' if r['nd'] else '\u2190 ' + r['by'], transform=axes[2].get_yaxis_transform(), va='center', fontsize=5.4, color='#475569', clip_on=False)
    axes[2].text(1.06, -1.4, T['by'], transform=axes[2].get_yaxis_transform(), fontsize=6.2, fontweight='bold', color='#475569', clip_on=False)
    fig.legend(handles=[Line2D([0], [0], marker='o', color=COL['separable'], linestyle='', markersize=6, label=T['c1']), Line2D([0], [0], marker='o', color=COL['standar'], linestyle='', markersize=6, label=T['c2']),
                        Line2D([0], [0], marker='o', color='#475569', markerfacecolor='#475569', linestyle='', markersize=6, label=T['nd']), Line2D([0], [0], marker='o', color='#475569', markerfacecolor='white', linestyle='', markersize=6, label=T['dd'])],
               loc='upper center', ncol=4, fontsize=6.3, frameon=False, bbox_to_anchor=(0.45, 0.995))
    fig.subplots_adjust(left=0.2, right=0.80, top=0.925, bottom=0.05); fig.savefig(OUT + f'fig_pend_dotplot{sfx}.pdf'); plt.close(fig)

# ---------- Tabel 1 ringkas (empat model: parameter, MAC, FLOPs, berkas dan memori aktivasi Float32) dan heatmap F1 per dataset (20 konfigurasi)
from matplotlib.colors import LinearSegmentedColormap
for lg, L in LG.items():
    sfx = '' if lg == 'en' else '_id'; dec = L['dec']; rows = ''
    for fam in ('1ch', '3ch'):
        for conv in ('standar', 'separable'):
            key = ('standard' if fam == '1ch' else 'pn') if conv == 'standar' else 'fp32'; r = R[fam][conv]['fp32']; mc = MAC[f'{fam}_{conv}']['macs_tflite']
            nm = f"{L['fam'][fam]} {L['conv'][conv]}{' (PhaseNet)' if (fam == '3ch' and conv == 'standar') else ''}"
            rows += f"{nm} & {n(PARAMS[f'{fam}_{key}'], 0, dec)} & {n(mc / 1e6, 2, dec)} & {n(2 * mc / 1e6, 2, dec)} & {n(r['file_kb'], 1, dec)} & {n(r['act_kb'], 0, dec)} \\\\\n"
    open(OUT + f'tab_models{sfx}.tex', 'w').write(rows + '\\bottomrule\n')
HT = {'en': dict(mean='Mean', cb='F1-score (%)', ch='-ch ', conv=('standard', 'separable')), 'id': dict(mean='Rerata', cb='F1-score (%)', ch=' kanal ', conv=('standar', 'separable'))}
CM = LinearSegmentedColormap.from_list('seq', ['#eff6ff', '#bfdbfe', '#60a5fa', '#2563eb', '#1e3a8a'])
for lg in ('en', 'id'):
    T = HT[lg]; sfx = '' if lg == 'en' else '_id'; A_, labs, gaps = [], [], []
    for fam in ('1ch', '3ch'):
        for conv in ('standar', 'separable'):
            for fmt in FMTS:
                v = R[fam][conv][fmt]; A_.append([v['f1_ds'][d] for d in DS] + [v['f1_mean4']]); labs.append(f"{fam[0]}{T['ch']}{T['conv'][0] if conv == 'standar' else T['conv'][1]} {LAB[fmt]}")
            gaps.append(len(A_))
    A_ = np.array(A_); xs = [0, 1, 2, 3, 4.35]; vmin, vmax = 74, 100; fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=300)
    ypos = [i + 0.35 * sum(1 for g in gaps[:-1] if i >= g) for i in range(len(A_))]
    for i, yy in enumerate(ypos):
        for jx in range(A_.shape[1]):
            v = A_[i, jx]; c = CM((v - vmin) / (vmax - vmin)); ax.add_patch(plt.Rectangle((xs[jx] - 0.48, yy - 0.46), 0.96, 0.92, color=c, zorder=2))
            ax.text(xs[jx], yy, (f'{v:.1f}'.replace('.', ',') if lg == 'id' else f'{v:.1f}'), ha='center', va='center', fontsize=6.4, color='white' if (v - vmin) / (vmax - vmin) > 0.55 else '#0f172a', fontweight='bold' if jx == 4 else 'normal', zorder=3)
    ax.set_yticks(ypos); ax.set_yticklabels(labs, fontsize=6.4); ax.set_xticks(xs); ax.set_xticklabels(['STEAD', 'INSTANCE', 'ETHZ', 'TXED', T['mean']], fontsize=6.8); ax.xaxis.tick_top(); ax.invert_yaxis()
    ax.set_xlim(-0.6, 4.85); ax.set_ylim(max(ypos) + 0.7, -0.9); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=CM), ax=ax, orientation='horizontal', fraction=0.025, pad=0.02, aspect=40); cb.ax.tick_params(labelsize=6); cb.set_label(T['cb'], fontsize=6.6)
    fig.subplots_adjust(left=0.36, right=0.98, top=0.93, bottom=0.07); fig.savefig(OUT + f'fig_pend_heatmap{sfx}.pdf'); plt.close(fig)

# ---------- gambar protokol (Gambar data dan alur pemrosesan satu jendela), EN dan ID
import textwrap
from matplotlib.patches import FancyBboxPatch, Rectangle
PT = {'en': dict(a='(a) Data', b='(b) Processing of one window', tr='Training: %s', va='%s', te='%s', stead='100,000 earthquake + 100,000 noise', n19='19,995 windows after excluding constant vertical components', only='Test only: 10,000 earthquake + 10,000 noise  (%s windows)',
                  lg=('training', 'validation (threshold chosen here)', 'test'), notes=('velocimeters', 'Swiss networks', 'Texas'), th=('160,000', '20,000', '20,000'), nn=('19,999', '19,999', '20,000'),
                  steps=('Vertical trace\n100 Hz', 'Band-pass\n1\u201345 Hz', 'Window\n3072 samples\n(30.72 s)', 'Zero mean,\nunit variance', 'TinyPhaseNet\n(any format)', 'P, S and noise\nprobability\ncurves', 'Max P probability\n\u2265 threshold?', 'Pick = sample of\nmaximum\nprobability'),
                  trn='Training: P arrival at a random position in the window; Gaussian labels (\u03c3 = 0.1 s) at the analyst P and S picks; 25 epochs, Adam, batch size 64, cosine learning rate 1e-3 \u2192 1e-5; seeds 42, 43 and 44.',
                  evl='Evaluation: window-level F1-score; a pick is correct when the window fires and the pick lies within 0.1 s or 0.5 s of the analyst pick; the threshold maximizes F1 on the STEAD validation set and is then fixed.'),
      'id': dict(a='(a) Data', b='(b) Pemrosesan satu jendela', tr='Latih: %s', va='%s', te='%s', stead='100.000 gempa + 100.000 derau', n19='19.995 jendela setelah komponen vertikal konstan dikeluarkan', only='Hanya uji: 10.000 gempa + 10.000 derau  (%s jendela)',
                  lg=('latih', 'validasi (ambang dipilih di sini)', 'uji'), notes=('velosimeter', 'jaringan Swiss', 'Texas'), th=('160.000', '20.000', '20.000'), nn=('19.999', '19.999', '20.000'),
                  steps=('Jejak vertikal\n100 Hz', 'Band-pass\n1\u201345 Hz', 'Jendela\n3072 sampel\n(30,72 s)', 'Rerata nol,\nvarians satu', 'TinyPhaseNet\n(format apa pun)', 'Kurva probabilitas\nP, S, dan derau', 'Probabilitas P maks.\n\u2265 ambang?', 'Waktu tiba = sampel\nprobabilitas\nmaksimum'),
                  trn='Pelatihan: kedatangan P pada posisi acak dalam jendela; label Gaussian (\u03c3 = 0,1 s) pada \u0070ick P dan S analis; 25 epoch, Adam, ukuran batch 64, laju belajar cosine 1e-3 \u2192 1e-5; seed 42, 43, dan 44.',
                  evl='Evaluasi: F1-score tingkat jendela; sebuah pick benar bila jendela menyala dan pick berada dalam 0,1 s atau 0,5 s dari pick analis; ambang memaksimalkan F1 pada validasi STEAD lalu dikunci.')}
for lg in ('en', 'id'):
    T = PT[lg]; sfx = '' if lg == 'en' else '_id'; GR = '#475569'; TR, VA, TE = '#475569', '#cbd5e1', '#0f766e'
    fig = plt.figure(figsize=(7.4, 3.7), dpi=300); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis('off')
    ax.text(1, 60.2, T['a'], fontsize=8.4, fontweight='bold', color=INK, va='center'); X0, W_ = 27, 71; y_ = 52.0; x_ = X0
    for frac, col, tc, lab in ((0.8, TR, 'white', T['tr'] % T['th'][0]), (0.1, VA, INK, T['th'][1]), (0.1, TE, 'white', T['th'][2])):
        ax.add_patch(Rectangle((x_, y_), W_ * frac, 4.4, color=col)); ax.text(x_ + W_ * frac / 2, y_ + 2.2, lab, ha='center', va='center', fontsize=6.4 if frac > 0.5 else 5.6, color=tc, fontweight='bold'); x_ += W_ * frac
    ax.text(1, y_ + 3.0, 'STEAD', fontsize=7.4, fontweight='bold', color=INK, va='center'); ax.text(1, y_ + 0.2, T['stead'], fontsize=5.5, color=GR, va='center'); ax.text(X0 + W_ * 0.98, y_ - 1.6, T['n19'], fontsize=5.4, color=GR, ha='right', va='center')
    for k, nm in enumerate(('INSTANCE', 'ETHZ', 'TXED')):
        yy = 42.0 - k * 6.4; ax.text(1, yy + 3.0, nm, fontsize=7.4, fontweight='bold', color=INK, va='center'); ax.text(1, yy + 0.2, T['notes'][k], fontsize=5.5, color=GR, va='center')
        ax.add_patch(Rectangle((X0, yy), W_, 4.4, color=TE)); ax.text(X0 + W_ / 2, yy + 2.2, T['only'] % T['nn'][k], ha='center', va='center', fontsize=6.4, color='white', fontweight='bold')
    for i, (lab, col) in enumerate(zip(T['lg'], (TR, VA, TE))):
        xx = X0 + [0, 13, 46][i]; ax.add_patch(Rectangle((xx, 24.0), 1.8, 1.8, color=col)); ax.text(xx + 2.6, 24.9, lab, fontsize=5.8, va='center', color=INK)
    ax.text(1, 20.2, T['b'], fontsize=8.4, fontweight='bold', color=INK, va='center'); bw, gap = 10.7, 1.7; xs_ = [1 + i * (bw + gap) for i in range(8)]; fcs = ['#e2e8f0'] * 4 + ['#dbeafe', '#e2e8f0', '#fef3c7', '#e2e8f0']
    for txt, col, xx in zip(T['steps'], fcs, xs_):
        ax.add_patch(FancyBboxPatch((xx, 11.8), bw, 6.2, boxstyle='round,pad=0.1,rounding_size=0.8', fc=col, ec='#94a3b8', lw=0.8)); ax.text(xx + bw / 2, 14.9, txt, ha='center', va='center', fontsize=5.3, color=INK)
    for a_, b_ in zip(xs_[:-1], xs_[1:]): ax.annotate('', xy=(b_ - 0.05, 14.9), xytext=(a_ + bw + 0.05, 14.9), arrowprops=dict(arrowstyle='->', color=GR, lw=0.9))
    yy = 8.2
    for sx in (T['trn'], T['evl']):
        for ln in textwrap.wrap(sx, 150): ax.text(1, yy, ln, fontsize=5.7, color=GR, va='center'); yy -= 2.0
        yy -= 0.6
    fig.savefig(OUT + f'fig_pend_protokol{sfx}.pdf'); plt.close(fig)
print(open(OUT + 'tab_ladder.tex').read()); print({k: round(v, 2) for k, v in M.items()})
