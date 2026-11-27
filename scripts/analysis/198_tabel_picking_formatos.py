"""
Langkah 198 (draf pendamping kompresi): menggabungkan hasil 197_ (picking P dan S semua model dan format, seed 42-44) menjadi tabel dan makro.
Pemeriksaan konsistensi: baris satu kanal depthwise separable Float32 harus sama dengan picking_per_seed.json (84_picking_final.py) untuk seed 42-44 (model dan protokol sama).
Keluaran: ../draft_publikasi_1/tabel_pend/{tab_pickfmt_05,tab_pickfmt_01}{,_id}.tex (Recall@0,5 s dan @0,1 s; kolom P: STEAD, INSTANCE, ETHZ, TXED; S: STEAD, TXED; rerata +- simpangan baku atas tiga seed),
numbers_pickfmt{,_id}.tex (makro angka), results/final/picking_formatos.json (gabungan).
Pemakaian (dari folder ini): python 198_tabel_picking_formatos.py
"""
import json, glob, statistics as st
OUT = '../draft_publikasi_1/tabel_pend/'
SEEDS = ['42', '43', '44']; DS = ['stead_test', 'instance_test', 'ethz_test', 'txed_test']
R = {}
for f in sorted(glob.glob('results/final/picking_formatos_*.json')):
    for fam, v in json.load(open(f)).items(): R.setdefault(fam, {}).update(v)
PQ = json.load(open('results/final/ptq_padrao_3seed.json'))   # 199_: model konvolusi standar terkuantisasi (picking per set di r['picking'])
for fam in ('1ch', '3ch'):
    for key, fmt in (('s_q16x8', 'q16x8'), ('s_int8', 'int8'), ('s_w4a8', 'w4a8'), ('s_w4a16', 'w4a16')):
        R[fam][key] = {s: {d: PQ[fam][fmt][s]['picking'][d] for d in DS} for s in SEEDS}
    R[fam]['w4a16'] = {s: {d: json.load(open('results/final/picking_w4a16_separable.json'))[fam][s][d] for d in DS} for s in SEEDS}   # 203_: 4/16 separable
for fam in ('1ch', '3ch'):
    for key in ('standard', 's_q16x8', 's_w4a16', 's_int8', 's_w4a8', 'fp32', 'q16x8', 'w4a16', 'int8', 'w4a8'):
        for s in SEEDS:
            for d in DS:
                assert d in R[fam][key][s], f'hasil belum lengkap: {fam} {key} seed {s} {d}'
json.dump(R, open('results/final/picking_formatos.json', 'w'), indent=1)
# konsistensi dengan 84_ (satu kanal depthwise Float32)
P84 = json.load(open('results/final/picking_per_seed.json')); maks = 0.0
for s in SEEDS:
    for d in DS:
        for ph in ('P', 'S'):
            if ph in R['1ch']['fp32'][s][d]:
                for m in ('recall_0p1', 'recall_0p5'): maks = max(maks, abs(R['1ch']['fp32'][s][d][ph][m] - P84[s][d][ph][m]))
print(f'konsistensi 1ch fp32 terhadap 84_: selisih Recall maksimum = {maks:.2e}')
NAMES = {'en': {'1ch': 'Single-channel', '3ch': 'Three-channel', 'standard': 'standard Float32', 's_q16x8': 'standard 8/16', 's_int8': 'standard 8/8', 's_w4a8': 'standard 4/8 (emulated)', 's_w4a16': 'standard 4/16 (emulated)', 'w4a16': 'separable 4/16 (emulated)', 'fp32': 'separable Float32', 'q16x8': 'separable 8/16', 'int8': 'separable 8/8', 'w4a8': 'separable 4/8 (emulated)'},
         'id': {'1ch': 'Kanal tunggal', '3ch': 'Tiga kanal', 'standard': 'standar Float32', 's_q16x8': 'standar 8/16', 's_int8': 'standar 8/8', 's_w4a8': 'standar 4/8 (emulasi)', 's_w4a16': 'standar 4/16 (emulasi)', 'w4a16': '\\textit{separable} 4/16 (emulasi)', 'fp32': '\\textit{separable} Float32', 'q16x8': '\\textit{separable} 8/16', 'int8': '\\textit{separable} 8/8', 'w4a8': '\\textit{separable} 4/8 (emulasi)'}}


def cell(fam, key, d, ph, m, dec):
    xs = [100 * R[fam][key][s][d][ph][m] for s in SEEDS]
    return f"{st.mean(xs):.1f} $\\pm$ {st.stdev(xs):.1f}".replace('.', dec)


COLS = [('stead_test', 'P'), ('instance_test', 'P'), ('ethz_test', 'P'), ('txed_test', 'P'), ('stead_test', 'S'), ('txed_test', 'S')]
for lg, sfx, dec in (('en', '', '.'), ('id', '_id', ',')):
    N = NAMES[lg]
    for m, nm in (('recall_0p5', '05'), ('recall_0p1', '01')):
        rows = ''
        for fam in ('1ch', '3ch'):
            for key in ('standard', 's_q16x8', 's_w4a16', 's_int8', 's_w4a8', 'fp32', 'q16x8', 'w4a16', 'int8', 'w4a8'):
                rows += f"{N[fam]} {N[key]} & " + ' & '.join(cell(fam, key, d, ph, m, dec) for d, ph in COLS) + ' \\\\\n'
            if fam == '1ch': rows += '\\midrule\n'
        open(OUT + f'tab_pickfmt_{nm}{sfx}.tex', 'w').write(rows + '\\bottomrule\n')
# makro angka
mean = lambda fam, key, d, ph, m: st.mean(100 * R[fam][key][s][d][ph][m] for s in SEEDS)
M = {}
for fam, t in (('1ch', 'One'), ('3ch', 'Thr')):
    for key, kt in (('standard', 'Std'), ('fp32', 'Fp'), ('q16x8', 'Qq'), ('int8', 'Int'), ('w4a8', 'Four')):
        M[f'PfSteadP{kt}{t}'] = mean(fam, key, 'stead_test', 'P', 'recall_0p5'); M[f'PfSteadS{kt}{t}'] = mean(fam, key, 'stead_test', 'S', 'recall_0p5')
    # kehilangan Recall@0,5 s kedatangan P format 8/16 terhadap Float32 (nilai mutlak terbesar di antara dataset)
    M[f'PfQqDiffMax{t}'] = max(abs(mean(fam, 'q16x8', d, 'P', 'recall_0p5') - mean(fam, 'fp32', d, 'P', 'recall_0p5')) for d in DS)
    M[f'PfQqSDiffMax{t}'] = max(abs(mean(fam, 'q16x8', d, 'S', 'recall_0p5') - mean(fam, 'fp32', d, 'S', 'recall_0p5')) for d in ('stead_test', 'txed_test'))
    M[f'PfIntDropMin{t}'] = min(mean(fam, 'fp32', d, 'P', 'recall_0p5') - mean(fam, 'int8', d, 'P', 'recall_0p5') for d in DS); M[f'PfIntDropMax{t}'] = max(mean(fam, 'fp32', d, 'P', 'recall_0p5') - mean(fam, 'int8', d, 'P', 'recall_0p5') for d in DS)
    M[f'PfFourDropMin{t}'] = min(mean(fam, 'fp32', d, 'P', 'recall_0p5') - mean(fam, 'w4a8', d, 'P', 'recall_0p5') for d in DS); M[f'PfFourDropMax{t}'] = max(mean(fam, 'fp32', d, 'P', 'recall_0p5') - mean(fam, 'w4a8', d, 'P', 'recall_0p5') for d in DS)
M['PfStdGapMaxOne'] = max(mean('1ch', 'standard', d, 'P', 'recall_0p5') - mean('1ch', 'fp32', d, 'P', 'recall_0p5') for d in DS)
M['PfStdGapMaxThr'] = max(mean('3ch', 'standard', d, 'P', 'recall_0p5') - mean('3ch', 'fp32', d, 'P', 'recall_0p5') for d in DS)
for fam, t in (('1ch', 'One'), ('3ch', 'Thr')):   # simpangan baku antar seed terbesar untuk Recall@0,5 s kedatangan P (INT8 dan 4/8)
    M[f'PfSdMax{t}'] = max(st.stdev(100 * R[fam][key][s][d]['P']['recall_0p5'] for s in SEEDS) for key in ('int8', 'w4a8') for d in DS)
M['PfStdGapOne'] = min(mean('1ch', 'standard', d, 'P', 'recall_0p5') - mean('1ch', 'fp32', d, 'P', 'recall_0p5') for d in DS)
M['PfStdGapThr'] = min(mean('3ch', 'standard', d, 'P', 'recall_0p5') - mean('3ch', 'fp32', d, 'P', 'recall_0p5') for d in DS)
for fam, t in (('1ch', 'One'), ('3ch', 'Thr')):   # model standar terkuantisasi: penurunan Recall@0,5 s P terhadap standar Float32 (rentang atas empat dataset) dan selisih maksimum 8/16
    dr = lambda key: [mean(fam, 'standard', d, 'P', 'recall_0p5') - mean(fam, key, d, 'P', 'recall_0p5') for d in DS]
    M[f'PfSdIntDropMin{t}'] = min(dr('s_int8')); M[f'PfSdIntDropMax{t}'] = max(dr('s_int8')); M[f'PfSdFourDropMin{t}'] = min(dr('s_w4a8')); M[f'PfSdFourDropMax{t}'] = max(dr('s_w4a8'))
    M[f'PfSdWaDropMin{t}'] = min(dr('s_w4a16')); M[f'PfSdWaDropMax{t}'] = max(dr('s_w4a16'))
    drs = [mean(fam, 'fp32', d, 'P', 'recall_0p5') - mean(fam, 'w4a16', d, 'P', 'recall_0p5') for d in DS]; M[f'PfWaDropMin{t}'] = min(drs); M[f'PfWaDropMax{t}'] = max(drs)   # 4/16 separable (203_)
    M[f'PfSdQqDiffMax{t}'] = max(abs(x) for x in dr('s_q16x8'))
    M[f'PfSdQqSDiffMax{t}'] = max(abs(mean(fam, 'standard', d, 'S', 'recall_0p5') - mean(fam, 's_q16x8', d, 'S', 'recall_0p5')) for d in ('stead_test', 'txed_test'))
    M[f'PfSdSteadSOne{t}' if False else f'PfSdIntSDiffMax{t}'] = max(abs(mean(fam, 'standard', d, 'S', 'recall_0p5') - mean(fam, 's_int8', d, 'S', 'recall_0p5')) for d in ('stead_test', 'txed_test'))
for sfx, dec in (('', '.'), ('_id', ',')):
    open(OUT + f'numbers_pickfmt{sfx}.tex', 'w').write(''.join(f'\\newcommand{{\\dv{k}}}{{{v:.1f}}}\n'.replace('.', dec) if False else f'\\newcommand{{\\dv{k}}}{{{(f"{v:.1f}").replace(".", dec)}}}\n' for k, v in M.items()))
# tabel selisih ringkas: perubahan Recall@0,5 s kedatangan P (rerata empat dataset, poin) terhadap Float32 model SAMA dan seed sama; rerata +- simpangan baku sampel atas tiga seed
KEYS = {'separable': {'fp32': 'fp32', 'q16x8': 'q16x8', 'w4a16': 'w4a16', 'int8': 'int8', 'w4a8': 'w4a8'}, 'standar': {'fp32': 'standard', 'q16x8': 's_q16x8', 'w4a16': 's_w4a16', 'int8': 's_int8', 'w4a8': 's_w4a8'}}
BITS = {'q16x8': '8/16', 'w4a16': '4/16', 'int8': '8/8', 'w4a8': '4/8'}   # nama menurut bit bobot/aktivasi; padanan TFLite: 16x8 = 8/16, INT8 = 8/8


def mp(fam, conv, fmt, s_): return 100 * st.mean(R[fam][KEYS[conv][fmt]][s_][d]['P']['recall_0p5'] for d in DS)


def pm_(xs, dec):
    m_ = st.mean(xs); t_ = '0.0' if abs(m_) < 0.05 else f'{m_:+.1f}'.replace('-', '$-$')
    return f"{t_} $\\pm$ {st.stdev(xs):.1f}".replace('.', dec)


for sfx, dec in (('', '.'), ('_id', ',')):
    rows = ''
    for fmt in ('q16x8', 'w4a16', 'int8', 'w4a8'):
        rows += BITS[fmt].replace('/', ' & ') + ' & ' + ' & '.join(pm_([mp(fam, conv, fmt, s_) - mp(fam, conv, 'fp32', s_) for s_ in SEEDS], dec) for fam in ('1ch', '3ch') for conv in ('separable', 'standar')) + ' \\\\\n'
    open(OUT + f'tab_pickdelta{sfx}.tex', 'w').write(rows + '\\bottomrule\n')
print(open(OUT + 'tab_pickfmt_05.tex').read()); print(open(OUT + 'numbers_pickfmt.tex').read())
