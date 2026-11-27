"""
Langkah 193 (draf artikel perangkat Q, tahap 4): analisis Pareto atas DELAPAN konfigurasi yang terukur di ESP32-S3: enam depthwise separable (satu kanal dan tiga kanal; Float32, 16x8, INT8)
dan dua model konvolusi standar yang dilatih sendiri dengan resep yang sama (satu kanal standar = arsitektur TinyPhaseNet dengan Conv1D biasa; tiga kanal standar = arsitektur PhaseNet
asli yang dilatih ulang), keduanya diukur hanya dalam Float32 (192_). Sasaran: maksimalkan F1 rerata empat dataset (STEAD, INSTANCE, ETHZ, TXED); minimalkan energi per jendela; minimalkan latensi.
(a) batas Pareto pada rerata tiga seed; (b) ketahanan terhadap seed: tiap konfigurasi mengambil nilai salah satu dari ketiga seed-nya pada seluruh 3^8 = 6561 kombinasi, dilaporkan proporsi kombinasi
tempat tiap konfigurasi tak terdominasi; (c) aturan pemilihan yang sama dengan 164_ (R1 energi terendah dengan F1 STEAD >= ambang dan F1 rerata tiga dataset lain >= 90; R2 titik tekuk batas;
R3 kepekaan bobot utilitas), dihitung ulang pada delapan konfigurasi. Tidak mengubah 164_ (milik naskah jawaban).
Sumber: results/final/{spektrum_kuantisasi_3seed,quant16x8_per_seed,quant16x8_pndw_per_seed,hw_tiga_seed,hw_padrao_tiga_seed,pareto_kandidat}.json, results/ablation_final/eval_{standard,pn}.json.
Keluaran: results/final/pareto_delapan.json, ../draft_publikasi_1/tabel_dev/numbers_par{,_id}.tex (makro angka), dan ringkasan di layar.
Pemakaian (dari folder ini): python 193_pareto_delapan.py
"""
import json, itertools
import numpy as np

SEEDS = [42, 43, 44]; DS = ['stead_test', 'instance_test', 'ethz_test', 'txed_test']
S = json.load(open('results/final/spektrum_kuantisasi_3seed.json'))
q3 = {int(k): v for k, v in json.load(open('results/final/quant16x8_pndw_per_seed.json')).items()}
q1 = {int(k): v for k, v in json.load(open('results/final/quant16x8_per_seed.json')).items()}
H = json.load(open('results/final/hw_tiga_seed.json')); PD = json.load(open('results/final/hw_padrao_tiga_seed.json'))
std1 = {int(e['model'].split('seed')[1].split('.')[0]): e for e in json.load(open('results/ablation_final/eval_standard.json'))}
pn = {int(e['model'].split('seed')[1].split('.')[0]): e for e in json.load(open('results/ablation_final/eval_pn.json'))}
V6 = ['1ch_fp32', '1ch_16x8', '1ch_int8', '3ch_fp32', '3ch_16x8', '3ch_int8']; V8 = V6 + ['1ch_std', '3ch_std']
NAME = {'1ch_fp32': 'Satu kanal Float32', '1ch_16x8': 'Satu kanal 16x8', '1ch_int8': 'Satu kanal INT8', '3ch_fp32': 'Tiga kanal Float32', '3ch_16x8': 'Tiga kanal 16x8', '3ch_int8': 'Tiga kanal INT8',
        '1ch_std': 'Satu kanal standar Float32', '3ch_std': 'Tiga kanal standar Float32'}


def f1_seed(v, d, s):   # F1 (persen)
    if v == '1ch_std': return 100 * std1[s][d]['f1']
    if v == '3ch_std': return 100 * pn[s][d]['f1']
    fam, fmt = v.split('_'); k = {'fp32': 'fp32', '16x8': 'q16x8', 'int8': 'int8'}[fmt]
    if k == 'int8': return 100 * S[fam][str(s)]['int8']['kualitas'][d]['f1']
    return 100 * (q3 if fam == '3ch' else q1)[s][d][k]['f1']


def cost(v, s):
    h = PD[v[:3]][str(s)] if v.endswith('std') else H[v][str(s)]
    return h['latency_ms'], h['energy_mj']


val = {}
for v in V8:
    for s in SEEDS:
        lat, en = cost(v, s)
        val[(v, s)] = dict(f1=float(np.mean([f1_seed(v, d, s) for d in DS])), f1_stead=f1_seed(v, 'stead_test', s), f1_lintas=float(np.mean([f1_seed(v, d, s) for d in DS[1:]])), lat=lat, en=en)
mean = {v: {k: float(np.mean([val[(v, s)][k] for s in SEEDS])) for k in ('f1', 'f1_stead', 'f1_lintas', 'lat', 'en')} for v in V8}


def nondominated(pts):   # pts: {nama: (f1, lat, en)}
    nd = set()
    for a, (fa, la, ea) in pts.items():
        if not any((fb >= fa and lb <= la and eb <= ea) and (fb > fa or lb < la or eb < ea) for b, (fb, lb, eb) in pts.items() if b != a): nd.add(a)
    return nd


tup = lambda d: (d['f1'], d['lat'], d['en'])
front8 = nondominated({v: tup(mean[v]) for v in V8}); front6 = nondominated({v: tup(mean[v]) for v in V6})
cnt = {v: 0 for v in V8}; n = 0
for combo in itertools.product(SEEDS, repeat=len(V8)):
    for v in nondominated({V8[i]: tup(val[(V8[i], combo[i])]) for i in range(len(V8))}): cnt[v] += 1
    n += 1
cnt_same = {v: 0 for v in V8}
for s in SEEDS:
    for v in nondominated({v: tup(val[(v, s)]) for v in V8}): cnt_same[v] += 1

# R1: energi terendah yang memenuhi F1 STEAD >= ambang dan F1 rerata tiga dataset lintas-domain >= 90
R1 = {}
for th in (90, 95, 97, 98):
    feas = [v for v in V8 if mean[v]['f1_stead'] >= th and mean[v]['f1_lintas'] >= 90]; R1[str(th)] = min(feas, key=lambda v: mean[v]['en']) if feas else None
# R2: titik tekuk pada batas (energi, F1), jarak terjauh dari garis antara dua ujung batas (dinormalisasi)
fb = sorted(front8, key=lambda v: mean[v]['en']); R2 = None; R2d = []
if len(fb) >= 3:
    e = np.array([mean[v]['en'] for v in fb]); f = np.array([mean[v]['f1'] for v in fb]); en_ = (e - e.min()) / (e.max() - e.min()); fn_ = (f - f.min()) / (f.max() - f.min())
    p0, p1 = np.array([en_[0], fn_[0]]), np.array([en_[-1], fn_[-1]]); d = [abs(np.cross(p1 - p0, np.array([x, y]) - p0)) / np.linalg.norm(p1 - p0) for x, y in zip(en_, fn_)]
    R2 = fb[int(np.argmax(d))]; R2d = [round(float(x), 3) for x in d]
# R3: utilitas U = w * F1_norm - (1 - w) * energi_norm
e = np.array([mean[v]['en'] for v in V8]); f = np.array([mean[v]['f1'] for v in V8]); en_ = (e - e.min()) / (e.max() - e.min()); fn_ = (f - f.min()) / (f.max() - f.min())
win = [(round(float(w), 2), V8[int(np.argmax(w * fn_ - (1 - w) * en_))]) for w in np.linspace(0, 1, 21)]
iv = []
for w, nm_ in win:
    if iv and iv[-1][2] == nm_: iv[-1][1] = w
    else: iv.append([w, w, nm_])

out = dict(nama=NAME, rerata=mean, per_seed={f'{v}_{s}': val[(v, s)] for (v, s) in val}, batas_8_rerata=sorted(front8, key=lambda v: V8.index(v)), batas_6_rerata=sorted(front6, key=lambda v: V8.index(v)),
           n_kombinasi=n, proporsi_tak_terdominasi={v: cnt[v] / n for v in V8}, tak_terdominasi_seed_sama={v: cnt_same[v] for v in V8}, R1=R1, R2_titik_tekuk=R2, R2_jarak=R2d, R3_interval=iv)
json.dump(out, open('results/final/pareto_delapan.json', 'w'), indent=1, ensure_ascii=False)
# makro angka untuk naskah (Inggris: titik desimal; Indonesia: koma desimal)
M = {'ParNComb': (n, 0), 'FmStdOne': (mean['1ch_std']['f1'], 1), 'FmStdThr': (mean['3ch_std']['f1'], 1)}
for sfx, dec in (('', '.'), ('_id', ',')):
    open(f'../draft_publikasi_1/tabel_dev/numbers_par{sfx}.tex', 'w').write(''.join(f'\\newcommand{{\\dv{k}}}{{{(f"{v:,.{d}f}").replace(",", "#").replace(".", dec).replace("#", "." if dec == "," else ",")}}}\n' for k, (v, d) in M.items()))
print('mean (F1 averaged over four datasets; STEAD F1; cross-domain F1; latency ms; energy mJ):')
for v in V8: m = mean[v]; print(f"  {v:9s} {m['f1']:6.2f} {m['f1_stead']:6.2f} {m['f1_lintas']:6.2f} {m['lat']:8.1f} {m['en']:8.1f}")
print('front (8) on the mean:', out['batas_8_rerata'], '| front (6) on the mean:', out['batas_6_rerata'])
print(f'proportion of {n} kombinasi seed tempat konfigurasi tak terdominasi:', {v: round(100 * cnt[v] / n, 1) for v in V8})
print('non-dominated on same-seed (of 3):', cnt_same)
print('R1:', R1); print('R2 titik tekuk:', R2, R2d, '(batas urut energi:', fb, ')'); print('R3:', iv)
