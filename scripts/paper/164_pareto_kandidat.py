"""
Step 164 (answer to the advisor, Pareto optimum): analysis of the Pareto front from stored results, with no typed numbers.
Candidates (mean of three seeds): one channel: standard Float32, depthwise Float32, 16x8, INT8, 4/8; three channels: retrained PhaseNet (1b), depthwise Float32, 16x8, INT8, 4/8.
Objective A (all candidates, no device): maximize the F1-score averaged over four datasets; minimize Flash; minimize the minimum activation memory.
Objective B (candidates measured on the ESP32-S3: one-channel INT8 and 16x8, three-channel INT8 and 16x8): maximize the F1-score averaged over four datasets; minimize the energy per window; minimize the latency.
Selection rules (fixed before seeing the results): (R1) the lowest energy that satisfies the STEAD F1-score >= threshold (90, 95, 97, 98) and the mean cross-domain F1-score >= 90; (R2) knee point: the front point farthest from the line between the two ends of the front in the normalized (energy, F1) plane;
(R3) sensitivity: utility U = w * F1_norm - (1 - w) * energy_norm, the winner for w = 0..1. F1 differences smaller than the standard deviation across seeds are marked as indistinguishable.
Output: results/final/pareto_kandidat.json, ../jawaban_prof_arief_fase1/{tabel_spektrum/tab_pareto.tex, grafik_pareto_titik.pdf, grafik_pareto_titik.png}
"""
import json, re, statistics as st
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS = [42, 43, 44]; DS = ['stead_test', 'instance_test', 'ethz_test', 'txed_test']; OUT = '../jawaban_prof_arief_fase1'
S = json.load(open('results/final/spektrum_kuantisasi_3seed.json')); W = json.load(open('results/final/w4a8_perkanal_3seed.json'))
pn = {int(e['model'].split('seed')[1].split('.')[0]): e for e in json.load(open('results/ablation_final/eval_pn.json'))}
std1 = {int(e['model'].split('seed')[1].split('.')[0]): e for e in json.load(open('results/ablation_final/eval_standard.json'))}
q3 = {int(k): v for k, v in json.load(open('results/final/quant16x8_pndw_per_seed.json')).items()}; q1 = {int(k): v for k, v in json.load(open('results/final/quant16x8_per_seed.json')).items()}


def f1s(fam, key, d):   # F1 (persen) per seed
    if key == 'standard': return [100 * std1[s][d]['f1'] for s in SEEDS]
    if key == 'pn': return [100 * pn[s][d]['f1'] for s in SEEDS]
    if key == 'fp32': return [100 * (q3 if fam == '3ch' else q1)[s][d]['fp32']['f1'] for s in SEEDS]
    if key == 'q16x8': return [100 * (q3 if fam == '3ch' else q1)[s][d]['q16x8']['f1'] for s in SEEDS]
    if key == 'int8': return [100 * S[fam][str(s)]['int8']['kualitas'][d]['f1'] for s in SEEDS]
    return [100 * W[fam][str(s)]['pcmse']['kualitas'][d]['f1'] for s in SEEDS]


def parse(path, pats):
    t = open(path).read(); return {k: float(re.search(p, t).group(1)) for k, p in pats.items()}


P1 = {'lat': r'Latensi Invoke \(ms\)\s*\|\s*([\d.]+)', 'e': r'Energi per inferensi \(mJ\)\s*\|\s*([\d.]+)', 'ar': r'Tensor arena terpakai \(KB\)\s*\|\s*([\d.]+)'}
P3 = {'lat': r'Latensi Invoke \(Mean \+- Std\)\s*\|\s*([\d.]+)', 'e': r'Energi per Inferensi\s*\|\s*([\d.]+)', 'ar': r'Tensor Arena Digunakan\s*\|\s*([\d.]+)'}
RH = 'firmware/tinyphasenet_profiling/results_hw/'
H3S = json.load(open('results/final/hw_tiga_seed.json'))   # 175_: mean of three seeds (42, 43, 44) on the ESP32-S3 with the optimized kernel
DEV = {(f, k): dict(lat=H3S[f'{f}_{kk}']['rerata']['latency_ms'], e=H3S[f'{f}_{kk}']['rerata']['energy_mj'], ar=H3S[f'{f}_{kk}']['rerata']['arena_kb']) for f in ('1ch', '3ch') for k, kk in (('int8', 'int8'), ('q16x8', '16x8'), ('fp32', 'fp32'))}
I4 = json.load(open('results/final/int4_nyata.json'))   # REAL 4-bit file size (173_)
NAMA = {'standard': 'standar Float32', 'pn': 'PhaseNet (dilatih ulang)', 'fp32': 'Float32', 'q16x8': '16x8', 'int8': 'INT8', 'w4a8': '4/8 (4-bit)'}
CAND = [('1ch', k) for k in ('standard', 'fp32', 'q16x8', 'int8', 'w4a8')] + [('3ch', k) for k in ('pn', 'fp32', 'q16x8', 'int8', 'w4a8')]
rows = []
for fam, key in CAND:
    f = {d: f1s(fam, key, d) for d in DS}; mean4 = st.mean(st.mean(f[d]) for d in DS); lintas = st.mean(st.mean(f[d]) for d in DS[1:])
    flash = st.mean((I4[fam][str(s)]['bytes_int4'] if key == 'w4a8' else S[fam][str(s)][{'fp32': 'fp32', 'q16x8': 'q16x8', 'int8': 'int8', 'standard': 'standard', 'pn': 'pn'}[key]]['bytes']) / 1024 for s in SEEDS)
    akt = st.mean(S[fam][str(s)][{'w4a8': 'int8'}.get(key, key)]['arena_lb'] / 1024 for s in SEEDS)
    r = dict(fam=fam, key=key, nama=f"{'Satu kanal' if fam == '1ch' else 'Tiga kanal'} {NAMA[key]}", f1_stead=st.mean(f['stead_test']), sd_stead=st.stdev(f['stead_test']), f1_mean4=mean4, f1_lintas=lintas, flash_kb=flash, aktivasi_lb_kb=akt)
    if (fam, key) in DEV: r.update(dev_lat_ms=DEV[(fam, key)]['lat'], dev_energi_mJ=DEV[(fam, key)]['e'], dev_arena_kb=DEV[(fam, key)]['ar'])
    rows.append(r)


def nondom(items, spec):   # spec: list of (key, +1 maximize / -1 minimize); returns the indices that are not dominated
    keep = []
    for i, a in enumerate(items):
        dom = False
        for j, b in enumerate(items):
            if i == j: continue
            ge = all(s * (b[k] - a[k]) >= -1e-12 for k, s in spec); gt = any(s * (b[k] - a[k]) > 1e-12 for k, s in spec)
            if ge and gt: dom = True; break
        if not dom: keep.append(i)
    return keep


A = nondom(rows, [('f1_mean4', +1), ('flash_kb', -1), ('aktivasi_lb_kb', -1)])
dev = [r for r in rows if 'dev_energi_mJ' in r]; B = nondom(dev, [('f1_mean4', +1), ('dev_energi_mJ', -1), ('dev_lat_ms', -1)])
for i, r in enumerate(rows): r['pareto_A'] = i in A
for i, r in enumerate(dev): r['pareto_B'] = i in B
res = dict(kandidat=rows, pareto_A=[rows[i]['nama'] for i in A], pareto_B=[dev[i]['nama'] for i in B])
# R1
res['R1'] = {}
for th in (90, 95, 97, 98):
    feas = [r for r in dev if r['f1_stead'] >= th and r['f1_lintas'] >= 90]
    res['R1'][str(th)] = min(feas, key=lambda r: r['dev_energi_mJ'])['nama'] if feas else None
# R2 knee point on the front B (energy, F1)
fb = sorted([dev[i] for i in B], key=lambda r: r['dev_energi_mJ'])
if len(fb) >= 3:
    e = np.array([r['dev_energi_mJ'] for r in fb]); f = np.array([r['f1_mean4'] for r in fb]); en = (e - e.min()) / (e.max() - e.min()); fn = (f - f.min()) / (f.max() - f.min())
    p0, p1 = np.array([en[0], fn[0]]), np.array([en[-1], fn[-1]]); d = [abs((p1 - p0)[0] * (y - p0[1]) - (p1 - p0)[1] * (x - p0[0])) / np.linalg.norm(p1 - p0) for x, y in zip(en, fn)]
    res['R2_titik_tekuk'] = fb[int(np.argmax(d))]['nama']; res['R2_jarak'] = [round(float(v), 3) for v in d]
# R3 weight sensitivity
e = np.array([r['dev_energi_mJ'] for r in dev]); f = np.array([r['f1_mean4'] for r in dev]); en = (e - e.min()) / (e.max() - e.min()); fn = (f - f.min()) / (f.max() - f.min()); win = []
for w in np.linspace(0, 1, 21): win.append((round(float(w), 2), dev[int(np.argmax(w * fn - (1 - w) * en))]['nama']))
res['R3'] = win
iv = [];
for w, n in win:
    if iv and iv[-1][2] == n: iv[-1][1] = w
    else: iv.append([w, w, n])
res['R3_interval'] = iv
json.dump(res, open('results/final/pareto_kandidat.json', 'w'), indent=1, ensure_ascii=False)
print('Pareto A (F1, Flash, aktivasi):', res['pareto_A']); print('Pareto B (F1, energi, latensi; di papan):', res['pareto_B']); print('R1:', res['R1']); print('R2:', res.get('R2_titik_tekuk'), res.get('R2_jarak')); print('R3:', iv)
# table
tt = ''
for r in rows:
    tt += f"{r['nama']} & {r['f1_stead']:.2f} & {r['f1_lintas']:.2f} & {r['flash_kb']:,.1f} & {'ya' if r['pareto_A'] else 'tidak'} \\\\\n"
open(f'{OUT}/tabel_spektrum/tab_pareto.tex', 'w').write(re.sub(r'(?<=\d)\.(?=\d)', ',', tt.replace(',', '_')).replace('_', '.') + '\\bottomrule\n')
# table of the candidates measured on the device (six), with latency and energy
td = ''
for r in dev: td += f"{r['nama']} & {r['f1_stead']:.2f} & {r['f1_mean4']:.2f} & {r['dev_lat_ms']:,.1f} & {r['dev_energi_mJ']:,.1f} & {r['dev_arena_kb']:,.1f} & {'ya' if r['pareto_B'] else 'tidak'} \\\\\n"
open(f'{OUT}/tabel_spektrum/tab_pareto_perangkat.tex', 'w').write(re.sub(r'(?<=\d)\.(?=\d)', ',', td.replace(',', '_')).replace('_', '.') + '\\bottomrule\n')
# figure
fig, axs = plt.subplots(2, 2, figsize=(8.4, 7.0), dpi=300); ax = axs.flat; c = {'1ch': '#1d4ed8', '3ch': '#0f766e'}
ypos = list(range(len(rows)))[::-1]; names = [r['nama'] for r in rows]
def barpanel(a, vals, flags, fmt, xmax, title, xlabel):
    for y, r, v, fl in zip(ypos, rows, vals, flags):
        if v is None:
            a.text(0.01 * xmax, y, 'tidak diukur', va='center', fontsize=7, color='#64748b', style='italic'); continue
        a.barh(y, v, height=0.68, color=c[r['fam']], alpha=1 if fl else 0.3, hatch=None if fl else '////', edgecolor='#0f172a', linewidth=0.8 if fl else 0.4, zorder=3)
        a.text(v + 0.012 * xmax, y, fmt(v), va='center', fontsize=7.4, fontweight='bold' if fl else 'normal', color='#0f172a')
    a.set_xlim(0, xmax); a.set_ylim(min(ypos) - 0.7, max(ypos) + 0.95); a.set_title(title, fontsize=8.8, fontweight='bold', pad=6); a.set_xlabel(xlabel, fontsize=8); a.grid(axis='x', alpha=0.3, ls='--', zorder=0); a.tick_params(labelsize=7.5)
barpanel(ax[0], [r['f1_mean4'] for r in rows], [r['pareto_A'] for r in rows], lambda v: f'{v:.1f}'.replace('.', ','), 118, '(a) F1 rerata empat dataset', 'F1 (%)')
barpanel(ax[1], [r['flash_kb'] for r in rows], [r['pareto_A'] for r in rows], lambda v: f'{v:,.0f}'.replace(',', '.'), 1400, '(b) Ukuran berkas (Flash)', 'KB')
barpanel(ax[2], [r.get('dev_lat_ms') for r in rows], [r.get('pareto_B', False) for r in rows], lambda v: f'{v:,.0f}'.replace(',', '.'), 3100, '(c) Latensi per jendela di ESP32-S3', 'ms')
barpanel(ax[3], [r.get('dev_energi_mJ') for r in rows], [r.get('pareto_B', False) for r in rows], lambda v: f'{v:,.0f}'.replace(',', '.'), 1500, '(d) Energi per jendela di ESP32-S3', 'mJ')
for a_ in ax: a_.set_yticks(ypos); a_.set_yticklabels(names, fontsize=7.8)
from matplotlib.patches import Patch
h = [Patch(facecolor=c['1ch'], edgecolor='#0f172a', label='Satu kanal (Model 1-Kanal depthwise)'), Patch(facecolor=c['3ch'], edgecolor='#0f172a', label='Tiga kanal (PhaseNet depthwise)'), Patch(facecolor='#94a3b8', edgecolor='#0f172a', label='tidak didominasi (warna pekat)'), Patch(facecolor='#e2e8f0', edgecolor='#0f172a', hatch='////', label='didominasi (pucat, berarsir)')]
fig.legend(handles=h, loc='upper center', ncol=2, fontsize=7.8, frameon=False, bbox_to_anchor=(0.5, 1.02)); plt.tight_layout(rect=[0, 0, 1, 0.94], h_pad=1.6, w_pad=1.2)
fig.savefig(f'{OUT}/grafik_pareto_titik.pdf', bbox_inches='tight'); fig.savefig(f'{OUT}/grafik_pareto_titik.png', bbox_inches='tight'); print('figures and tables done')

# ---- manuscript paragraphs generated from the results ----
g = lambda n: next(r for r in rows if r['nama'] == n); dec = lambda t: re.sub(r'(?<=\d)\.(?=\d)', ',', t)
tak = [r['nama'] for r in rows if not r['pareto_A']]; i3, q1c, q3c, i1, f1c = g('Tiga kanal INT8'), g('Satu kanal 16x8'), g('Tiga kanal 16x8'), g('Satu kanal INT8'), g('Satu kanal Float32')
iv_ = res['R3_interval']; mid = next((x for x in iv_ if x[2] == 'Satu kanal 16x8'), iv_[0])
th = lambda x: f'{x:,.0f}'.replace(',', '§')   # temporary thousands separator so that the commas of the sentence do not change
def bandingkan(a, b):   # comparison sentence of the two one-channel formats from the measured numbers
    ar = lambda x, y: 'lebih rendah' if x < y else ('lebih tinggi' if x > y else 'sama')
    return (f"Float32 dan 16x8 satu kanal memakai arsitektur yang sama, jadi perbandingannya langsung. {a['nama']} mendapat F1 rerata {a['f1_mean4']:.2f}\\%, energi {th(a['dev_energi_mJ'])}~mJ, dan latensi {th(a['dev_lat_ms'])}~ms. "
            f"{b['nama']} mendapat F1 rerata {b['f1_mean4']:.2f}\\% ({ar(b['f1_mean4'], a['f1_mean4'])}), energi {th(b['dev_energi_mJ'])}~mJ ({ar(b['dev_energi_mJ'], a['dev_energi_mJ'])}), dan latensi {th(b['dev_lat_ms'])}~ms ({ar(b['dev_lat_ms'], a['dev_lat_ms'])}). "
            f"Ukuran berkasnya {a['flash_kb']:.0f}~KB dan {b['flash_kb']:.0f}~KB, sedangkan memori aktivasi minimumnya {a['aktivasi_lb_kb']:.0f}~KB dan {b['aktivasi_lb_kb']:.0f}~KB.")


grup = []   # STEAD F1 thresholds that give the same R1 winner
for t in ('90', '95', '97', '98'):
    if grup and grup[-1][0] == res['R1'][t]: grup[-1][1].append(t)
    else: grup.append([res['R1'][t], [t]])
r1txt = ' '.join(f"Pada ambang {t[0]}\\%, pemenangnya {n}." if len(t) == 1 else f"Pada ambang {t[0]}--{t[-1]}\\%, pemenangnya {n}." for n, t in grup)
bdom = [r['nama'] for r in dev if r['nama'] not in res['pareto_B']]
flip = ("Satu kanal Float32 berubah dari didominasi (tanpa perangkat) menjadi tidak didominasi (dengan perangkat), sebab latensi dan energi baru terlihat setelah diukur di perangkat. Analisis tanpa perangkat saja akan menyimpulkan bahwa 16x8 lebih baik daripada Float32 pada semua sasaran." if ('Satu kanal Float32' in tak and 'Satu kanal Float32' in res['pareto_B'] and 'Satu kanal 16x8' not in tak) else '')
box = ("\n\n\\hypertarget{kotak:pareto_perangkat}{}\n\\begin{kotakcatatan}{Tanpa dan dengan data perangkat: kesimpulan Float32 berbalik}\n\\small\n" + flip + "\n\\end{kotakcatatan}\n") if flip else ''
r1dom = ' '.join(f"{n} menang pada aturan ini walau didominasi pada analisis dengan perangkat, karena aturan pertama hanya memperhitungkan energi dan ambang F1." for n in dict.fromkeys(n for n, _ in grup if n in bdom)); r1dom = (r1dom + ' ') if r1dom else ''
tex = (f"Ada sepuluh kandidat, lima dari satu kanal dan lima dari tiga kanal, semuanya rerata tiga \\emph{{seed}} (Tabel~\\ref{{tab:pareto}}, Gambar~\\ref{{fig:pareto}}). "
       f"Sebuah kandidat \\emph{{didominasi}} bila ada kandidat lain yang tidak lebih buruk pada semua sasaran dan lebih baik pada setidaknya satu sasaran. Kandidat yang tidak didominasi berada di batas Pareto, dan hanya kandidat inilah yang masuk akal dipilih.\n\n"
       f"Tanpa data perangkat, sasarannya adalah F1 rerata empat dataset, Flash, dan memori aktivasi minimum. Hasilnya, {len(A)} kandidat tidak didominasi dan {len(tak)} didominasi ({', '.join(tak)}). " + ("Satu kanal Float32 didominasi di sini karena berkas dan memori aktivasinya lebih besar daripada 16x8 dengan F1 yang setara. " if ('Satu kanal Float32' in tak and 'Satu kanal 16x8' not in tak) else '') +
       f"Dengan data perangkat ESP32-S3, sasarannya menjadi F1 rerata, energi per jendela, dan waktu (Tabel~\\ref{{tab:pareto_perangkat}}). Hasilnya, {len(res['pareto_B'])} dari {len(dev)} kandidat berada di batas Pareto ({', '.join(res['pareto_B'])})" + (f", dan yang didominasi adalah {', '.join(bdom)}. " if bdom else ", sehingga tidak ada yang didominasi. ") + box + "\n\n"
       + f"Kandidat dipilih dengan tiga aturan yang ditetapkan sebelum hasilnya dilihat. Aturan pertama memilih energi terkecil yang memenuhi ambang F1 (F1 STEAD sekurang-kurangnya sebesar ambang dan F1 lintas-domain sekurang-kurangnya 90\\%). Aturan kedua memilih titik tekuk batas Pareto, yaitu titik yang paling jauh dari garis lurus antara kedua ujung batas. Aturan ketiga memakai skor berbobot antara akurasi dan energi (bobot 0 hanya energi, bobot 1 hanya akurasi). "
       f"Hasil aturan pertama: {r1txt} {r1dom}Hasil aturan kedua: titik tekuk batas adalah {res['R2_titik_tekuk']}. Hasil aturan ketiga: " + ', '.join(f"{n} menang untuk bobot {a:.2f}" + (f" sampai {b:.2f}" if b != a else '') for a, b, n in iv_) + ".\n\n"
       f"Tidak ada satu kandidat yang unggul pada semua aturan. Pilihan akhir bergantung pada ambang F1 yang dituntut dan pada kendala memori, yaitu ukuran berkas dan memori aktivasi.")
open(f'{OUT}/tabel_spektrum/par_pareto.tex', 'w').write(dec(tex).replace('§', '.') + '\n')
print('paragraphs done')
