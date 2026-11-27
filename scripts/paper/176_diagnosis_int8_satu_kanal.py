"""
Step 176 (answer to the advisor, Subsection 3.3 and Appendix E): diagnosis of one-channel INT8 on the ESP32-S3. Compares the per-window score on the board with the Python TFLite score (100 STEAD windows, first pass) for seeds 42, 43, 44.
Sources: diagnostic firmware logs (env int8_skor; lines 'SKOR <window> <board> <python> <label>'), copied to firmware/tinyphasenet_profiling/results_hw/sesi_07nov_skor_int8_1ch_seed<seed>.log.
Usage (from this folder): python 176_diagnosis_int8_satu_kanal.py
Output: results/final/diagnosis_int8_1ch.json; the summary is printed.
"""
import re, json, numpy as np
D = 'firmware/tinyphasenet_profiling/results_hw/'; THR = {'42': json.load(open('results/final/spektrum_kuantisasi_3seed.json'))['1ch']['42']['int8']['thr'], '43': json.load(open('results/final/spektrum_kuantisasi_3seed.json'))['1ch']['43']['int8']['thr'], '44': json.load(open('results/final/spektrum_kuantisasi_3seed.json'))['1ch']['44']['int8']['thr']}
OUT = {}
for sd in ('42', '43', '44'):
    rows = [(int(m[1]), float(m[2]), float(m[3]), int(m[4])) for m in re.finditer(r'^SKOR (\d+) ([\d.]+) ([\d.]+) (\d)', open(D + f'sesi_07nov_skor_int8_1ch_seed{sd}.log', errors='ignore').read().replace('\r', ''), re.M)]
    assert len(rows) == 100, (sd, len(rows))
    step = 1 / 256; q = lambda v: np.round(np.array(v) / step) * step   # the INT8 output is scaled by 1/256; printing 6 decimals rounds 0.00390625 to 0.003906, so it is restored to a multiple of 1/256
    b = q([r[1] for r in rows]); p = q([r[2] for r in rows]); y = np.array([r[3] for r in rows]); t = THR[sd]
    d = b - p; pb, pp = b >= t, p >= t
    f1 = lambda pr: (2 * (pr & (y == 1)).sum() / max(1, 2 * (pr & (y == 1)).sum() + (pr & (y == 0)).sum() + (~pr & (y == 1)).sum()))
    near = np.abs(p - t) <= 2 * step      # windows whose Python score is within two quantization steps of the threshold
    OUT[sd] = dict(thr=t, thr_langkah=t / step, selisih_maks=float(np.abs(d).max()), selisih_median=float(np.median(np.abs(d))), selisih_rerata=float(np.abs(d).mean()),
                   skor_sama_persis=int((np.abs(d) < 1e-9).sum()), keputusan_beda=int((pb != pp).sum()), dekat_ambang=int(near.sum()), beda_dan_dekat=int(((pb != pp) & near).sum()),
                   f1_papan=float(f1(pb)), f1_python=float(f1(pp)), skor_python_nol=int((p == 0).sum()), skor_papan_nol=int((b == 0).sum()),
                   p_nonnol_gempa=int(((p > 0) & (y == 1)).sum()), p_nonnol_derau=int(((p > 0) & (y == 0)).sum()))
    print(sd, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in OUT[sd].items()})
json.dump(OUT, open('results/final/diagnosis_int8_1ch.json', 'w'), indent=1)

# ---- diagnosis table for Appendix E (Indonesian: decimal comma)
dc = lambda v, nd=1: f'{v:.{nd}f}'.replace('.', ',')
rows = ''.join(f"{sd} & {int(round(o['thr_langkah']))} & {dc(100 * o['f1_python'])} & {dc(100 * o['f1_papan'])} & {o['keputusan_beda']} & {o['dekat_ambang']} & {dc(o['selisih_median'], 3)} & {dc(o['selisih_maks'], 2)} \\\\\n" for sd, o in OUT.items())
open('../jawaban_prof_arief_fase1/tabel_spektrum/tab_diagnosis_int8.tex', 'w').write(rows + '\\bottomrule\n')

# ---- macros for the text (numbers_diagnosis.tex)
H3S = json.load(open('results/final/hw_tiga_seed.json')); T = lambda f: ' / '.join(dc(f(o)) for o in OUT.values())
mac = {'dgEfPy': T(lambda o: 100 * o['f1_python']), 'dgEfPa': T(lambda o: 100 * o['f1_papan']), 'dgBedaWin': ' / '.join(str(o['keputusan_beda']) for o in OUT.values()),
       'dgSelisihMax': dc(max(100 * (o['f1_python'] - o['f1_papan']) for o in OUT.values())), 'dgDekatTiga': str(OUT['43']['dekat_ambang']),
       'sdLatMax': dc(max(H3S[k]['sd']['latency_ms'] for k in H3S)), 'sdEnPctMax': dc(max(100 * H3S[k]['sd']['energy_mj'] / H3S[k]['rerata']['energy_mj'] for k in H3S))}
open('../jawaban_prof_arief_fase1/numbers_diagnosis.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in mac.items())); print(mac)
