"""
Step 231 (companion and device summaries): summary of the ESP32-S3 measurements of the quantized STANDARD-CONVOLUTION (PhaseNet) models in 16x8 and INT8, one channel and three channels; seeds 42, 43, 44.
Completes 192_ (standard Float32). Firmware from 230_make_fw_standar.py; the same protocol as TinyPhaseNet (optimized TRANSPOSE_CONV kernel, 100 STEAD windows, INA219); one channel: internal SRAM arena; three channels: PSRAM arena.
Sources: firmware/tinyphasenet_profiling/results_hw/sesi_27nov_padrao_seed<S>_{1ch,3ch}_{16x8,int8}_stead.log; separable counterparts: results/final/hw_tiga_seed.json.
Output: results/final/hw_padrao_kuant_tiga_seed.json (per seed, mean, sample standard deviation, standard/separable ratio) and a summary on screen. Usage (from this folder): python 231_ringkas_hw_padrao_kuantisasi.py
"""
import re, json, statistics as st
D = 'firmware/tinyphasenet_profiling/results_hw/'; SEEDS = ('42', '43', '44')


def hw(path):
    t = open(path, errors='ignore').read().replace('\r', ''); g = lambda p: re.search(p, t)
    if g(r'Latensi Invoke \(Median\)'):
        lat = float(g(r'Latensi Invoke \(Median\)\s*\|\s*([\d.]+)')[1]); ar = float(g(r'Tensor Arena Digunakan\s*\|\s*([\d.]+)')[1]); en = float(g(r'Energi per Inferensi\s*\|\s*([\d.]+)')[1])
        k = g(r'Kesesuaian Keputusan\s*\|\s*(\d+) / (\d+)'); beda, n = int(k[2]) - int(k[1]), int(k[2]); f1 = float(g(r'F1-Score:\s*([\d.]+)')[1])
    else:
        lat = float(g(r'Latensi Invoke \(ms\)\s*\|\s*([\d.]+)')[1]); ar = float(g(r'Tensor arena terpakai \(KB\)\s*\|\s*([\d.]+)')[1]); en = float(g(r'Energi per inferensi \(mJ\)\s*\|\s*([\d.]+)')[1])
        k = g(r'Keputusan berbeda\s*:\s*(\d+) / (\d+)'); beda, n = int(k[1]), int(k[2]); f1 = float(g(r'F1\s+([\d.]+)\s+FPR')[1])
    return dict(latency_ms=lat, arena_kb=ar, energy_mj=en, decisions_diff=beda, decisions_n=n, f1_100=f1)


H = json.load(open('results/final/hw_tiga_seed.json')); out = {}
for v in ('1ch_16x8', '1ch_int8', '3ch_16x8', '3ch_int8'):
    R = {s: hw(D + f'sesi_27nov_padrao_seed{s}_{v}_stead.log') for s in SEEDS}
    for key in ('latency_ms', 'energy_mj', 'arena_kb', 'f1_100', 'decisions_diff'):
        xs = [R[s][key] for s in SEEDS]; R.setdefault('rerata', {})[key] = st.mean(xs); R.setdefault('sd', {})[key] = st.stdev(xs)
    R['razao_standar_separable'] = {k: R['rerata'][k] / H[v]['rerata'][k] for k in ('latency_ms', 'energy_mj', 'arena_kb')}
    out[v] = R; r, q = R['rerata'], R['razao_standar_separable']
    print(f"{v}: standar " + ' / '.join(f"{R[s]['latency_ms']:.1f}" for s in SEEDS) + f" ms (rerata {r['latency_ms']:.1f}±{R['sd']['latency_ms']:.1f}); energi {r['energi_mj'] if False else r['energy_mj']:.1f}±{R['sd']['energy_mj']:.1f} mJ; arena {r['arena_kb']:.1f} KB; "
          f"F1 papan {100 * r['f1_100']:.1f}±{100 * R['sd']['f1_100']:.1f}%; differing decisions {r['decisions_diff']:.0f}; rasio vs separable: latensi {q['latency_ms']:.2f}x, energi {q['energy_mj']:.2f}x, arena {q['arena_kb']:.2f}x")
json.dump(out, open('results/final/hw_padrao_kuant_tiga_seed.json', 'w'), indent=1)

# ---- device table (ringkasan_hasil_pendamping_perangkat/tabel/tab_perangkat.tex): twelve rows (Indonesian format)
PD = json.load(open('results/final/hw_padrao_tiga_seed.json'))
idn = lambda x, nd: f'{x:,.{nd}f}'.replace(',', '§').replace('.', ',').replace('§', '.')
SRC = [('TinyPhaseNet satu kanal Float32', H['1ch_fp32'], 'SRAM'), ('TinyPhaseNet satu kanal 16x8', H['1ch_16x8'], 'SRAM'), ('TinyPhaseNet satu kanal INT8', H['1ch_int8'], 'SRAM'),
       ('TinyPhaseNet tiga kanal Float32', H['3ch_fp32'], 'PSRAM'), ('TinyPhaseNet tiga kanal 16x8', H['3ch_16x8'], 'PSRAM'), ('TinyPhaseNet tiga kanal INT8', H['3ch_int8'], 'PSRAM'),
       ('PhaseNet satu kanal Float32', PD['1ch'], 'SRAM'), ('PhaseNet satu kanal 16x8', out['1ch_16x8'], 'SRAM'), ('PhaseNet satu kanal INT8', out['1ch_int8'], 'SRAM'),
       ('PhaseNet tiga kanal Float32', PD['3ch'], 'PSRAM'), ('PhaseNet tiga kanal 16x8', out['3ch_16x8'], 'PSRAM'), ('PhaseNet tiga kanal INT8', out['3ch_int8'], 'PSRAM')]
rows = []
for nm, d, loc in SRC:
    r, s = d['rerata'], d['sd']
    rows.append(f"{nm} & {idn(r['arena_kb'], 1)} ({loc}) & {idn(r['latency_ms'], 1)} $\\pm$ {idn(s['latency_ms'], 1)} & {idn(r['energy_mj'], 1)} $\\pm$ {idn(s['energy_mj'], 1)} & {idn(100 * r['latency_ms'] / 30720, 1)} & {idn(100 * r['f1_100'], 1)} $\\pm$ {idn(100 * s['f1_100'], 1)} \\\\")
import sys
old = [l for l in open('../ringkasan_hasil_pendamping_perangkat/tabel/tab_perangkat.tex').read().split('\n') if '&' in l]
new8 = rows[:6] + rows[6:7] + rows[9:10]
for o, n in zip(old, new8): print('SAME' if o.strip() == n.strip() else f'BEDA\n  lama: {o}\n  baru: {n}')
open('results/final/tab_perangkat_dua_belas.tex', 'w').write('\n'.join(rows[:3]) + '\n\\midrule\n' + '\n'.join(rows[3:6]) + '\n\\midrule\n' + '\n'.join(rows[6:9]) + '\n\\midrule\n' + '\n'.join(rows[9:12]) + '\n\\bottomrule\n')
