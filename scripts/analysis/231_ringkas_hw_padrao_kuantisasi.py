"""
Langkah 231 (ringkasan hasil pendamping dan perangkat): ringkasan pengukuran ESP32-S3 model KONVOLUSI STANDAR (PhaseNet) terkuantisasi 16x8 dan INT8, satu kanal dan tiga kanal; seed 42, 43, 44.
Melengkapi 192_ (standar Float32). Firmware dari 230_make_fw_standar.py; protokol sama dengan TinyPhaseNet (kernel TRANSPOSE_CONV optimasi, 100 jendela STEAD, INA219); satu kanal: arena SRAM internal; tiga kanal: arena PSRAM.
Sumber: firmware/tinyphasenet_profiling/results_hw/sesi_27nov_padrao_seed<S>_{1ch,3ch}_{16x8,int8}_stead.log; padanan separable: results/final/hw_tiga_seed.json.
Keluaran: results/final/hw_padrao_kuant_tiga_seed.json (per seed, rerata, simpangan baku sampel, rasio standar/separable) dan ringkasan di layar. Pemakaian (dari folder ini): python 231_ringkas_hw_padrao_kuantisasi.py
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

