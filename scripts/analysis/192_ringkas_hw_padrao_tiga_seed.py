"""
Langkah 192 (draf artikel perangkat Q, tahap 4): ringkasan pengukuran ESP32-S3 untuk model KONVOLUSI STANDAR Float32 (satu kanal dan tiga kanal; seed 42, 43, 44)
dan perbandingannya dengan padanan depthwise separable (hw_tiga_seed.json, 175_): kernel TRANSPOSE_CONV pengganti, 100 jendela STEAD, protokol sama.
Sumber: firmware/tinyphasenet_profiling/results_hw/sesi_10nov_padrao_seed<S>_{1ch_fp32_sram,3ch_fp32_psram}_stead.log (satu kanal: arena SRAM internal; tiga kanal: arena PSRAM);
padanan: results/final/hw_tiga_seed.json ('1ch_fp32' = SRAM, '3ch_fp32' = PSRAM).
Keluaran: results/final/hw_padrao_tiga_seed.json (per seed, rerata, simpangan baku sampel, rasio standar/depthwise) dan ringkasan di layar.
Pemakaian (dari folder ini): python 192_ringkas_hw_padrao_tiga_seed.py
"""
import re, json, statistics as st
D = 'firmware/tinyphasenet_profiling/results_hw/'
LOG = {'1ch': 'sesi_10nov_padrao_seed{s}_1ch_fp32_sram_stead.log', '3ch': 'sesi_10nov_padrao_seed{s}_3ch_fp32_psram_stead.log'}


def hw(path):
    t = open(path, errors='ignore').read().replace('\r', ''); g = lambda p: float(re.search(p, t)[1])
    k = re.search(r'Kesesuaian Keputusan\s*\|\s*(\d+) / (\d+)', t)
    return dict(latency_ms=g(r'Latensi Invoke \(Median\)\s*\|\s*([\d.]+)'), arena_kb=g(r'Tensor Arena Digunakan\s*\|\s*([\d.]+)'), energy_mj=g(r'Energi per Inferensi\s*\|\s*([\d.]+)'),
                idle_mw=g(r'Daya Idle\s*\|\s*([\d.]+)'), active_mw=g(r'Daya Aktif \(saat Invoke\)\s*\|\s*([\d.]+)'), decisions_diff=int(k[2]) - int(k[1]), f1_100=g(r'F1-Score:\s*([\d.]+)'))


H = json.load(open('results/final/hw_tiga_seed.json')); out = {}
for fam in ('1ch', '3ch'):
    std = {s: hw(D + LOG[fam].format(s=s)) for s in ('42', '43', '44')}
    for key in ('latency_ms', 'arena_kb', 'energy_mj', 'idle_mw', 'active_mw', 'decisions_diff', 'f1_100'):
        xs = [std[s][key] for s in ('42', '43', '44')]; std.setdefault('rerata', {})[key] = st.mean(xs); std.setdefault('sd', {})[key] = st.stdev(xs)
    dw = H[f'{fam}_fp32']
    std['depthwise_rerata'] = dw['rerata']; std['depthwise_sd'] = dw['sd']
    std['razao_standar_depthwise'] = {k: std['rerata'][k] / dw['rerata'][k] for k in ('latency_ms', 'energy_mj', 'arena_kb')}
    out[fam] = std
    r, d, q = std['rerata'], dw['rerata'], std['razao_standar_depthwise']
    print(f"{fam}: standar {r['latency_ms']:.2f} +- {std['sd']['latency_ms']:.2f} ms, {r['energy_mj']:.2f} +- {std['sd']['energy_mj']:.2f} mJ, arena {r['arena_kb']:.2f} KB, on-board F1 {100 * r['f1_100']:.1f}%, differing decisions {r['decisions_diff']:.0f}/100")
    print(f"     depthwise {d['latency_ms']:.2f} ms, {d['energy_mj']:.2f} mJ, arena {d['arena_kb']:.2f} KB, on-board F1 {100 * d['f1_100']:.1f}% | rasio standar/depthwise: latensi {q['latency_ms']:.2f}x, energi {q['energy_mj']:.2f}x, arena {q['arena_kb']:.2f}x")
json.dump(out, open('results/final/hw_padrao_tiga_seed.json', 'w'), indent=1)
