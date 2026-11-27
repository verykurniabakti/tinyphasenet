"""
Langkah 189 (data suplemen draf artikel perangkat Q): hasil pengukuran ESP32-S3 per seed (42, 43, 44) dalam CSV.
Sumber: results/final/hw_tiga_seed.json (175_), tanpa angka ketikan tangan. Menggantikan tabel per seed yang tadinya Lampiran naskah.
Kolom: configuration, seed, latency_ms, arena_kb, energy_mj, f1_device_percent, differing_decisions, decisions_compared; baris seed 'mean' dan 'sd' memakai rerata dan simpangan baku sampel tiga seed.
Pemakaian (dari folder ini): python 189_suplemen_per_seed.py [folder_keluaran]   (bawaan: ../supplement/)
Keluaran: <folder_keluaran>/per_seed_results.csv
"""
import csv, json, os, sys


def r3(v):
    return round(v, 3) if isinstance(v, float) else v


OUT = sys.argv[1] if len(sys.argv) > 1 else '../supplement/'
os.makedirs(OUT, exist_ok=True)
H = json.load(open('results/final/hw_tiga_seed.json'))
NAME = {'1ch_fp32': '1-ch Float32', '1ch_16x8': '1-ch 16x8', '1ch_int8': '1-ch INT8', '3ch_fp32': '3-ch Float32', '3ch_16x8': '3-ch 16x8', '3ch_int8': '3-ch INT8',
        '1ch_fp32_psram': '1-ch Float32 (arena in PSRAM)'}
with open(os.path.join(OUT, 'per_seed_results.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['configuration', 'seed', 'latency_ms', 'arena_kb', 'energy_mj', 'f1_device_percent', 'differing_decisions', 'decisions_compared'])
    for k, lab in NAME.items():
        for s in ('42', '43', '44'):
            r = H[k][s]
            w.writerow([lab, s, r['latency_ms'], r['arena_kb'], r['energy_mj'], round(100 * r['f1_100'], 2), r['decisions_diff'], r['decisions_n']])
        for s in ('rerata', 'sd'):
            r = H[k][s]
            w.writerow([lab, 'mean' if s == 'rerata' else 'sd', r3(r.get('latency_ms', '')), r3(r.get('arena_kb', '')), r3(r.get('energy_mj', '')),
                        round(100 * r['f1_100'], 2) if 'f1_100' in r else '', r3(r.get('decisions_diff', '')), r.get('decisions_n', '')])
print('ok', OUT)
