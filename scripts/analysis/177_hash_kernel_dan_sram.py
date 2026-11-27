"""
Langkah 177 (jawaban Prof. Arief, Lampiran E): (a) kebenaran kernel TRANSPOSE_CONV optimasi pada 100 jendela STEAD nyata (bukan satu masukan acak), (b) Float32 dengan arena di SRAM internal.
Sumber: firmware/tinyphasenet_profiling/results_hw/sesi_07nov_hash100_<model>_kernel_{optimasi,referensi,bawaan}.log (baris 'HASH <jendela> <FNV-1a 32-bit seluruh tensor keluaran>'; seed 42, 100 jendela uji STEAD, 50 gempa dan 50 derau)
        dan sesi_07nov_float32_{1ch_sram_internal,3ch_sram_internal_gagal}.log.
Kernel referensi satu kanal 16x8 dijalankan dengan arena di PSRAM (scratch int64 tidak muat di SRAM); lokasi memori tidak memengaruhi nilai keluaran.
Keluaran: results/final/hash_kernel_100jendela.json, ../jawaban_prof_arief_fase1/numbers_hash.tex. Pemakaian (dari folder ini): python 177_hash_kernel_dan_sram.py
"""
import re, json
D = 'firmware/tinyphasenet_profiling/results_hw/'
def H(f): return {int(m[1]): m[2] for m in re.finditer(r'^HASH (\d+) ([0-9a-f]{8})', open(D + f, errors='ignore').read().replace('\r', ''), re.M)}
P = [('1ch_int8', 'optimasi', 'referensi'), ('1ch_16x8', 'optimasi', 'referensi'), ('3ch_int8', 'optimasi', 'referensi'), ('3ch_16x8', 'optimasi', 'referensi'), ('1ch_fp32', 'optimasi', 'bawaan'), ('3ch_fp32', 'optimasi', 'bawaan')]
R = {}
for m, a, b in P:
    x, y = H(f'sesi_07nov_hash100_{m}_kernel_{a}.log'), H(f'sesi_07nov_hash100_{m}_kernel_{b}.log')
    assert len(x) == 100 and len(y) == 100, (m, len(x), len(y))
    R[m] = dict(jendela=100, sama=sum(1 for w in x if y[w] == x[w]), unik=len(set(x.values())))
    assert R[m]['sama'] == 100, (m, R[m])   # kebenaran: seluruh hash identik
t = open(D + 'sesi_07nov_float32_1ch_sram_internal.log', errors='ignore').read().replace('\r', ''); g = lambda p: re.search(p, t)
S1 = dict(lokasi=g(r'Lokasi Tensor Arena\s*\|\s*(.+)')[1].strip(), arena_kb=float(g(r'Tensor Arena Digunakan\s*\|\s*([\d.]+)')[1]), lat_ms=float(g(r'Latensi Invoke \(Median\)\s*\|\s*([\d.]+)')[1]), energi_mj=float(g(r'Energi per Inferensi\s*\|\s*([\d.]+)')[1]),
          keputusan=g(r'Kesesuaian Keputusan\s*\|\s*(\d+) / (\d+)')[0], f1=float(g(r'F1-Score:\s*([\d.]+)')[1]))
t3 = open(D + 'sesi_07nov_float32_3ch_sram_internal_gagal.log', errors='ignore').read().replace('\r', '')
rq = re.search(r'Requested: (\d+), available (\d+)', t3); S3 = dict(diminta_kb=int(rq[1]) / 1024, tersedia_kb=int(rq[2]) / 1024, gagal='AllocateTensors() GAGAL' in t3)
HJ = json.load(open('results/final/hw_tiga_seed.json'))
def sram3(f):
    t = open(D + f, errors='ignore').read().replace('\r', ''); g = lambda p: re.search(p, t)
    return dict(lokasi=g(r'Lokasi Tensor Arena\s*\|\s*(.+)')[1].strip(), lat_ms=float(g(r'Latensi Invoke \(Median\)\s*\|\s*([\d.]+)')[1]), energi_mj=float(g(r'Energi per Inferensi\s*\|\s*([\d.]+)')[1]), keputusan=g(r'Kesesuaian Keputusan\s*\|\s*(\d+) / (\d+)')[0])
S3Q, S3I = sram3('sesi_07nov_3ch_16x8_sram_internal.log'), sram3('sesi_07nov_3ch_int8_sram_internal.log')
assert S3Q['lokasi'] == 'Internal SRAM' and S3I['lokasi'] == 'Internal SRAM'
PS = json.load(open('results/final/hw_tiga_seed.json'))['1ch_fp32_psram']['42']   # PSRAM, seed 42 (angka utama satu kanal Float32 kini SRAM; 175_)
out = dict(hash=R, sram_1ch=S1, sram_3ch_fp32=S3, sram_3ch_16x8=S3Q, sram_3ch_int8=S3I, psram_1ch_seed42=PS)
json.dump(out, open('results/final/hash_kernel_100jendela.json', 'w'), indent=1)
dc = lambda v, nd=1: f'{v:.{nd}f}'.replace('.', ',')
mac = {'hsJendela': '100', 'hsModel': str(len(P)), 'hsTotal': str(sum(r['sama'] for r in R.values())), 'hsUnikIntTig': str(R['3ch_int8']['unik']), 'hsUnikIntSat': str(R['1ch_int8']['unik']),
       'srLatSat': dc(S1['lat_ms']), 'srEnSat': dc(S1['energi_mj']), 'srArSat': dc(S1['arena_kb']), 'srLatPs': dc(PS['latency_ms']), 'srEnPs': dc(PS['energy_mj']),
       'srPctLat': dc(100 * (1 - S1['lat_ms'] / PS['latency_ms'])), 'srLatRr': dc(HJ['1ch_fp32']['rerata']['latency_ms']), 'srEnRr': dc(HJ['1ch_fp32']['rerata']['energy_mj']), 'srPctEn': dc(100 * (1 - S1['energi_mj'] / PS['energy_mj'])), 'srLatTigQq': dc(S3Q['lat_ms']), 'srEnTigQq': dc(S3Q['energi_mj']), 'srLatTigInt': dc(S3I['lat_ms']), 'srEnTigInt': dc(S3I['energi_mj']),
       'srPctLatTigQq': dc(100 * (1 - S3Q['lat_ms'] / HJ['3ch_16x8']['42']['latency_ms'])), 'srPctEnTigQq': dc(100 * (1 - S3Q['energi_mj'] / HJ['3ch_16x8']['42']['energy_mj'])),
       'srPctLatTigInt': dc(100 * (1 - S3I['lat_ms'] / HJ['3ch_int8']['42']['latency_ms'])), 'srPctEnTigInt': dc(100 * (1 - S3I['energi_mj'] / HJ['3ch_int8']['42']['energy_mj'])),
       'srDimintaTig': dc(S3['diminta_kb'], 0), 'srTersediaTig': dc(S3['tersedia_kb'], 0)}
open('../jawaban_prof_arief_fase1/numbers_hash.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in mac.items())); print(json.dumps(out, indent=1)); print(mac)
