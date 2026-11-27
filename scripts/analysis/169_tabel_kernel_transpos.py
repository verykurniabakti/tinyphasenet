"""
Langkah 169 (jawaban Prof. Arief, Lampiran E): hasil kernel TRANSPOSE_CONV jalur cepat di ESP32-S3, dibandingkan dengan kernel referensi.
Sumber (semua di firmware/tinyphasenet_profiling/results_hw/, sesi 6 Nov 2026):
  - profil per operator: sesi_06nov_kernel_transpos_{ref,cepat}_profil_<model>.log  (OPPROF, mikrodetik per operator, rerata 20 pemanggilan; OPPROF_HASH = hash keluaran model)
  - firmware penuh sebelum kernel baru: sesi_06nov_int8_final_stead.log, sesi_06nov_16x8_seed42_stead.log, sesi_06nov_3ch_int8_stead.log, sesi_06nov_3ch_16x8_stead.log
  - firmware penuh dengan kernel baru: sesi_06nov_kernel_baru_<model>_stead.log
Keluaran: ../jawaban_prof_arief_fase1/tabel_spektrum/tab_kernel_transpos_op.tex, tab_kernel_transpos_hw.tex, numbers_kernel.tex; results/final/kernel_transpos_esp32.json
Pemakaian (dari folder ini): python 169_tabel_kernel_transpos.py
"""
import re, json
D = 'firmware/tinyphasenet_profiling/results_hw/'; OUT = '../jawaban_prof_arief_fase1/'
V = ['1ch_fp32', '1ch_16x8', '1ch_int8', '3ch_fp32', '3ch_16x8', '3ch_int8']
NM = {'1ch_int8': 'Satu kanal INT8', '1ch_16x8': 'Satu kanal 16x8', '1ch_fp32': 'Satu kanal Float32', '3ch_int8': 'Tiga kanal INT8', '3ch_16x8': 'Tiga kanal 16x8', '3ch_fp32': 'Tiga kanal Float32'}
LAMA = {'1ch_int8': 'sesi_06nov_int8_final_stead.log', '1ch_16x8': 'sesi_06nov_16x8_seed42_stead.log', '3ch_int8': 'sesi_06nov_3ch_int8_stead.log', '3ch_16x8': 'sesi_06nov_3ch_16x8_stead.log',
        '1ch_fp32': 'sesi_06nov_float32_1ch_stead.log', '3ch_fp32': 'sesi_06nov_float32_3ch_stead.log'}
BARU = {v: (f'sesi_07nov_float32_fastpath_{v[:3]}_stead.log' if v.endswith('fp32') else f'sesi_06nov_kernel_baru_{v}_stead.log') for v in V}
PREF = {v: (f'sesi_07nov_float32_profil_ref_{v[:3]}.log' if v.endswith('fp32') else f'sesi_06nov_kernel_transpos_ref_profil_{v}.log') for v in V}
PCEP = {v: (f'sesi_07nov_float32_profil_fastpath_{v[:3]}.log' if v.endswith('fp32') else f'sesi_06nov_kernel_transpos_cepat_profil_{v}.log') for v in V}


def prof(path):
    t = open(path, errors='ignore').read().replace('\r', '')
    inv = float(re.search(r'invoke_ms_mean=([\d.]+)', t)[1]); h = re.search(r'OPPROF_HASH (\S+)', t)[1]
    ops = [(m[2], float(m[3]) / 1000) for m in re.finditer(r'^OPPROF (\d+) (\S+) ([\d.]+)', t, re.M)]
    tc = [x for n, x in ops if n == 'TRANSPOSE_CONV']
    return dict(invoke_ms=inv, hash=h, tc_layers=tc, tc_ms=sum(tc), total_ms=sum(x for _, x in ops))


def hw(path):
    t = open(path, errors='ignore').read().replace('\r', ''); g = lambda p: re.search(p, t)
    if g(r'Latensi Invoke \(Median\)'):   # format tiga kanal
        lat = float(g(r'Latensi Invoke \(Median\)\s*\|\s*([\d.]+)')[1]); ar = float(g(r'Tensor Arena Digunakan\s*\|\s*([\d.]+)')[1]); en = float(g(r'Energi per Inferensi\s*\|\s*([\d.]+)')[1])
        k = g(r'Kesesuaian Keputusan\s*\|\s*(\d+) / (\d+)'); beda, n = int(k[2]) - int(k[1]), int(k[2])
        f1 = float(g(r'F1-Score:\s*([\d.]+)')[1])
    else:                                  # format satu kanal
        lat = float(g(r'Latensi Invoke \(ms\)\s*\|\s*([\d.]+)')[1]); ar = float(g(r'Tensor arena terpakai \(KB\)\s*\|\s*([\d.]+)')[1]); en = float(g(r'Energi per inferensi \(mJ\)\s*\|\s*([\d.]+)')[1])
        k = g(r'Keputusan berbeda\s*:\s*(\d+) / (\d+)'); beda, n = int(k[1]), int(k[2])
        f1 = float(g(r'F1\s+([\d.]+)\s+FPR')[1])
    return dict(latency_ms=lat, arena_kb=ar, energy_mj=en, decisions_diff=beda, decisions_n=n, f1_100=f1)


R = {v: dict(ref=prof(D + PREF[v]), cepat=prof(D + PCEP[v]), lama=hw(D + LAMA[v]), baru=hw(D + BARU[v])) for v in V}
for v in V: assert R[v]['ref']['hash'] == R[v]['cepat']['hash'], f'hash beda pada {v}'   # kebenaran: keluaran identik (bilangan bulat dan float bit demi bit)
H3S = json.load(open('results/final/hw_tiga_seed.json')); FP = {k: {**R[f'{k}_fp32']['baru'], **{x: H3S[f'{k}_fp32']['rerata'][x] for x in ('latency_ms', 'energy_mj', 'arena_kb')}} for k in ('1ch', '3ch')}   # Float32 sesudah: rerata tiga seed (175_; satu kanal di SRAM, tiga kanal di PSRAM); pasangan sebelum/sesudah kernel tetap seed 42 dan PSRAM   # Float32 dengan fast-path kernel (170_ untuk firmware; 07 Nov); nilai kernel bawaan: R['<fam>_fp32']['lama']
R['float32'] = FP
json.dump(R, open('results/final/kernel_transpos_esp32.json', 'w'), indent=1)

f1 = lambda x: f'{x:,.1f}'.replace(',', '§').replace('.', ',').replace('§', '.')
f2 = lambda x: f'{x:.2f}'.replace('.', ',')
x_ = lambda a, b: f'{a / b:.1f}'.replace('.', ',') + '$\\times$'
op = []
for v in V:
    r, c = R[v]['ref'], R[v]['cepat']
    op.append(f"{NM[v]} & {f1(r['tc_ms'])} & {f1(c['tc_ms'])} & {x_(r['tc_ms'], c['tc_ms'])} & {f1(r['invoke_ms'])} & {f1(c['invoke_ms'])} & {x_(r['invoke_ms'], c['invoke_ms'])} & Ya \\\\")
open(OUT + 'tabel_spektrum/tab_kernel_transpos_op.tex', 'w').write('\n'.join(op) + '\n\\bottomrule\n')
RL = [R[v]['ref']['invoke_ms'] / R[v]['cepat']['invoke_ms'] for v in V]; RI = [R[v]['ref']['tc_ms'] / R[v]['cepat']['tc_ms'] for v in V]
M = {'kcLatRatMin': min(RL), 'kcLatRatMax': max(RL), 'kcTcRatMinInt': min(RI[i] for i, v in enumerate(V) if not v.endswith('fp32')), 'kcTcRatMaxInt': max(RI[i] for i, v in enumerate(V) if not v.endswith('fp32')), 'kcTcRatFpSat': RI[V.index('1ch_fp32')], 'kcTcRatFpTig': RI[V.index('3ch_fp32')],
     'kcTcPctFpSat': 100 * R['1ch_fp32']['ref']['tc_ms'] / R['1ch_fp32']['ref']['total_ms'], 'kcTcPctFpTig': 100 * R['3ch_fp32']['ref']['tc_ms'] / R['3ch_fp32']['ref']['total_ms'],
     'kcLatFpSatLama': R['1ch_fp32']['lama']['latency_ms'], 'kcLatFpTigLama': R['3ch_fp32']['lama']['latency_ms'], 'kcEnFpSatLama': R['1ch_fp32']['lama']['energy_mj'], 'kcEnFpTigLama': R['3ch_fp32']['lama']['energy_mj'],
     'kcTcRatMin': min(RI), 'kcTcRatMax': max(RI),
     'kcLatQqBaru': R['1ch_16x8']['baru']['latency_ms'], 'kcLatIntBaru': R['1ch_int8']['baru']['latency_ms'], 'kcLatIntLama': R['1ch_int8']['lama']['latency_ms'],
     'kcArIntLama': R['1ch_int8']['lama']['arena_kb'], 'kcArIntBaru': R['1ch_int8']['baru']['arena_kb'], 'kcArQqLama': R['1ch_16x8']['lama']['arena_kb'], 'kcArQqBaru': R['1ch_16x8']['baru']['arena_kb'],
     'kcLatFpSatPs': R['1ch_fp32']['baru']['latency_ms'], 'kcEnFpSatPs': R['1ch_fp32']['baru']['energy_mj'], 'kcLatFpSat': FP['1ch']['latency_ms'], 'kcLatFpTig': FP['3ch']['latency_ms'], 'kcEnFpSat': FP['1ch']['energy_mj'], 'kcEnFpTig': FP['3ch']['energy_mj'], 'kcArFpSat': FP['1ch']['arena_kb'], 'kcArFpTig': FP['3ch']['arena_kb'], 'kcDutyFpSat': FP['1ch']['latency_ms'] / 30720 * 100, 'kcDutyFpTig': FP['3ch']['latency_ms'] / 30720 * 100,
     'kcArTigQqLama': R['3ch_16x8']['lama']['arena_kb'], 'kcArTigQqBaru': R['3ch_16x8']['baru']['arena_kb']}
open(OUT + 'numbers_kernel.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{f1(v)}}}\n' for k, v in M.items()))
print('ok', {k: round(v, 2) for k, v in M.items()})
