"""
Step 235 (companion compression manuscript): table that sets the simulation against the ESP32-S3 measurements (12 configurations) and number macros, English and Indonesian.
Recomputes as in 233_ (static count = MACs x instructions per MAC / 240 MHz; simulated energy = 264 mW x latency; analytical model only Float32), with the format names of the companion manuscript (8/16, 8/8).
Output: ../draft_publikasi_1/tabel_pend/tab_sim_ukur{,_id}.tex and numbers_sand{,_id}.tex. Usage (from this folder): python 235_sandingan_draf_pendamping.py
"""
import json
R = 'results/final/'; OUT = '../draft_publikasi_1/tabel_pend/'
S = json.load(open(R + 'simulasi_biaya_mcu.json')); X = json.load(open(R + 'xtensa_instruksi.json'))['hasil']; M = json.load(open(R + 'mac_flops.json'))
H = json.load(open(R + 'hw_tiga_seed.json')); PD = json.load(open(R + 'hw_padrao_tiga_seed.json')); PQ = json.load(open(R + 'hw_padrao_kuant_tiga_seed.json'))
MACT = {f: sum(S['beban'][f][k] for k in ('pw', 'dw', 'tc')) for f in ('1ch', '3ch')}; MACS = {f: M[f + '_standar']['macs_tflite'] for f in ('1ch', '3ch')}
AN_T = {x['model']: x['latensi_ms'] for x in S['simulasi_asumsi'] if x['mcu'] == 'ESP32-S3'}; AN_S = {x['model']: x['latensi_ms'] for x in S['simulasi_asumsi_standar'] if x['mcu'] == 'ESP32-S3'}
EN_T = {x['model']: x['energi_mj'] for x in S['simulasi_asumsi'] if x['mcu'] == 'ESP32-S3'}; EN_S = {x['model']: x['energi_mj'] for x in S['simulasi_asumsi_standar'] if x['mcu'] == 'ESP32-S3'}
P_ASUMSI = 264.0; NM = {'fp32': 'Float32', '16x8': '8/16', 'int8': '8/8'}
LANG = {'en': dict(dec='.', thou=',', ch={'1ch': 'single-channel', '3ch': 'three-channel'}), 'id': dict(dec=',', thou='.', ch={'1ch': 'satu kanal', '3ch': 'tiga kanal'})}
num = lambda v, nd, L: f'{v:,.{nd}f}'.replace(',', '\u0001').replace('.', L['dec']).replace('\u0001', L['thou'])
rec = []   # (key name, fam, fmt, conv, meas, stat, an, emeas, estat, ean)
for conv, nmc in (('t', 'TinyPhaseNet'), ('s', 'PhaseNet')):
    for fam in ('1ch', '3ch'):
        for f in ('fp32', '16x8', 'int8'):
            if conv == 't': hh = H[f'{fam}_{f}']['rerata']; mac = MACT[fam]; an = AN_T[fam] if f == 'fp32' else None; ean = EN_T[fam] if f == 'fp32' else None
            else: hh = (PD[fam] if f == 'fp32' else PQ[f'{fam}_{f}'])['rerata']; mac = MACS[fam]; an = AN_S[fam] if f == 'fp32' else None; ean = EN_S[fam] if f == 'fp32' else None
            stat = mac * X[f]['instruksi_per_mac'] / 240e3
            rec.append(dict(nmc=nmc, fam=fam, f=f, meas=hh['latency_ms'], stat=stat, an=an, emeas=hh['energy_mj'], estat=P_ASUMSI * stat / 1000, ean=ean, p=1000 * hh['energy_mj'] / hh['latency_ms']))
for lg, L in LANG.items():
    sfx = '' if lg == 'en' else '_id'; n = lambda v, nd=1: num(v, nd, L); groups = []
    for k in range(0, 12, 3):
        g = []
        for r in rec[k:k + 3]:
            pc = lambda a, b: n(100 * a / b, 0)
            g.append(f"{r['nmc']} {L['ch'][r['fam']]} {NM[r['f']]} & {'SRAM' if r['fam'] == '1ch' else 'PSRAM'} & {n(r['meas'])} & {pc(r['stat'], r['meas'])} & " + (pc(r['an'], r['meas']) if r['an'] else '--')
                     + f" & {n(r['emeas'])} & {pc(r['estat'], r['emeas'])} & " + (pc(r['ean'], r['emeas']) if r['ean'] else '--') + ' \\\\')
        groups.append('\n'.join(g))
    open(OUT + f'tab_sim_ukur{sfx}.tex', 'w').write('\n\\midrule\n'.join(groups) + '\n\\bottomrule\n')
    pc = lambda a, b: 100 * a / b; fp = [r for r in rec if r['an']]
    Mc = {'dvSandStatLatMin': n(min(pc(r['stat'], r['meas']) for r in rec), 0), 'dvSandStatLatMax': n(max(pc(r['stat'], r['meas']) for r in rec), 0),
          'dvSandAnLatMin': n(min(pc(r['an'], r['meas']) for r in fp), 0), 'dvSandAnLatMax': n(max(pc(r['an'], r['meas']) for r in fp), 0),
          'dvSandStatEnMin': n(min(pc(r['estat'], r['emeas']) for r in rec), 0), 'dvSandStatEnMax': n(max(pc(r['estat'], r['emeas']) for r in rec), 0),
          'dvSandAnEnMin': n(min(pc(r['ean'], r['emeas']) for r in fp), 0), 'dvSandAnEnMax': n(max(pc(r['ean'], r['emeas']) for r in fp), 0),
          'dvSandPwMin': n(min(r['p'] for r in rec), 0), 'dvSandPwMax': n(max(r['p'] for r in rec), 0), 'dvSandPwAssumed': n(P_ASUMSI, 0)}
    open(OUT + f'numbers_sand{sfx}.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in Mc.items()))
print({k: v for k, v in Mc.items()})
