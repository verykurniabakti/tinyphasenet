"""
Step 178 (follow-up manuscript with the device, draft_publikasi_1/draft_perangkat_q.tex and _id.tex): tables, number macros, and two figures, English and Indonesian.
All numbers are read from stored results (no hand-typed numbers):
  results/final/pareto_kandidat.json      F1 (STEAD, mean of four datasets), file size, activation-memory lower bound, Pareto result (164_)
  results/final/hw_tiga_seed.json         ESP32-S3: latency, energy, arena, on-board F1 per seed and mean (175_); one-channel Float32 = SRAM arena
  results/final/kernel_transpos_esp32.json  reference and optimized kernel, seed 42, PSRAM arena for Float32 (169_)
  results/final/hash_kernel_100jendela.json 100-window hash test and SRAM arena result (177_)
  results/final/diagnosis_int8_1ch.json   one-channel INT8 diagnosis (176_)
Output: ../draft_publikasi_1/tabel_dev/{tab_*.tex, numbers_dev.tex} and the *_id.tex versions; fig_dev_kernel{,_id}.pdf, fig_dev_pareto{,_id}.pdf.
Usage (from this folder): python 178_tabel_draft_perangkat.py
"""
import json, os, statistics as st
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = 'results/final/'; OUT = '../draft_publikasi_1/tabel_dev/'; os.makedirs(OUT, exist_ok=True)
P = json.load(open(R + 'pareto_kandidat.json')); H = json.load(open(R + 'hw_tiga_seed.json')); K = json.load(open(R + 'kernel_transpos_esp32.json'))
PD = json.load(open(R + 'hw_padrao_tiga_seed.json'))   # 192_: standard-convolution Float32 models (stage 4)
X = json.load(open(R + 'hash_kernel_100jendela.json')); D = json.load(open(R + 'diagnosis_int8_1ch.json'))
V = ['1ch_fp32', '1ch_16x8', '1ch_int8', '3ch_fp32', '3ch_16x8', '3ch_int8']
TAG = {'1ch_fp32': 'OneFp', '1ch_16x8': 'OneQq', '1ch_int8': 'OneInt', '3ch_fp32': 'ThrFp', '3ch_16x8': 'ThrQq', '3ch_int8': 'ThrInt'}
FMT = {'fp32': 'Float32', 'q16x8': '16x8', 'int8': 'INT8', 'w4a8': '4/8'}
cand = {(r['fam'], r['key']): r for r in P['kandidat']}
cv = lambda v: cand[(v[:3], {'fp32': 'fp32', '16x8': 'q16x8', 'int8': 'int8'}[v[4:]])]
LANG = {'en': dict(name={'1ch': 'Single-channel', '3ch': 'Three-channel'}, std='standard', dec='.', thou=',', one='Single-channel', thr='Three-channel', yes='yes', no='no', na='n/a'),
        'id': dict(name={'1ch': 'Satu kanal', '3ch': 'Tiga kanal'}, std='standar', dec=',', thou='.', one='Satu kanal', thr='Tiga kanal', yes='ya', no='tidak', na='n/a')}


def num(v, nd, L):
    s = f'{v:,.{nd}f}'
    return s.replace(',', '\u0001').replace('.', L['dec']).replace('\u0001', L['thou'])


def nd(items, spec):
    keep = []
    for i, a in enumerate(items):
        dom = any(all(s * (b[k] - a[k]) >= -1e-12 for k, s in spec) and any(s * (b[k] - a[k]) > 1e-12 for k, s in spec) for j, b in enumerate(items) if j != i)
        if not dom: keep.append(i)
    return keep


_A = json.load(open(R + 'jawaban_arief_tiga_tahap.json'))['seeds']['42']['separable']; _B = json.load(open(R + 'jawaban_arief_phasenet.json'))['seeds']['42']['pndw']
MACS = {'1ch': _A['macs'], '3ch': _B['macs']}; PARS = {'1ch': _A['params'], '3ch': _B['params']}
MEAS = [dict(v=v, name=v, f1=cv(v)['f1_mean4'], lat=H[v]['rerata']['latency_ms'], en=H[v]['rerata']['energy_mj']) for v in V]
NDOM = [MEAS[i]['v'] for i in nd(MEAS, [('f1', +1), ('en', -1), ('lat', -1)])]
P8 = json.load(open(R + 'pareto_delapan.json')); NDOM8 = P8['batas_8_rerata']   # 193_: eight configurations (six depthwise + two standard Float32 models)
MEAS8 = MEAS + [dict(v=f'{f}_std', name=f'{f}_std', f1=P8['rerata'][f'{f}_std']['f1'], lat=P8['rerata'][f'{f}_std']['lat'], en=P8['rerata'][f'{f}_std']['en']) for f in ('1ch', '3ch')]
mac = {}
for lg, L in LANG.items():
    n = lambda v, d=1: num(v, d, L); sfx = '' if lg == 'en' else '_id'
    nm = lambda v: f"{L['name'][v[:3]]} {FMT[{'16x8': 'q16x8'}.get(v[4:], v[4:])]}"
    # --- table 1: format, size, quality
    rows = ''
    for fam in ('1ch', '3ch'):
        for k in ('fp32', 'q16x8', 'int8', 'w4a8'):
            r = cand[(fam, k)]; bits = {'fp32': '32 / 32', 'q16x8': '8 / 16', 'int8': '8 / 8', 'w4a8': '4 / 8'}[k]
            rows += f"{nm(fam + '_' + {'q16x8': '16x8'}.get(k, k)) if k != 'w4a8' else L['name'][fam] + ' 4/8'} & {bits} & {n(r['flash_kb'])} & {n(r['aktivasi_lb_kb'], 0)} & {n(r['f1_stead'])} & {n(r['f1_mean4'])} \\\\\n"
        rows += '\\midrule\n' if fam == '1ch' else ''
    open(OUT + f'tab_formats{sfx}.tex', 'w').write(rows + '\\bottomrule\n')
    # --- table 2: device measurements
    rows = ''
    for v in V:
        h = H[v]; r, s = h['rerata'], h['sd']; loc = 'SRAM' if v.startswith('1ch') else 'PSRAM'
        rows += (f"{nm(v)} & {n(r['arena_kb'])} ({loc}) & {n(r['latency_ms'])} $\\pm$ {n(s['latency_ms'])} & {n(r['energy_mj'])} $\\pm$ {n(s['energy_mj'])} & {n(100 * r['latency_ms'] / 30720)} & "
                 f"{n(100 * r['f1_100'])} $\\pm$ {n(100 * s['f1_100'])} \\\\\n")
        if v == '1ch_int8': rows += '\\midrule\n'
    rows += '\\midrule\n'   # standard-convolution models (Float32), the counterparts of the Float32 rows above
    for fam in ('1ch', '3ch'):
        r, s = PD[fam]['rerata'], PD[fam]['sd']; loc = 'SRAM' if fam == '1ch' else 'PSRAM'
        rows += (f"{L['name'][fam]} {L['std']} Float32 & {n(r['arena_kb'])} ({loc}) & {n(r['latency_ms'])} $\\pm$ {n(s['latency_ms'])} & {n(r['energy_mj'])} $\\pm$ {n(s['energy_mj'])} & {n(100 * r['latency_ms'] / 30720)} & "
                 f"{n(100 * r['f1_100'])} $\\pm$ {n(100 * s['f1_100'])} \\\\\n")
    open(OUT + f'tab_device{sfx}.tex', 'w').write(rows + '\\bottomrule\n')
    # --- table 3: kernel
    rows = ''; sp = {}
    for v in V:
        k = K[v]; rf, cp = k['ref'], k['cepat']; lm, br = k['lama'], k['baru']
        sp[v] = dict(tc=rf['tc_ms'] / cp['tc_ms'], tot=rf['invoke_ms'] / cp['invoke_ms'], full=lm['latency_ms'] / br['latency_ms'], en=lm['energy_mj'] / br['energy_mj'])
        rows += (f"{nm(v)} & {n(rf['tc_ms'])} & {n(cp['tc_ms'])} & {n(sp[v]['tc'])}$\\times$ & {n(lm['latency_ms'])} & {n(br['latency_ms'])} & {n(sp[v]['full'])}$\\times$ & "
                 f"{n(lm['arena_kb'], 0)} & {n(br['arena_kb'], 0)} \\\\\n")
        if v == '1ch_int8': rows += '\\midrule\n'
    open(OUT + f'tab_kernel{sfx}.tex', 'w').write(rows + '\\bottomrule\n')
    # --- table 4: Pareto (measured candidates)
    rows = ''
    for m in MEAS:
        r = cv(m['v']); rows += f"{nm(m['v'])} & {n(r['f1_mean4'])} & {n(r['flash_kb'])} & {n(m['lat'])} & {n(m['en'])} & {L['yes'] if m['v'] in NDOM else L['no']} \\\\\n"
        if m['v'] == '1ch_int8': rows += '\\midrule\n'
    open(OUT + f'tab_pareto{sfx}.tex', 'w').write(rows + '\\bottomrule\n')
    # --- table 5: arena location
    ps1 = H['1ch_fp32_psram']['42']; s1 = X['sram_1ch']
    sr = {'3ch_16x8': X['sram_3ch_16x8'], '3ch_int8': X['sram_3ch_int8']}
    rows = f"{nm('1ch_fp32')} & {n(ps1['latency_ms'])} & {n(ps1['energy_mj'])} & {n(s1['lat_ms'])} & {n(s1['energi_mj'])} & {n(100 * (1 - s1['lat_ms'] / ps1['latency_ms']))} & {n(100 * (1 - s1['energi_mj'] / ps1['energy_mj']))} \\\\\n"
    for v in ('3ch_16x8', '3ch_int8'):
        p, s = H[v]['42'], sr[v]; rows += f"{nm(v)} & {n(p['latency_ms'])} & {n(p['energy_mj'])} & {n(s['lat_ms'])} & {n(s['energi_mj'])} & {n(100 * (1 - s['lat_ms'] / p['latency_ms']))} & {n(100 * (1 - s['energi_mj'] / p['energy_mj']))} \\\\\n"
    open(OUT + f'tab_sram{sfx}.tex', 'w').write(rows + '\\bottomrule\n')
    # --- table 6: one-channel INT8 diagnosis
    rows = ''.join(f"{sd} & {int(round(o['thr_langkah']))} & {n(100 * o['f1_python'])} & {n(100 * o['f1_papan'])} & {o['keputusan_beda']} & {o['dekat_ambang']} & {n(o['selisih_maks'], 2)} \\\\\n" for sd, o in D.items())
    open(OUT + f'tab_diag{sfx}.tex', 'w').write(rows + '\\bottomrule\n')
    # --- table 7: per seed
    rows = ''
    for v in V:
        t = lambda key, d, sc=1: ' / '.join(num(H[v][s][key] * sc, d, L) for s in ('42', '43', '44'))
        rows += f"{nm(v)} & {t('latency_ms', 1)} & {t('energy_mj', 1)} & {t('f1_100', 1, 100)} & {t('decisions_diff', 0)} \\\\\n"
    open(OUT + f'tab_perseed{sfx}.tex', 'w').write(rows + '\\bottomrule\n')
    # --- makro
    M = {}
    for v in V:
        t = TAG[v]; r, s = H[v]['rerata'], H[v]['sd']; c = cv(v)
        M.update({f'dvLat{t}': n(r['latency_ms']), f'dvEn{t}': n(r['energy_mj']), f'dvAr{t}': n(r['arena_kb']), f'dvSdLat{t}': n(s['latency_ms']), f'dvSdEn{t}': n(s['energy_mj']), f'dvEf{t}': n(100 * r['f1_100']), f'dvSdEf{t}': n(100 * s['f1_100']),
                  f'dvFile{t}': n(c['flash_kb']), f'dvAct{t}': n(c['aktivasi_lb_kb'], 0), f'dvFst{t}': n(c['f1_stead']), f'dvFm{t}': n(c['f1_mean4']), f'dvDuty{t}': n(100 * r['latency_ms'] / 30720),
                  f'dvTcX{t}': n(sp[v]['tc']), f'dvTotX{t}': n(sp[v]['tot']), f'dvFullX{t}': n(sp[v]['full']), f'dvEnX{t}': n(sp[v]['en']),
                  f'dvLatRef{t}': n(K[v]['lama']['latency_ms']), f'dvArRef{t}': n(K[v]['lama']['arena_kb'], 0)})
    for fam, t in (('1ch', 'One'), ('3ch', 'Thr')):
        f0 = cand[(fam, 'fp32')]['f1_mean4']
        for k, kt in (('q16x8', 'Qq'), ('int8', 'Int'), ('w4a8', 'Four')): M[f'dvD{kt}{t}'] = n(abs(cand[(fam, k)]['f1_mean4'] - f0))
    M.update({'dvTcXMin': n(min(v['tc'] for v in sp.values())), 'dvTcXMax': n(max(v['tc'] for v in sp.values())), 'dvTotXMin': n(min(v['tot'] for v in sp.values())), 'dvTotXMax': n(max(v['tot'] for v in sp.values())),
              'dvFullXMin': n(min(v['full'] for v in sp.values())), 'dvFullXMax': n(max(v['full'] for v in sp.values())), 'dvHashN': str(sum(r['sama'] for r in X['hash'].values())), 'dvHashModels': str(len(X['hash'])),
              'dvSramLatOne': n(s1['lat_ms']), 'dvSramEnOne': n(s1['energi_mj']), 'dvPsLatOne': n(ps1['latency_ms']), 'dvPsEnOne': n(ps1['energy_mj']), 'dvSramPctLat': n(100 * (1 - s1['lat_ms'] / ps1['latency_ms'])), 'dvSramPctEn': n(100 * (1 - s1['energi_mj'] / ps1['energy_mj'])),
              'dvSramNeedThr': n(X['sram_3ch_fp32']['diminta_kb'], 0), 'dvSramHaveThr': n(X['sram_3ch_fp32']['tersedia_kb'], 0), 'dvEnRatioQqFp': n(H['1ch_16x8']['rerata']['energy_mj'] / H['1ch_fp32']['rerata']['energy_mj']), 'dvLatRatioQqFp': n(H['1ch_16x8']['rerata']['latency_ms'] / H['1ch_fp32']['rerata']['latency_ms']),
              'dvLatRatioThrQqOne': n(H['3ch_16x8']['rerata']['latency_ms'] / H['1ch_16x8']['rerata']['latency_ms']),
              'dvDiagPy': ' / '.join(n(100 * o['f1_python']) for o in D.values()), 'dvDiagBd': ' / '.join(n(100 * o['f1_papan']) for o in D.values()), 'dvDiagNear': str(D['43']['dekat_ambang']), 'dvDiagFlip': ' / '.join(str(o['keputusan_beda']) for o in D.values()),
              'dvTcShareMin': n(min(100 * K[v]['ref']['tc_ms'] / K[v]['ref']['invoke_ms'] for v in V), 0), 'dvTcShareMax': n(max(100 * K[v]['ref']['tc_ms'] / K[v]['ref']['invoke_ms'] for v in V), 0),
              'dvTcShareOneFp': n(100 * K['1ch_fp32']['ref']['tc_ms'] / K['1ch_fp32']['ref']['invoke_ms'], 0), 'dvTcShareThrFp': n(100 * K['3ch_fp32']['ref']['tc_ms'] / K['3ch_fp32']['ref']['invoke_ms'], 0),
              'dvDiagLossMax': n(max(100 * (o['f1_python'] - o['f1_papan']) for o in D.values())), 'dvDiagFlipMax': str(max(o['keputusan_beda'] for o in D.values())),
              'dvMacOne': n(MACS['1ch'] / 1e6, 2), 'dvMacThr': n(MACS['3ch'] / 1e6, 2), 'dvMacRatio': n(MACS['3ch'] / MACS['1ch']), 'dvParRatio': n(PARS['3ch'] / PARS['1ch']), 'dvParOne': num(PARS['1ch'], 0, L), 'dvParThr': num(PARS['3ch'], 0, L),
              'dvSdLatMax': n(max(H[v]['sd']['latency_ms'] for v in V)), 'dvSdEnPctMax': n(max(100 * H[v]['sd']['energy_mj'] / H[v]['rerata']['energy_mj'] for v in V)), 'dvNdom': ', '.join(nm(v) for v in NDOM)})
    for fam, t in (('1ch', 'One'), ('3ch', 'Thr')):   # standard-convolution Float32 models on the device (192_)
        r, q = PD[fam]['rerata'], PD[fam]['razao_standar_depthwise']
        M.update({f'dvStdLat{t}': n(r['latency_ms']), f'dvStdEn{t}': n(r['energy_mj']), f'dvStdAr{t}': n(r['arena_kb']), f'dvStdXLat{t}': n(q['latency_ms']), f'dvStdXEn{t}': n(q['energy_mj'])})
    open(OUT + f'numbers_dev{sfx}.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in M.items()))

# ---- figures (English and Indonesian)
plt.rcParams['font.family'] = 'sans-serif'; plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
INK = '#0f172a'; C_OLD, C_NEW = '#cbd5e1', '#1d4ed8'; C1, C3 = '#0f766e', '#c2410c'
TXT = {'en': dict(lat='Latency (ms per window)', en='Energy (mJ per window)', old='Reference kernel', new='Replacement kernel (ratio)', f1='Mean F1-score, four datasets (%)', one='Single-channel', thr='Three-channel', nd='Non-dominated', dom='Dominated', a='(a)', b='(b)'),
       'id': dict(lat='Latensi (ms per jendela)', en='Energi (mJ per jendela)', old='Reference kernel', new='Replacement kernel (rasio)', f1='F1-score rerata empat dataset (%)', one='Satu kanal', thr='Tiga kanal', nd='Tidak terdominasi', dom='Terdominasi', a='(a)', b='(b)')}
for lg, T in TXT.items():
    sfx = '' if lg == 'en' else '_id'; L = LANG[lg]
    lab = [f"{L['name'][v[:3]]} {FMT[{'16x8': 'q16x8'}.get(v[4:], v[4:])]}" for v in V]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), dpi=300); ypos = np.arange(len(V))[::-1].astype(float); h = 0.36
    for ax, (key, title) in zip(axes, (('latency_ms', T['lat']), ('energy_mj', T['en']))):
        old = [K[v]['lama'][key] for v in V]; new = [K[v]['baru'][key] for v in V]; xmax = max(old) * 1.38
        for i, (o, nw) in enumerate(zip(old, new)):
            ax.barh(ypos[i] + h / 2, o, height=h, color=C_OLD, edgecolor=INK, linewidth=0.5, zorder=3); ax.barh(ypos[i] - h / 2, nw, height=h, color=C_NEW, edgecolor=INK, linewidth=0.5, zorder=3)
            ax.text(o + xmax * 0.012, ypos[i] + h / 2, num(o, 0, L), va='center', fontsize=6.4, color='#475569'); ax.text(nw + xmax * 0.012, ypos[i] - h / 2, f"{num(nw, 0, L)} ({o / nw:.1f}$\\times$)".replace('.', L['dec']) if False else f"{num(nw, 0, L)} ({num(o / nw, 1, L)}x)", va='center', fontsize=6.4, fontweight='bold', color=C_NEW)
        ax.set_xlim(0, xmax); ax.set_yticks(ypos); ax.set_yticklabels(lab if ax is axes[0] else [], fontsize=7); ax.set_title(title, fontsize=8.2, fontweight='bold'); ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0); ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=6.8)
        for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
        ax.axhline((ypos[2] + ypos[3]) / 2, color='#94a3b8', linewidth=0.6, linestyle=':')
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=C_OLD, ec=INK), plt.Rectangle((0, 0), 1, 1, fc=C_NEW, ec=INK)], labels=[T['old'], T['new']], loc='upper center', ncol=2, fontsize=7, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(OUT + f'fig_dev_kernel{sfx}.pdf', bbox_inches='tight'); plt.close(fig)
    # Pareto
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), dpi=300)
    MK = {'fp32': 'o', '16x8': 's', 'int8': '^', 'std': 'D'}; STDLAB = {'en': 'Standard', 'id': 'Standar'}[lg]
    for ax, (key, xl) in zip(axes, (('en', T['en']), ('lat', T['lat']))):
        for m in MEAS8:
            fam, fmt_ = m['v'][:3], m['v'][4:]; filled = m['v'] in NDOM8
            ax.scatter(m[key], m['f1'], marker=MK[fmt_], s=48, facecolor=(C1 if fam == '1ch' else C3) if filled else 'white', edgecolor=C1 if fam == '1ch' else C3, linewidth=1.3, zorder=3)
            dx, dy, ha_ = {('3ch', 'int8'): (0, -12, 'center')}.get((fam, fmt_), (5, 4, 'left'))   # three-channel INT8: label below the point so that it does not cover the one-channel 16x8 marker
            ax.annotate(STDLAB if fmt_ == 'std' else FMT[{'16x8': 'q16x8'}.get(fmt_, fmt_)], (m[key], m['f1']), textcoords='offset points', xytext=(dx, dy), ha=ha_, fontsize=6.3, color=INK)
        ax.set_xscale('log'); ax.set_xlabel(xl, fontsize=7.6); ax.set_ylabel(T['f1'] if ax is axes[0] else '', fontsize=7.6); ax.tick_params(labelsize=6.8); ax.grid(linestyle='--', alpha=0.35, zorder=0)
        for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
    hs = [Line2D([0], [0], marker='o', color='w', markerfacecolor=C1, markeredgecolor=C1, markersize=6, label=T['one']), Line2D([0], [0], marker='o', color='w', markerfacecolor=C3, markeredgecolor=C3, markersize=6, label=T['thr']),
          Line2D([0], [0], marker='o', color='w', markerfacecolor=INK, markeredgecolor=INK, markersize=6, label=T['nd']), Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor=INK, markersize=6, label=T['dom'])]
    fig.legend(handles=hs, loc='upper center', ncol=4, fontsize=7, frameon=False); plt.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(OUT + f'fig_dev_pareto{sfx}.pdf', bbox_inches='tight'); plt.close(fig)
print('ok', NDOM, NDOM8)
