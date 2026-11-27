#!/usr/bin/env python3
"""Regenerates the tables, number macros, and figures of the device manuscript from the results and logs in this release folder.

The paper scripts expect the original folder layout (results/, firmware/tinyphasenet_profiling/results_hw/, and sibling output folders).
This script builds that layout in _work/ (results copied, models and logs only linked) and then runs the scripts in order.
Usage:  python3 reproduce.py [--python /path/to/python] [--compare /path/to/original/tabel_dev]
Output: _work/draft_publikasi_1/tabel_dev/ (.tex tables, number macros, figures), _work/draft_publikasi_1/tabel_pend/ (companion manuscript) and _work/jawaban_prof_arief_fase1/.
Needs:  Python with numpy, matplotlib, tensorflow (only for 167_diagram_arsitektur.py and 187_tabel_depthwise.py).
"""
import argparse, filecmp, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, '_work')
EXP = os.path.join(WORK, 'eksperimen_tinyphasenet_v2')
ORDER = ['164_pareto_kandidat.py', '169_tabel_kernel_transpos.py', '175_ringkas_hw_tiga_seed.py', '176_diagnosis_int8_satu_kanal.py',
         '177_hash_kernel_dan_sram.py', '190_ketidakpastian_statistik.py', '192_ringkas_hw_padrao_tiga_seed.py', '193_pareto_delapan.py', '194_tabel_picking.py', '178_tabel_draft_perangkat.py', '179_gambar_profil_kernel.py', '187_tabel_depthwise.py', '188_gambar_depthwise.py', '189_suplemen_per_seed.py',
         '202_mac_flops.py', '196_tabel_pendamping.py', '198_tabel_picking_formatos.py', '201_tabel_tangga.py',
         '231_ringkas_hw_padrao_kuantisasi.py', '232_pareto_dua_belas.py', '234_sinkron_draf_perangkat.py', '235_sandingan_draf_pendamping.py', '236_gambar_protokol_sederhana.py']   # twelve configurations (standard PhaseNet 16x8 and INT8 measured on 27 Nov 2026) and simulation-versus-measurement comparison for the companion manuscript   # last three scripts: companion manuscript (tables and figures in draft_publikasi_1/tabel_pend)
ARG_SCRIPTS = [('167_diagram_arsitektur.py', 'en'), ('167_diagram_arsitektur.py', 'id')]


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--compare', default=None, help='original tabel_dev folder to compare with the regenerated output')
    ap.add_argument('--compare-pend', default=None, help='original tabel_pend folder (companion manuscript) to compare with the regenerated output')
    a = ap.parse_args()

    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(EXP)
    # scripts (flat) and core modules
    for sub in ('core', 'pipeline', 'paper'):
        for f in os.listdir(os.path.join(HERE, 'scripts', sub)):
            if f.endswith('.py'):
                shutil.copy2(os.path.join(HERE, 'scripts', sub, f), os.path.join(EXP, f))
    # results: copied (the scripts rewrite some of the JSON); models and logs: read-only links
    shutil.copytree(os.path.join(HERE, 'results'), os.path.join(EXP, 'results'))
    link(os.path.join(HERE, 'models'), os.path.join(EXP, 'models'))
    link(os.path.join(HERE, 'firmware'), os.path.join(EXP, 'firmware', 'tinyphasenet_profiling'))
    # sibling output folders
    for d in ('draft_publikasi_1/tabel_dev', 'draft_publikasi_1/tabel_pend', 'jawaban_prof_arief_fase1/tabel_spektrum', 'ringkasan_hasil_pendamping_perangkat/tabel', 'ringkasan_hasil_pendamping_perangkat/gambar'):
        os.makedirs(os.path.join(WORK, d), exist_ok=True)

    shutil.copy2(os.path.join(HERE, 'results', 'ringkasan', 'tab_perangkat.tex'), os.path.join(WORK, 'ringkasan_hasil_pendamping_perangkat', 'tabel', 'tab_perangkat.tex'))   # input of 231_ (six TinyPhaseNet rows)
    failed = []
    jobs = [(s, []) for s in ORDER] + [(s, [m]) for s, m in ARG_SCRIPTS]
    for s, extra in jobs:
        r = subprocess.run([a.python, s] + extra, cwd=EXP, capture_output=True, text=True)
        status = 'ok' if r.returncode == 0 else 'FAILED'
        print(f'{status:6s} {s} {" ".join(extra)}')
        if r.returncode != 0:
            failed.append(s)
            print('   ', (r.stderr or r.stdout).strip().splitlines()[-1][:200] if (r.stderr or r.stdout).strip() else '')

    if a.compare:
        out = os.path.join(WORK, 'draft_publikasi_1', 'tabel_dev')
        same, diff, absent = 0, [], []
        for f in sorted(os.listdir(a.compare)):
            if not f.endswith('.tex'):
                continue
            g = os.path.join(out, f)
            if not os.path.exists(g):
                absent.append(f)
            elif filecmp.cmp(os.path.join(a.compare, f), g, shallow=False):
                same += 1
            else:
                diff.append(f)
        print(f'\ncomparison with {a.compare}: same {same}, different {len(diff)}, not generated {len(absent)}')
        for f in diff:
            print('  different ', f)
        for f in absent:
            print('  absent ', f)
    if a.compare_pend:
        out = os.path.join(WORK, 'draft_publikasi_1', 'tabel_pend'); same, diff, absent = 0, [], []
        for f in sorted(os.listdir(a.compare_pend)):
            if not f.endswith('.tex'):
                continue
            g = os.path.join(out, f)
            if not os.path.exists(g):
                absent.append(f)
            elif filecmp.cmp(os.path.join(a.compare_pend, f), g, shallow=False):
                same += 1
            else:
                diff.append(f)
        print(f'\ncomparison with {a.compare_pend}: same {same}, different {len(diff)}, not generated {len(absent)}')
        for f in diff:
            print('  different ', f)
        for f in absent:
            print('  absent ', f)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
