#!/usr/bin/env python3
"""Re-derives result files of the manuscripts from the stored measurement summaries, trained models and evaluation results in this release folder
and checks that they equal the stored ones. The raw device logs are not archived, so the scripts that parse them are not run (see README).

The analysis scripts expect the original folder layout (results/, models/, firmware/tinyphasenet_profiling/results_hw/).
This script builds that layout in _work/ (results copied, models and logs only linked), runs the analysis scripts in order, and compares every result
file that a script rewrote (results/final/*.json) with the stored file in this folder. Some scripts also write LaTeX fragments and figures into
sibling folders of _work/; these are by-products and are not part of the release.
Usage:  python3 reproduce.py [--python /path/to/python]
Needs:  the Python environment of requirements.txt / environment.yml (tensorflow is only used by 187_tabel_depthwise.py).
Exit status: 0 if every script ran and every rewritten result file is identical to the stored one.
"""
import argparse, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, '_work')
EXP = os.path.join(WORK, 'eksperimen_tinyphasenet_v2')
ORDER = ['193_pareto_delapan.py', '187_tabel_depthwise.py', '189_suplemen_per_seed.py', '202_mac_flops.py', '196_tabel_pendamping.py',
         '198_tabel_picking_formatos.py', '201_tabel_tangga.py', '232_pareto_dua_belas.py']   # the scripts that parse the raw device logs (164, 169, 175-177, 190, 192, 195, 231) are not run: the logs are not archived


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--python', default=sys.executable)
    a = ap.parse_args()

    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(EXP)
    # scripts (flat) and core modules
    for sub in ('core', 'pipeline', 'analysis'):
        for f in os.listdir(os.path.join(HERE, 'scripts', sub)):
            if f.endswith('.py'):
                shutil.copy2(os.path.join(HERE, 'scripts', sub, f), os.path.join(EXP, f))
    # results: copied (the scripts rewrite some of the JSON); models and logs: read-only links
    shutil.copytree(os.path.join(HERE, 'results'), os.path.join(EXP, 'results'))
    link(os.path.join(HERE, 'models'), os.path.join(EXP, 'models'))
    link(os.path.join(HERE, 'firmware'), os.path.join(EXP, 'firmware', 'tinyphasenet_profiling'))
    # sibling output folders for the by-products of the scripts
    for d in ('draft_publikasi_1/tabel_dev', 'draft_publikasi_1/tabel_pend', 'jawaban_prof_arief_fase1/tabel_spektrum', 'jawaban_prof_arief_fase1/tabel_pn', 'supplement'):
        os.makedirs(os.path.join(WORK, d), exist_ok=True)

    final = os.path.join(EXP, 'results', 'final')
    before = {f: os.path.getmtime(os.path.join(final, f)) for f in os.listdir(final) if f.endswith('.json')}
    t0 = time.time(); time.sleep(0.05)
    failed = []
    for s in ORDER:
        r = subprocess.run([a.python, s], cwd=EXP, capture_output=True, text=True)
        status = 'ok' if r.returncode == 0 else 'FAILED'
        print(f'{status:6s} {s}')
        if r.returncode != 0:
            failed.append(s)
            print('   ', (r.stderr or r.stdout).strip().splitlines()[-1][:200] if (r.stderr or r.stdout).strip() else '')

    same, diff = [], []
    for f in sorted(os.listdir(final)):
        if not f.endswith('.json'):
            continue
        p = os.path.join(final, f)
        if f in before and os.path.getmtime(p) <= t0:
            continue                                   # not rewritten by any script
        stored = os.path.join(HERE, 'results', 'final', f)
        if os.path.exists(stored) and json.load(open(stored)) == json.load(open(p)):
            same.append(f)
        else:
            diff.append(f)
    print(f'\nresult files rewritten by the scripts: {len(same) + len(diff)}; identical to the stored ones: {len(same)}; different: {len(diff)}')
    for f in diff:
        print('  different ', f)
    sys.exit(1 if (failed or diff) else 0)


if __name__ == '__main__':
    main()
