"""
Step 194 (device manuscript, Q, standalone accuracy): accuracy of P and S arrival-time picking of the one-channel Float32 TinyPhaseNet on the four test datasets.
Source: results/final/picking_per_seed.json (84_picking_final.py; seeds 42, 43, 44; 'fired' = maximum P probability >= the threshold chosen on STEAD validation;
arrival time = sample of maximum probability; Recall@tol over ALL earthquake windows (P) or windows with S inside the window (S); mean and standard deviation of the residual only from picks with |residual| < 0.5 s).
Output: ../draft_publikasi_1/tabel_dev/tab_pick{,_id}.tex (table rows) and the number macros numbers_pick{,_id}.tex.
Usage (from this folder): python 194_tabel_picking.py
"""
import json, statistics as st
SEEDS = ['42', '43', '44']; OUT = '../draft_publikasi_1/tabel_dev/'
Pk = json.load(open('results/final/picking_per_seed.json'))
ROWS = [('STEAD', 'stead_test', 'P'), ('STEAD', 'stead_test', 'S'), ('INSTANCE', 'instance_test', 'P'), ('ETHZ', 'ethz_test', 'P'), ('TXED', 'txed_test', 'P'), ('TXED', 'txed_test', 'S')]


def num(v, d, dec, thou):
    s = f'{v:,.{d}f}'
    return s.replace(',', '\u0001').replace('.', dec).replace('\u0001', thou)


for sfx, dec, thou in (('', '.', ','), ('_id', ',', '.')):
    rows = ''; M = {}
    for ds, key, ph in ROWS:
        v = [Pk[s][key][ph] for s in SEEDS]; n = v[0]['n_total']
        r1 = [100 * x['recall_0p1'] for x in v]; r5 = [100 * x['recall_0p5'] for x in v]; mu = [x['mean_ms'] for x in v]; sd = [x['std_ms'] for x in v]
        pm = lambda xs, d=1: f"{num(st.mean(xs), d, dec, thou)} $\\pm$ {num(st.stdev(xs), d, dec, thou)}"
        rows += f"{ds} & {ph} & {num(n, 0, dec, thou)} & {pm(r1)} & {pm(r5)} & {num(st.mean(mu), 1, dec, thou)} & {num(st.mean(sd), 1, dec, thou)} \\\\\n"
        if ds == 'STEAD' and ph == 'P': M.update({'dvPkStP': num(st.mean(r5), 1, dec, thou), 'dvPkStPFine': num(st.mean(r1), 1, dec, thou), 'dvPkStMs': num(st.mean(mu), 0, dec, thou)})
        if ds == 'STEAD' and ph == 'S': M.update({'dvPkStS': num(st.mean(r5), 1, dec, thou)})
        if ds == 'TXED' and ph == 'S': M.update({'dvPkTxS': num(st.mean(r5), 1, dec, thou)})
        if ds == 'INSTANCE': M.update({'dvPkIn': num(st.mean(r5), 1, dec, thou)})
        if ds == 'ETHZ': M.update({'dvPkEt': num(st.mean(r5), 1, dec, thou)})
        if ds == 'TXED' and ph == 'P': M.update({'dvPkTx': num(st.mean(r5), 1, dec, thou)})
        if ds == 'INSTANCE' and True: pass
        if ds == 'ETHZ' and True: pass
        if ds == 'STEAD' and ph == 'P': M['dvPkNStead'] = num(n, 0, dec, thou)
    allmu = [st.mean([Pk[s_][k][p]['mean_ms'] for s_ in SEEDS]) for _, k, p in ROWS]; M.update({'dvPkMsMin': num(min(allmu), 0, dec, thou), 'dvPkMsMax': num(max(allmu), 0, dec, thou)})
    open(OUT + f'tab_pick{sfx}.tex', 'w').write(rows + '\\bottomrule\n')
    open(OUT + f'numbers_pick{sfx}.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in M.items()))
print(open(OUT + 'tab_pick.tex').read()); print(open(OUT + 'numbers_pick.tex').read())
