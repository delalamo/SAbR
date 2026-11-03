"""Extra samples for DE2 verification (same feature code as v01):
  fresh_matched     : a NEW random matched clean-EM control set (year x chain type x nanobody, 3 per confirmed error unit,
                      seed 2026, excluding units already in the investigator's sample)
  same_entry_correct: chains with no disagreement run of any kind in the same EM entries as the D/E errors
"""
import sys, json, time
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE2_framework_distortion_integrity')
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import v01_recompute_fw as V

if __name__ == '__main__':
    D, SP, OUT = V.D, V.SP, V.OUT
    C = pd.read_csv(f'{D}/chains.csv.gz'); ent = pd.read_csv(f'{D}/entries.csv')
    RA = pd.read_csv(f'{D}/runs_all.csv')
    A = pd.read_csv(f'{OUT}/analysis_table.csv')
    e = A[(A.group == 'error') & (A.confirm != 'not confirmed')]
    S = pd.read_csv(f'{D}/cache/case_files_DE/struct_sample.csv')
    used = set(S.unit)
    clean = set(ent[~ent.err_any].pdb)
    pool = C[(C.method == 'EM') & C.pdb.isin(clean) & ~C.unit.isin(used)].drop_duplicates('unit')
    rng = np.random.default_rng(2026); picks = []
    for (y, ct, vh), g in e.groupby(['year', 'ctype', 'vhh']):
        cand = pool[(pool.year == y) & (pool.ctype == ct) & (pool.vhh == vh)]
        n = min(len(cand), 3 * len(g))
        if n: picks.append(cand.sample(n, random_state=int(rng.integers(1e9))))
    fm = pd.concat(picks).assign(group='fresh_matched')
    anyk = set(zip(RA.pdb, RA.chain))
    se = C[C.pdb.isin(e.pdb) & ~pd.Series([k in anyk for k in zip(C.pdb, C.chain)], index=C.index)].assign(group='same_entry_correct')
    sel = pd.concat([fm, se])
    print(sel.groupby('group').agg(chains=('pdb', 'size'), units=('unit', 'nunique'), entries=('pdb', 'nunique')), flush=True)
    keys = set(zip(sel.pdb, sel.chain))
    want = {('5dt1', 'H'), ('6cbv', 'H'), ('4jqi', 'H'), ('4m48', 'H'), ('1n8z', 'A'), ('6cbv', 'L'), ('4jqi', 'L'), ('4m48', 'L'),
            ('5dt1', 'L'), ('5cck', 'L'), ('6axk', 'B'), ('5iq9', 'B'), ('4b50', 'A'), ('7z1d', 'FFF'), ('6yz5', 'F'), ('7jkm', 'K')}
    tp = {}
    for l in open(f'{SP}/retest/templates.jsonl'):
        r = json.loads(l)
        if r.get('status') == 'ok' and (r['pdb'], r['chain']) in want: tp[(r['pdb'], r['chain'])] = r['residues']
    REF = {k: V.ref_coords(*k, tp[k]) for k in want}
    ORIG = {'H': REF[('5dt1', 'H')], 'K': REF[('1n8z', 'A')], 'L': REF[('5dt1', 'L')]}
    PANEL = {'H': [REF[k] for k in [('5dt1', 'H'), ('6cbv', 'H'), ('4jqi', 'H'), ('4m48', 'H')]],
             'K': [REF[k] for k in [('1n8z', 'A'), ('6cbv', 'L'), ('4jqi', 'L'), ('4m48', 'L')]],
             'L': [REF[k] for k in [('5dt1', 'L'), ('5cck', 'L'), ('6axk', 'B'), ('5iq9', 'B')]],
             'VHH': [REF[k] for k in [('4b50', 'A'), ('7z1d', 'FFF'), ('6yz5', 'F'), ('7jkm', 'K')]]}
    rec = {}
    for fn in ['validation', 'snapshot_rest', 'new']:
        for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
            r = json.loads(l)
            if r.get('status') != 'ok': continue
            k = (r['pdb'], r['chain'])
            if k in keys and k not in rec: rec[k] = r['residues']
    runpos = {}
    for r in RA.itertuples():
        runpos.setdefault((r.pdb, r.chain), set()).update(range(int(r.imgt0) - 1, int(r.imgt1) + 2))
    info = dict(zip(zip(C.pdb, C.chain), zip(C.ctype, C.vhh)))
    jobs = []
    for p, g in sel.drop_duplicates(['pdb', 'chain']).groupby('pdb'):
        items = [(c, rec[(p, c)], runpos.get((p, c), set()), *info[(p, c)]) for c in sorted(set(g.chain)) if (p, c) in rec]
        for ctk in sorted({(it[3], it[4]) for it in items}):
            its = [it for it in items if (it[3], it[4]) == ctk]
            jobs.append((p, its, {'orig': ORIG[ctk[0]], 'panel': PANEL['VHH' if ctk[1] == 1 else ctk[0]]}))
    print('jobs', len(jobs), flush=True)
    res = []; t0 = time.time()
    with ProcessPoolExecutor(2) as ex:
        for i, o in enumerate(ex.map(V.work, jobs, chunksize=1)):
            res.extend(o)
            if i % 25 == 0: print(i, round(time.time() - t0), flush=True)
    F = pd.DataFrame(res)
    F.to_csv(f'{OUT}/fw_recomputed_extra.csv', index=False)
    sel[['pdb', 'chain', 'group', 'unit', 'paper', 'year', 'ctype', 'vhh', 'res', 'species', 'v_id', 'v_gene', 'L1', 'L2', 'L3', 'sars2', 'deposit']].to_csv(
        f'{OUT}/sample_groups_extra.csv', index=False)
    print(F.status.value_counts())
