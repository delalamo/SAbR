"""Independent re-derivation of the paper-confirmed case set (verify_antibody_ordinary).
Written from scratch: Table S1 x SAbR records x runs_all.csv x synthesis artifact ledger.
Differences from build_cases.py on purpose: one record source for both the V-fraction and the run mapping
(validation -> snapshot_rest -> new, later wins), and a second matching rule (any overlapping run, not only the
best-overlap run) to see whether the 'best run' choice changes any status."""
import json, re, sys
import numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_antibody_ordinary/tables'
import os; os.makedirs(OUT, exist_ok=True)

recs = {}
alt = {}  # first-seen record (to detect precedence differences)
for fn in ['validation.jsonl', 'snapshot_rest.jsonl', 'new.jsonl']:
    with open(f'{SP}/sabr/out/{fn}') as fh:
        for line in fh:
            r = json.loads(line)
            if r.get('status') != 'ok' or not r.get('residues'):
                continue
            k = (r['pdb'], r['chain'])
            if k in recs and recs[k] != r['residues']:
                alt[k] = recs[k]
            recs[k] = r['residues']
print('records', len(recs), 'keys with differing duplicate records', len(alt))

C = pd.read_csv(f'{SP}/deeper/chains.csv.gz')
chain_keys = set(zip(C.pdb, C.chain))
T = pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx')
pc = T.PDB_CHAIN_ID.str.split(':', n=1, expand=True)
T['pdb'] = pc[0].str.lower(); T['chain'] = pc[1]
m = T.RESIDUE_RANGE.astype(str).str.extract(r'^\s*(-?\d+)\s*-\s*(-?\d+)\s*$')
print('unparsed ranges', int(m[0].isna().sum()))
T['rs'] = m[0].astype(float); T['re'] = m[1].astype(float)
T = T[[k in chain_keys for k in zip(T.pdb, T.chain)]].copy()
print('Table S1 rows on scanned chains', len(T))

def vfrac(row, rec):
    sel = [x for x in rec if row.rs <= x[0] <= row.re]
    return (np.mean([x[4] is not None for x in sel]) if sel else 0.0), len(sel)
T['vfrac'], T['nres'] = zip(*[vfrac(r, recs[(r.pdb, r.chain)]) for r in T.itertuples()])
# alternative precedence (first-seen record) for the V fraction, to check sensitivity
T['vfrac_alt'] = [vfrac(r, alt.get((r.pdb, r.chain), recs[(r.pdb, r.chain)]))[0] for r in T.itertuples()]
print('>=50% V:', int((T.vfrac >= .5).sum()), '| alt precedence:', int((T.vfrac_alt >= .5).sum()))
T = T[T.vfrac >= .5].copy()

R = pd.read_csv(f'{SP}/deeper/runs_all.csv')
L = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/synthesis/artifact_ledger_runs.csv')
L = L[['pdb', 'chain', 'imgt0', 'imgt1', 'flag']].drop_duplicates()
R = R.merge(L, on=['pdb', 'chain', 'imgt0', 'imgt1'], how='left')
bad = 0
sets = []
for r in R.itertuples():
    v = [x for x in recs[(r.pdb, r.chain)] if x[4] is not None]
    if not (v[r.i0][4] == r.imgt0 and v[r.i1][4] == r.imgt1):
        bad += 1
    sets.append({v[j][0] for j in range(r.i0, r.i1 + 1)})
R['auth'] = sets
print('runs with IMGT index mismatch', bad)

STRICT_ART = {'indel_gap_artifact_confirmed', 'DE_chain_end_artifact', 'amyloid_fibril', 'not_a_register_shift'}
LOOSE = {'indel_gap_artifact_probable', 'DE_not_confirmed_other'}
def classify(run):
    cls = str(run.cls)
    if cls in ('1a', '1b', '2'): return 'copy of an earlier model'
    if cls == '4': return 'known detector artifact (origin 4)'
    f = str(run.flag)
    if f in STRICT_ART: return 'artifact (synthesis ledger, strict)'
    if f == 'near_copy': return 'near copy (ledger)'
    if f in LOOSE: return 'confirmed, loose-ledger doubt'
    return 'confirmed first occurrence'

rows = []
for e in T.itertuples():
    fl = set(range(int(e.rs), int(e.re) + 1))
    cand = R[(R.pdb == e.pdb) & (R.chain == e.chain)]
    ov = [(len(fl & r.auth), r) for r in cand.itertuples()]
    ov = [(o, r) for o, r in ov if o > 0]
    d = dict(pdb=e.pdb, chain=e.chain, s1_range=e.RESIDUE_RANGE, n_overlap_runs=len(ov))
    if ov:
        o, best = max(ov, key=lambda t: t[0])
        d.update(overlap=o, imgt0=best.imgt0, imgt1=best.imgt1, cls=str(best.cls), mode=best.mode, status=classify(best))
        sts = {classify(r) for _, r in ov}
        d['any_first'] = 'confirmed first occurrence' in sts
        d['all_status'] = '|'.join(sorted(sts))
    else:
        d.update(overlap=0, status='not confirmed by SAbR/ANARCI', any_first=False, all_status='')
    rows.append(d)
V = pd.DataFrame(rows)
V.to_csv(f'{OUT}/v01_paper_errors_rederived.csv', index=False)
print(V.status.value_counts().to_string())
print('errors where a non-best overlapping run is a first occurrence but best is not:',
      int((V.any_first & (V.status != 'confirmed first occurrence')).sum()))

# compare with investigator's annotation
A = pd.read_csv(f'{SP}/paperconf/paper_errors_annotated.csv')
key = ['pdb', 'chain', 's1_range']
mg = A[key + ['status', 'imgt0', 'imgt1']].merge(V[key + ['status', 'imgt0', 'imgt1']], on=key, how='outer',
                                                 suffixes=('_inv', '_me'), indicator=True)
print('merge', mg._merge.value_counts().to_dict())
diff = mg[(mg._merge != 'both') | (mg.status_inv != mg.status_me)]
print('status discrepancies', len(diff)); print(diff.to_string())
diff.to_csv(f'{OUT}/v01_discrepancies.csv', index=False)

# chain / unit level
C = C.set_index(['pdb', 'chain'])
strict = set(zip(V[V.status == 'confirmed first occurrence'].pdb, V[V.status == 'confirmed first occurrence'].chain))
F = pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz')
inv = set(zip(F[F.paper_case_strict].pdb, F[F.paper_case_strict].chain))
print('strict chains me', len(strict), 'inv', len(inv), 'sym diff', sorted(strict ^ inv))
d = C.loc[sorted(strict)].reset_index()
print('entries', d.pdb.nunique(), 'units', d.unit.nunique(), 'units(paper,vseq)', len(d.drop_duplicates(['paper', 'vseq'])),
      'papers', d.paper.nunique(), 'EM', int((d.method == 'EM').sum()), 'Xray', int((d.method != 'EM').sum()))
print('release max', d.release.max())
# controls
Cr = C.reset_index()
s1 = set(T.pdb)  # entries in Table S1 among scanned chains only -- check vs all Table S1 entries
s1_all = set(pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx').PDB_CHAIN_ID.str.split(':').str[0].str.lower())
ctrl = (Cr.release.astype(str) <= '2022-04-05') & ~Cr.pdb.isin(s1_all) & ~Cr.err_any.astype(bool)
print('controls chains', int(ctrl.sum()), 'inv', int(F.paper_control.sum()),
      'agree', int((ctrl.values == F.set_index(['pdb', 'chain']).loc[list(zip(Cr.pdb, Cr.chain)), 'paper_control'].values).sum()))
