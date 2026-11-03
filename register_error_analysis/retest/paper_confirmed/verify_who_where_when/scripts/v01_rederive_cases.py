"""Independent re-derivation of the paper-confirmed case set (adversarial check).

Differences in approach from paperconf/build_cases.py, on purpose:
  * one record per (pdb, chain), read in the order validation -> snapshot_rest -> new (later wins), used for BOTH the
    V-fraction of the Table S1 range and the residue numbers of each run (build_cases uses a different priority for
    the V-fraction);
  * every run that overlaps the S1 range is kept, not only the best-overlap run, so a S1 error that overlaps a
    first-occurrence run and a copy run can be flagged;
  * the status is derived from runs_all.cls / copied_or_artifact plus the synthesis ledger flag.
Writes tables/v01_rederived_cases.csv and prints a diff against paper_errors_annotated.csv.
"""
import json, sys, pandas as pd, numpy as np
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_who_where_when/tables'
LEDGER = '/home/user/SAbR/register_error_analysis/retest/deeper/synthesis/artifact_ledger_runs.csv'

rec = {}; nrec = {}
for fn in ['validation.jsonl', 'snapshot_rest.jsonl', 'new.jsonl']:
    with open(f'{SP}/sabr/out/{fn}') as fh:
        for line in fh:
            r = json.loads(line)
            if r.get('status') == 'ok' and r.get('residues'):
                k = (r['pdb'], r['chain'])
                rec[k] = r['residues']; nrec[k] = nrec.get(k, 0) + 1
print('records', len(rec), '| keys in >1 file', sum(v > 1 for v in nrec.values()))

chains = pd.read_csv(f'{SP}/deeper/chains.csv.gz')
ckeys = set(zip(chains.pdb, chains.chain))

s1 = pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx')
pc = s1.PDB_CHAIN_ID.str.split(':', n=1, expand=True)
s1['pdb'] = pc[0].str.lower(); s1['chain'] = pc[1]
m = s1.RESIDUE_RANGE.astype(str).str.extract(r'^\s*(-?\d+)\s*-\s*(-?\d+)\s*$')
print('S1 rows', len(s1), '| unparsable ranges', int(m[0].isna().sum()))
s1['lo'] = m[0].astype(float); s1['hi'] = m[1].astype(float)
s1 = s1[s1.lo.notna()].copy(); s1[['lo', 'hi']] = s1[['lo', 'hi']].astype(int)
print('S1 rows on scanned single-domain V chains', int(sum(k in ckeys for k in zip(s1.pdb, s1.chain))))
# case-insensitive chain match would add:
lowkeys = {(p, c.upper()) for p, c in ckeys}
print('S1 rows matching only case-insensitively', int(sum((k not in ckeys) and ((k[0], str(k[1]).upper()) in lowkeys)
                                                          for k in zip(s1.pdb, s1.chain))))
s1 = s1[[k in ckeys for k in zip(s1.pdb, s1.chain)]].copy()

def vfrac(r):
    res = rec.get((r.pdb, r.chain), [])
    inr = [x for x in res if r.lo <= x[0] <= r.hi]
    return (np.mean([x[4] is not None for x in inr]) if inr else np.nan), len(inr)
vv = s1.apply(vfrac, axis=1, result_type='expand'); s1['vfrac'] = vv[0]; s1['n_in_range'] = vv[1]
print('S1 errors with no residue found in range', int((s1.n_in_range == 0).sum()))
s1v = s1[s1.vfrac >= 0.5].copy()
print('S1 errors >=50% in V region', len(s1v), '| unique chains', s1v[['pdb', 'chain']].drop_duplicates().shape[0])

runs = pd.read_csv(f'{SP}/deeper/runs_all.csv')
led = pd.read_csv(LEDGER)[['pdb', 'chain', 'imgt0', 'imgt1', 'flag']]
print('ledger duplicate keys', int(led.duplicated(['pdb', 'chain', 'imgt0', 'imgt1']).sum()))
runs = runs.merge(led.drop_duplicates(['pdb', 'chain', 'imgt0', 'imgt1']), on=['pdb', 'chain', 'imgt0', 'imgt1'], how='left')
bad = 0; resn = []
for r in runs.itertuples():
    v = [x for x in rec[(r.pdb, r.chain)] if x[4] is not None]
    ok = 0 <= r.i0 <= r.i1 < len(v) and v[r.i0][4] == r.imgt0 and v[r.i1][4] == r.imgt1
    bad += not ok
    resn.append({v[j][0] for j in range(r.i0, r.i1 + 1)} if ok else set())
runs['resn'] = resn
print('runs whose i0/i1 do not match imgt0/imgt1 in the chosen record:', bad)

STRICT_ART = {'indel_gap_artifact_confirmed', 'DE_chain_end_artifact', 'amyloid_fibril', 'not_a_register_shift'}
def rstatus(r):
    c = str(r.cls)
    if c in ('1a', '1b', '2'): return 'copy'
    if c == '4': return 'detector_artifact'
    f = str(r.flag)
    if f in STRICT_ART: return 'artifact'
    if f == 'near_copy': return 'near_copy'
    if f in ('indel_gap_artifact_probable', 'DE_not_confirmed_other'): return 'loose_doubt'
    return 'first'
runs['rstat'] = runs.apply(rstatus, axis=1)

rows = []
for e in s1v.itertuples():
    fl = set(range(e.lo, e.hi + 1))
    cand = runs[(runs.pdb == e.pdb) & (runs.chain == e.chain)]
    ov = [(len(fl & r.resn), r) for r in cand.itertuples() if fl & r.resn]
    ov.sort(key=lambda t: -t[0])
    stats = [r.rstat for _, r in ov]
    best = ov[0][1] if ov else None
    st = {'first': 'confirmed first occurrence', 'copy': 'copy of an earlier model', 'near_copy': 'near copy (ledger)',
          'artifact': 'artifact (synthesis ledger, strict)', 'loose_doubt': 'confirmed, loose-ledger doubt',
          'detector_artifact': 'known detector artifact (origin 4)'}[best.rstat] if best is not None else 'not confirmed by SAbR/ANARCI'
    rows.append(dict(pdb=e.pdb, chain=e.chain, s1_range=e.RESIDUE_RANGE, lo=e.lo, hi=e.hi, vfrac=e.vfrac,
                     n_overlap_runs=len(ov), best_overlap=ov[0][0] if ov else 0, v_status=st,
                     any_first=('first' in stats), all_stats='|'.join(stats),
                     imgt0=best.imgt0 if best is not None else np.nan, mode=best.mode if best is not None else None))
V = pd.DataFrame(rows)
V.to_csv(f'{OUT}/v01_rederived_cases.csv', index=False)
print(V.v_status.value_counts().to_string())
print('S1 errors overlapping >1 run:', int((V.n_overlap_runs > 1).sum()))
print('best run not first but some overlapping run is first:', int(((V.v_status != 'confirmed first occurrence') & V.any_first).sum()))

P = pd.read_csv(f'{SP}/paperconf/paper_errors_annotated.csv')
P['k'] = P.pdb + ':' + P.chain + ':' + P.s1_range.astype(str); V['k'] = V.pdb + ':' + V.chain + ':' + V.s1_range.astype(str)
print('investigator rows', len(P), 'mine', len(V))
print('only investigator:', sorted(set(P.k) - set(V.k)))
print('only mine:', sorted(set(V.k) - set(P.k)))
J = P.merge(V, on='k', suffixes=('_inv', '_mine'))
d = J[J.status != J.v_status]
print('status disagreements:', len(d)); print(d[['k', 'status', 'v_status', 'all_stats']].to_string())

# chain-level strict set and summary
strict = V[V.v_status == 'confirmed first occurrence'][['pdb', 'chain']].drop_duplicates()
F = pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz')
inv = F[F.paper_case_strict][['pdb', 'chain']]
a = set(zip(strict.pdb, strict.chain)); b = set(zip(inv.pdb, inv.chain))
print('strict chains mine', len(a), 'investigator', len(b), '| diff', sorted(a ^ b))
C = chains.merge(strict.assign(mine=True), on=['pdb', 'chain'], how='left'); C['mine'] = C.mine.fillna(False).astype(bool)
d = C[C.mine]; u = d.drop_duplicates(['paper', 'vseq'])
print('mine: chains', len(d), 'entries', d.pdb.nunique(), 'units', len(u), 'papers', d.paper.nunique(),
      'EM chains', int((d.method == 'EM').sum()), 'X-ray', int((d.method != 'EM').sum()),
      'EM entries', d[d.method == 'EM'].pdb.nunique(), 'X entries', d[d.method != 'EM'].pdb.nunique(),
      'EM papers', d[d.method == 'EM'].paper.nunique(), 'max release', d.release.max())
print(d.groupby('paper').pdb.nunique().sort_values(ascending=False).head(6).to_string())
