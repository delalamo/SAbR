"""Circularity / detector checks for 'atypical FR3 length -> mode C'.
1. Where do FR3-atypical units go? (C first-occurrence, clean control, copied/artifact-only, other-error)
2. Same tabulation for 3-5 A X-ray units (the detector, not cryo-EM modelling, should dominate if X-ray behaves like EM).
3. Gap-pattern stratification: C-run rate (any run, incl. artifact/copied) by ANARCI FR3 gap pattern, EM and X-ray.
4. Run-to-gap adjacency: is each first-occurrence C run within 3 residues of the chain's ANARCI FR3 gap/insertion?
Outputs tables/v02_*.csv"""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
DE = '/home/user/SAbR/register_error_analysis/retest/deeper'
OUT = f'{DE}/verify/case_files_ABCF_FR3_INDEL_C_integrity'
amy = set(pd.read_csv(f'{DE}/pdb_meta/tables/amyloid_fibril_chains.csv').pdb.str.lower())
RA = pd.read_csv(f'{SP}/deeper/runs_all.csv'); RA = RA[~RA.pdb.isin(amy)]
RF = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv'); RF = RF[~RF.pdb.isin(amy)]
C = pd.read_pickle(f'{SP}/deeper/cache/verify_fr3_integrity/chains_feat.pkl')
Cu = C.set_index(['pdb', 'chain']).unit.to_dict()
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
out = []
for meth in ['EM', 'X-ray']:
    U = pd.read_pickle(f'{SP}/deeper/cache/verify_fr3_integrity/units_{"em" if meth == "EM" else "xray"}.pkl')
    U['yC'] = U.modes.map(lambda s: 'C_FR3' in s); U['C_any'] = U.modes_all.map(lambda s: 'C_FR3' in s)
    U['grp'] = ['C_first' if y else ('clean' if not ea else ('other_first' if ef else 'copied/artifact_only')) for y, ea, ef in zip(U.yC, U.err_any, U.err_first)]
    t = pd.crosstab(U.grp, U.fr3_atyp); t['method'] = meth; out.append(t.reset_index())
    print(meth); print(t)
    for x in [1, 0]:
        g = U[U.fr3_atyp == x]
        print(f'  fr3_atyp={x}: n={len(g)}  any C run {g.C_any.sum()} ({g.C_any.mean():.1%})  first-occ C {g.yC.sum()} ({g.yC.mean():.1%})')
pd.concat(out).to_csv(f'{OUT}/tables/v02_fate_of_fr3_atyp_units.csv', index=False)

# origin of runs in copied/artifact-only FR3-atypical EM units
U = pd.read_pickle(f'{SP}/deeper/cache/verify_fr3_integrity/units_em.pkl')
ca = U[(U.err_any) & (~U.err_first) & (U.fr3_atyp == 1)].unit
R = RA[RA['mode'] == 'C_FR3'].copy(); R['unit'] = [Cu.get((p, c)) for p, c in zip(R.pdb, R.chain)]
print('origin of C runs in copied/artifact-only FR3-atyp EM units:')
print(R[R.unit.isin(ca)].drop_duplicates('unit').origin.value_counts().to_string())

# 3. gap-pattern stratification, all scan units (EM + X-ray), any C run (first or copied/artifact)
rows = []
for meth in ['EM', 'X-ray']:
    U = pd.read_pickle(f'{SP}/deeper/cache/verify_fr3_integrity/units_{"em" if meth == "EM" else "xray"}.pkl')
    U['C_any'] = U.modes_all.map(lambda s: 'C_FR3' in s); U['yC'] = U.modes.map(lambda s: 'C_FR3' in s)
    U['pat'] = [str(g) if a else 'typical' for g, a in zip(U.fr3_gaps, U.fr3_atyp)]
    U['pat'] = U.cclass + ' ' + U.pat
    for (pat), g in U.groupby('pat'):
        rows.append(dict(method=meth, pattern=pat, n_units=len(g), any_C_run=int(g.C_any.sum()), first_occ_C=int(g.yC.sum()),
                         clean=int((~g.err_any).sum())))
P = pd.DataFrame(rows)
P = P[(P.n_units >= 1) & ~P.pattern.str.endswith('typical')].sort_values(['pattern', 'method'])
P.to_csv(f'{OUT}/tables/v02_gap_pattern_rates.csv', index=False)
print(P.to_string())

# 4. run-to-gap adjacency for first-occurrence C runs in FR3-atyp chains, and for C runs in typical chains (location)
Cf = C.set_index(['pdb', 'chain'])
RFc = RF[RF['mode'] == 'C_FR3'].copy()
adj = []
for r in RFc.itertuples():
    if (r.pdb, r.chain) not in Cf.index: continue
    c = Cf.loc[(r.pdb, r.chain)]; c = c.iloc[0] if isinstance(c, pd.DataFrame) else c
    gp = [int(str(g).rstrip('ABCDEFGH')) for g in (c.fr3_gaps or ()) if str(g) != '73']
    d = min([max(0, r.imgt0 - g, g - r.imgt1) for g in gp], default=np.nan)
    adj.append(dict(pdb=r.pdb, chain=r.chain, method=r.method, year=r.year, imgt0=r.imgt0, imgt1=r.imgt1, delta=r.delta,
                    fr3_atyp=c.fr3_atyp, gaps=c.fr3_gaps, dist_run_to_gap=d))
A = pd.DataFrame(adj); A.to_csv(f'{OUT}/tables/v02_run_gap_adjacency.csv', index=False)
print(A[A.fr3_atyp == 1].to_string())
