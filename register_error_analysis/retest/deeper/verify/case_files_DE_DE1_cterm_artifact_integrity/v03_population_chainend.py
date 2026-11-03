"""V03: population view. For every scanned single-domain V chain in chains.csv, the last modelled ANARCI IMGT position
(from the scan records) vs whether the chain carries a first-occurrence D+ run (and any D+ run incl. copied/artifact).
If SAbR manufactures D+ calls at chain ends, chains whose model stops at IMGT <=121 should carry D+ calls far more
often, in X-ray as well as EM, and the calls should end exactly at the last residue.
Output: tables/v03_chainend_rates.csv, tables/v03_short_chain_dplus.csv
"""
import json, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D = f'{SP}/deeper'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE1_cterm_artifact_integrity'
C = pd.read_csv(f'{D}/chains.csv.gz')
R = pd.read_csv(f'{D}/runs_first_occurrence.csv'); RA = pd.read_csv(f'{D}/runs_all.csv')
keys = set(zip(C.pdb, C.chain))
last, lastsab = {}, {}
for f in ['validation', 'snapshot_rest', 'new']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        r = json.loads(l)
        k = (r['pdb'], r['chain'])
        if k not in keys or r.get('status') != 'ok': continue
        v = [x for x in r['residues'] if x[4] is not None]
        if not v: continue
        last[k] = max(x[4] for x in v)
        lastsab[k] = v[-1][6]
C['last_imgt'] = [last.get(k) for k in zip(C.pdb, C.chain)]
C['last_sabr'] = [lastsab.get(k) for k in zip(C.pdb, C.chain)]
C = C[C.last_imgt.notna()].copy()
dp_first = set(zip(R[R['mode'] == 'Dplus_CDR3short'].pdb, R[R['mode'] == 'Dplus_CDR3short'].chain))
dp_all = set(zip(RA[RA['mode'] == 'Dplus_CDR3short'].pdb, RA[RA['mode'] == 'Dplus_CDR3short'].chain)) if 'mode' in RA else set()
C['dplus_first'] = [k in dp_first for k in zip(C.pdb, C.chain)]
C['dplus_any'] = [k in dp_all for k in zip(C.pdb, C.chain)]
C['end_cat'] = pd.cut(C.last_imgt, [0, 117, 118, 121, 124, 129], labels=['<=117', '118', '119-121', '122-124', '>=125'])
rows = []
for meth in ['EM', 'X-ray']:
    for cat, g in C[C.method == meth].groupby('end_cat', observed=True):
        rows.append(dict(method=meth, end=cat, chains=len(g), units=g.unit.nunique(), dplus_first_chains=int(g.dplus_first.sum()),
                         dplus_first_units=g[g.dplus_first].unit.nunique(), dplus_any_chains=int(g.dplus_any.sum()),
                         rate_first_units=g[g.dplus_first].unit.nunique() / g.unit.nunique()))
T = pd.DataFrame(rows)
# unit-level test within EM: unit short if any chain of the unit ends <=121; outcome: any chain with D+ first-occurrence
U = C[C.method == 'EM'].groupby('unit').agg(short=('last_imgt', lambda s: (s <= 121).any()), dplus=('dplus_first', 'any'),
                                           year=('year', 'min'), res=('res', 'mean'), ctype=('ctype', 'first'), paper=('paper', 'first')).reset_index()
a = int((U.short & U.dplus).sum()); b = int((U.short & ~U.dplus).sum()); c = int((~U.short & U.dplus).sum()); d = int((~U.short & ~U.dplus).sum())
orf, pf = fisher_exact([[a, b], [c, d]])
U['yb'] = pd.cut(U.year, [0, 2019, 2020, 2021, 2023, 2030]).astype(str)
try:
    m = smf.logit('dplus ~ short + C(yb) + res + C(ctype)', data=U.assign(short=U.short.astype(int), dplus=U.dplus.astype(int))).fit(
        disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(U.paper)[0]})
    adj = (float(np.exp(m.params['short'])), float(m.pvalues['short']))
except Exception as e:
    adj = (np.nan, np.nan)
T.to_csv(f'{OUT}/tables/v03_chainend_rates.csv', index=False)
pd.set_option('display.width', 250)
print(T.to_string())
print(f'EM units: short-ended with D+ {a}/{a+b} = {a/(a+b):.3f}; longer {c}/{c+d} = {c/(c+d):.4f}; Fisher OR {orf:.1f} p {pf:.2g}; adjusted OR {adj[0]:.1f} p {adj[1]:.2g}')
# D+ chains that end <=121: does the run end at the last modelled residue?
S = C[C.dplus_first & (C.last_imgt <= 121)]
rr = R[R['mode'] == 'Dplus_CDR3short'].groupby(['pdb', 'chain']).agg(imgt1=('imgt1', 'max'), delta=('delta', 'first')).reset_index()
S = S.merge(rr, on=['pdb', 'chain'])
S['run_ends_at_last'] = S.imgt1 >= S.last_imgt
S[['pdb', 'chain', 'unit', 'method', 'year', 'ctype', 'last_imgt', 'last_sabr', 'imgt1', 'delta', 'run_ends_at_last', 'paper']].to_csv(f'{OUT}/tables/v03_short_chain_dplus.csv', index=False)
print('D+ first-occurrence chains ending <=121:', len(S), 'units', S.unit.nunique(), 'run ends at the last modelled residue:', int(S.run_ends_at_last.sum()))
print('last SAbR number of those chains (>128 = numbering ran past FR4):', S.last_sabr.describe().round(1).to_dict())
