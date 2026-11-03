"""Independent recount of the pdb_meta 'amyloid_artifact' finding from the shared data.
Re-derives every number in the finding without using the investigator's tables."""
import re, numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_amyloid_artifact_integrity'
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80); pd.set_option('display.max_rows', 300)

ent = pd.read_csv(f'{SP}/entries.csv')
ch = pd.read_csv(f'{SP}/chains.csv.gz')
runs = pd.read_csv(f'{SP}/runs_first_occurrence.csv')

# ---- my own amyloid definition: broader title search, then hand check
pat_broad = r'(?i)amyloid|fibril|amyloidosis'
cand = ent[ent.title.fillna('').str.contains(pat_broad)]
print('broad title candidates:'); print(cand[['pdb', 'method', 'res', 'err_first', 'err_any', 'n_V_chains', 'title']].to_string())
inv_pat = r'amyloid(?!-beta)'
inv = set(ent[ent.title.fillna('').str.contains(inv_pat, case=False)].pdb)
AMY = {'6ic3', '6z1o', '7nsl', '9els', '9faa', '9fab', '9fac', '9liv', '9liw', '9lix', '9liy', '9lj0', '9lja', '9ljb', '9v91'}
print('investigator regex set == claimed list:', inv == AMY, sorted(inv ^ AMY))

ent['amy'] = ent.pdb.isin(AMY); ch['amy'] = ch.pdb.isin(AMY); runs['amy'] = runs.pdb.isin(AMY)
em = ent[ent.method == 'EM']
print('\nEM error entries:', em.err_first.sum(), ' amyloid among them:', (em.err_first & em.amy).sum())
print('clean entries (all methods):', (~ent.err_any).sum(), ' amyloid clean:', ((~ent.err_any) & ent.amy).sum())
print('amyloid entries err_first/err_any:', ent[ent.amy][['err_first', 'err_any']].value_counts().to_dict())

# ---- runs
print('\nfirst-occurrence runs:', len(runs), ' amyloid:', runs.amy.sum(), f'({runs.amy.mean()*100:.1f}%)')
print(runs[runs.amy].groupby('pdb').agg(n=('chain', 'size'), chains=('chain', 'nunique'), modes=('mode', lambda s: ','.join(sorted(set(s)))),
      imgt0=('imgt0', 'min'), imgt1=('imgt1', 'max'), maxlen=('length', 'max'), deltas=('delta', lambda s: ','.join(map(str, sorted(set(s.astype(int))))))).to_string())
print('run length, amyloid vs other: median', runs[runs.amy].length.median(), runs[~runs.amy].length.median())

# ---- units
eu = ch[ch.err_first].drop_duplicates('unit')
print('\nerror units all methods:', eu.unit.nunique(), ' EM-only error units:', ch[ch.err_first & (ch.method == 'EM')].unit.nunique())
print('amyloid error units:', eu[eu.amy].unit.nunique())
# lambda: ctype L
lam = ch[ch.err_first & (ch.ctype == 'L')]
print('lambda error units (all methods):', lam.unit.nunique(), ' amyloid:', lam[lam.amy].unit.nunique(),
      ' | EM only:', lam[lam.method == 'EM'].unit.nunique())
# mode A units
def has_mode(s, m):
    return s.fillna('').str.split('|').apply(lambda L: any(x.strip() == m for x in re.split(r'[|;,]', '|'.join(L))))
ma = ch[ch.err_first & ch.err_modes.fillna('').str.contains('A_FR1Nterm')]
print('mode-A error units:', ma.unit.nunique(), ' amyloid:', ma[ma.amy].unit.nunique(), ' | EM:', ma[ma.method == 'EM'].unit.nunique())

# all chains in amyloid entries flagged?
a = ch[ch.amy]
print('\namyloid V chains:', len(a), ' err_first:', a.err_first.sum(), ' ctype:', a.ctype.value_counts().to_dict())
print(a.groupby('pdb').agg(n=('chain', 'size'), err=('err_first', 'sum'), v=('v_gene', 'first'), L1=('L1', 'first'),
      L3=('L3', 'first'), vlen=('vseq', lambda s: s.str.len().median()), seqlen=('seq', lambda s: s.str.len().median()),
      modes=('err_modes', lambda s: ','.join(sorted(set(s.fillna('')))))).to_string())
print('amyloid V genes:', a.drop_duplicates('unit').v_gene.value_counts().to_dict())
print('amyloid L1:', a.drop_duplicates('unit').L1.value_counts().to_dict())
