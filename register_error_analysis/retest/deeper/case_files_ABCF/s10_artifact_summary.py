"""Which first-occurrence runs are convention artifacts of rare indels (confirmed on <2.6 A X-ray structures, s09)?

Confirmed artifact patterns (ANARCI-absent IMGT positions in the chain, beyond the chain-type norm) and the window
in which SAbR and ANARCI then disagree even at high resolution:
  heavy FR3 gap at 84            -> IMGT 81-83, shift +1   (4/4 structures at 1.38-2.0 A)
  heavy FR3 gap at 80 and 84     -> IMGT 81-83             (1/1, 2.3 A)
  heavy FR3 gap at 84 and 85     -> IMGT 81-83, shift +2   (3/3, 1.5-2.5 A)
  heavy FR1 gap at 26            -> IMGT 26-35             (1/1, 2.49 A)
Not artifacts: heavy FR3 gap at 81 or 82 (0/5 disagree at high resolution); heavy FR3 gap at 96 (7te4 at 2.46 A agrees
with ANARCI, so the 93-95 runs of its EM relatives 7te9/7teb are genuine deviations).
A run is "artifact-explained" if its chain carries a confirmed pattern and the run lies within the window +-2.
Outputs: tables/artifact_runs.csv; re-test of atypical FR3 length -> C_FR3 after removing artifact-explained units.
"""
import pickle, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF'
AN = pickle.load(open(f'{SP}/deeper/cache/seq_novelty/anarci_num.pkl', 'rb'))
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
amy = set(pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/amyloid_fibril_chains.csv').pdb.str.lower())
RF = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv'); RF = RF[~RF.pdb.isin(amy)]
RA = pd.read_csv(f'{SP}/deeper/runs_all.csv'); RA = RA[~RA.pdb.isin(amy)]
Cc = C.set_index(['pdb', 'chain'])
def missing(seq, light):
    for d in AN.get(seq) or []:
        if (d[0] in 'KL') == light:
            pos = {int(p) for p, i, a in d[4] if str(i).strip() == ''}
            return set(range(min(pos), 105)) - pos
    return None
PAT = [('FR3 gap 84', {84}, (79, 85)), ('FR3 gap 80+84', {80, 84}, (79, 85)), ('FR3 gap 84+85', {84, 85}, (79, 86)), ('FR1 gap 26', {26}, (24, 37))]
def classify(pdb, ch, ct, i0, i1):
    if ct != 'H' or (pdb, ch) not in Cc.index: return None
    r = Cc.loc[(pdb, ch)]
    r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
    m = missing(r.seq, False) or missing(r.vseq, False)
    if m is None: return None
    fr3 = {k for k in m if 74 <= k <= 100} ; fr1 = {k for k in m if 11 <= k <= 26 and k != 10}
    for name, gaps, (lo, hi) in PAT:
        cur = fr1 if name.startswith('FR1') else fr3
        if cur == gaps and lo <= i0 and i1 <= hi: return name
    return None
for D, tag in [(RF, 'first'), (RA, 'all')]:
    D['artifact'] = [classify(p, c, t, a, b) for p, c, t, a, b in zip(D.pdb, D.chain, D.ctype, D.imgt0, D.imgt1)]
RF['unit'] = RF.paper.astype(str) + '|' + RF.seqkey.astype(str)
A = RF[RF.artifact.notna()]
A[['pdb', 'chain', 'mode', 'imgt0', 'imgt1', 'delta', 'method', 'year', 'last_author', 'artifact', 'origin']].to_csv(f'{OUT}/tables/artifact_runs.csv', index=False)
pd.set_option('display.width', 250)
print(A.drop_duplicates('unit')[['pdb', 'chain', 'mode', 'imgt0', 'imgt1', 'delta', 'method', 'year', 'last_author', 'artifact']].to_string())
s = RF.groupby('mode').agg(units=('unit', 'nunique'), runs=('unit', 'size'))
s['artifact_units'] = A.groupby('mode').unit.nunique(); s['artifact_runs'] = A.groupby('mode').size()
s['artifact_units_EM'] = A[A.method == 'EM'].groupby('mode').unit.nunique()
s['units_EM'] = RF[RF.method == 'EM'].groupby('mode').unit.nunique()
print(s.fillna(0).astype(int))
print('all runs incl. copied/artifact: artifact-explained runs', RA.artifact.notna().sum(), 'of', len(RA), RA[RA.artifact.notna()].groupby(['mode', 'origin']).size().to_string())
s.to_csv(f'{OUT}/tables/artifact_share_by_mode.csv')
# germline of the artifact chains
g = C.merge(A[['pdb', 'chain', 'artifact']].drop_duplicates(), on=['pdb', 'chain'])
print(g.groupby('artifact').v_gene.agg(lambda x: x.value_counts().to_dict()))
