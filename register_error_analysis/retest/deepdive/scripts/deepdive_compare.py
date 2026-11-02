"""Compare first-occurrence error units with correctly modelled units at the same location (cryo-EM only).
Reads the unit table written by deepdive.py. For each mode: every feature with case vs control summary, p (Mann-Whitney
or Fisher) and BH q within the mode; residue enrichment at each IMGT position of the window (within chain type);
the list of member structures.
"""
import numpy as np, pandas as pd, warnings
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
O = '/home/user/SAbR/register_error_analysis/retest/deepdive/tables'
D = pd.read_pickle(f'{SP}/unexpl/deepdive_units_full.pkl')
log = open(f'{O}/deepdive_compare.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 50); pd.set_option('display.max_rows', 200)
D['irm_AF'] = D.irm.eq('AlphaFold model'); D['irm_PDB'] = D.irm.eq('experimental (PDB) model'); D['irm_none'] = D.irm.eq('none recorded')
D['fit_rigid'] = D.fit.eq('rigid body only'); D['fit_flex'] = D.fit.eq('flexible fit'); D['fit_abinitio'] = D.fit.eq('ab initio')
D['mouse'] = D.species.eq('mouse'); D['lambda'] = D.ctype.eq('L'); D['kappa'] = D.ctype.eq('K')
D['role_subject'] = D.role.astype(str).str.contains('alone|subject|main', case=False)
NUM = ['res', 'lq_z', 'year', 'v_id', 'L1', 'L2', 'L3', 'nbr_sim', 'nbr2_sim', 'landmarks', 'gly', 'pro', 'ser_thr', 'first_imgt', 'unmod_domain',
       'unmod_near', 'win_fit_pct', 'win_fit']
BIN = ['nterm_missing', 'fv_only', 'vhh', 'sars2', 'aux', 'sw_Coot', 'sw_ISOLDE', 'sw_Rosetta', 'sw_Phenix', 'L1_atyp', 'L2_atyp', 'L3_atyp',
       'irm_AF', 'irm_PDB', 'irm_none', 'fit_rigid', 'fit_flex', 'fit_abinitio', 'mouse', 'kappa', 'lambda']
allrows = []
for m in ['A', 'F', 'B', 'C', 'D+', 'D-', 'E']:
    d = D[(D['mode'] == m) & (D.method == 'EM')].copy()
    c, k = d[d.case], d[~d.case]
    P(f'\n######## mode {m}: EM case units {len(c)}, EM control units {len(k)}  (all methods: {int(D[(D["mode"]==m)].case.sum())} cases)')
    rows = []
    for f in NUM:
        a, b = c[f].dropna().astype(float), k[f].dropna().astype(float)
        if len(a) < 5 or len(b) < 20: continue
        p = mannwhitneyu(a, b).pvalue
        rows.append(dict(feature=f, case=round(a.median(), 3), control=round(b.median(), 3), case_mean=round(a.mean(), 3), control_mean=round(b.mean(), 3), p=p, kind='median'))
    for f in BIN:
        a, b = c[f].dropna().astype(bool), k[f].dropna().astype(bool)
        if len(a) < 5: continue
        p = fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1]
        rows.append(dict(feature=f, case=round(100 * a.mean(), 1), control=round(100 * b.mean(), 1), p=p, kind='%'))
    for f in ['j_gene', 'vfam', 'ctype']:
        for lev, n in c[f].value_counts().items():
            if n < 3: continue
            a = c[f].eq(lev); b = k[f].eq(lev)
            p = fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1]
            rows.append(dict(feature=f'{f}={lev}', case=round(100 * a.mean(), 1), control=round(100 * b.mean(), 1), p=p, kind='%'))
    T = pd.DataFrame(rows); T['q'] = multipletests(T.p, method='fdr_bh')[1]; T = T.sort_values('p')
    T['mode'] = m; allrows.append(T)
    P(T[['feature', 'kind', 'case', 'control', 'p', 'q']].head(18).to_string(index=False))
    # residue enrichment per IMGT position, within chain type
    rr = []
    for ct, g in d.groupby('ctype'):
        if g.case.sum() < 5: continue
        pa = pd.DataFrame([x if isinstance(x, dict) else {} for x in g.pos_aa], index=g.index)
        for pos in pa.columns:
            col = pa[pos]
            for aa in col.dropna().unique():
                a = col[g.case].eq(aa); b = col[~g.case].eq(aa)
                if a.sum() < 3: continue
                p = fisher_exact([[a.sum(), g.case.sum() - a.sum()], [b.sum(), (~g.case).sum() - b.sum()]])[1]
                rr.append(dict(ctype=ct, imgt=pos, aa=aa, case_pct=round(100 * a.mean(), 1), control_pct=round(100 * b.mean(), 1), p=p))
    if rr:
        RR = pd.DataFrame(rr); RR['q'] = multipletests(RR.p, method='fdr_bh')[1]; RR = RR.sort_values('p')
        RR.to_csv(f'{O}/deepdive_{m}_residues.csv', index=False)
        P('top residue enrichments:'); P(RR.head(8).to_string(index=False))
    mem = c[['pdb', 'chain', 'ctype', 'year', 'res', 'last_author', 'paper', 'win_seq', 'irm', 'fit', 'first_imgt', 'unmod_near', 'win_fit_pct', 'fv_only']]
    mem.assign(mode=m).to_csv(f'{O}/deepdive_{m}_members.csv', index=False)
pd.concat(allrows).to_csv(f'{O}/deepdive_features.csv', index=False)
log.close()
