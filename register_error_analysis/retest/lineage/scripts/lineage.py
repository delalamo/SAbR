"""Do the non-inherited register errors propagate through *related* (not identical) antibodies?

Data: the 15,865 single-domain antibody V chains of the 3-5 A SAbR-vs-ANARCI scan (through 2026; ../unexplained) and
their 666 disagreement runs, classified by origin in ../unexplained (runs_origin_v2.csv).

Error classes and the germline gene that would carry a lineage:
  junction  CDR3-FR4 junction (origin 5; Trp/Phe118 anchor displaced)      -> J gene
  bulge     FR4 bulge after the W/F-G motif, IMGT 120 (origin 6)           -> J gene
  cdr12     CDR1/CDR2 boundary, typical or atypical length (origins 7, 8)  -> V gene
  gap       chain end / next to an unbuilt loop (origin 3)                 -> V gene (control class)

Unit = antibody in a paper (paper x V-region sequence), so symmetry copies and the same model reused across a paper's
entries count once. Y(u, class) = any chain of the unit has a run of that class.

Lineage predecessor of unit u for a class: an erroneous unit of the same class, of a *different* antibody (different
V-region sequence), released before u was deposited (within 3 years), in one of these relationship tiers:
  close     >= 90% V-region identity (variants, humanised versions, clonal relatives)
  vj        same species, chain type, V gene and J gene, < 90% identity
  germline  same species, chain type and the class's germline gene (J or V) only
and the same signature where defined (junction/bulge: same shift sign; bulge also same IMGT start +-1).
Each tier is split into same lab (last author) and other lab.

Comparison: P(predecessor | unit erroneous) vs P(predecessor | unit clean), Mantel-Haenszel OR stratified by
germline gene x method (EM/X-ray) x period, so a germline that is simply error-prone (intrinsic susceptibility) does not
count as propagation. A placebo predecessor (erroneous unit of the same class but a *different* germline gene, same
chain type) estimates what general clustering in time/lab alone produces.
Outputs: ../tables/lineage_*.csv and lineage.log
"""
import json, re, os, numpy as np, pandas as pd
from Bio import Align
from statsmodels.stats.contingency_tables import StratifiedTable
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
O = '/home/user/SAbR/register_error_analysis/retest/lineage/tables'; os.makedirs(O, exist_ok=True)
log = open(f'{O}/lineage.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 40)

M = pd.read_pickle(f'{SP}/unexpl/master4.pkl'); G = json.load(open(f'{SP}/unexpl/germline.json'))
M['j_gene'] = M.seq.map(lambda s: (G.get(s) or {}).get('j_gene'))
gene = lambda g: g.split('*')[0] if isinstance(g, str) else None
M['V'] = M.species.fillna('?') + '|' + M.ctype + '|' + M.v_gene.map(gene).fillna('?')
M['J'] = M.species.fillna('?') + '|' + M.ctype + '|' + M.j_gene.map(gene).fillna('?')
M['lab'] = M.last_author.fillna('').str.lower().str.replace('[^a-z]', '', regex=True)
M['paper'] = M.cite_key.fillna(M.pdb)
M['dep'] = pd.to_datetime(M.deposit); M['rel'] = pd.to_datetime(M.release)
M['EM'] = M.method.eq('EM'); M['period'] = pd.cut(M.year, [0, 2017, 2019, 2021, 2030], labels=['<=2017', '2018-19', '2020-21', '>=2022']).astype(str)

R = pd.read_csv(f'{SP}/unexpl/runs_origin_v2.csv')
cls = {'5': 'junction', '6': 'bulge', '7': 'cdr12', '8': 'cdr12', '3': 'gap'}
R['cls'] = R.origin.str.split(' ').str[0].map(cls)
R = R[R.cls.notna()]
P('runs by class:', R.cls.value_counts().to_dict())

# units
U = M.groupby(['paper', 'seq']).agg(pdbs=('pdb', lambda x: sorted(set(x))), chains=('chain', 'size'), dep=('dep', 'min'), rel=('rel', 'min'),
                                     lab=('lab', 'first'), V=('V', 'first'), J=('J', 'first'), ctype=('ctype', 'first'), EM=('EM', 'any'),
                                     period=('period', 'first'), year=('year', 'min'), res=('res', 'min'), cdr3_len=('cdr3_len', 'first')).reset_index()
key = dict(zip(zip(M.pdb, M.chain), zip(M.paper, M.seq)))
R['unit'] = [key.get(k) for k in zip(R.pdb, R.chain)]
R = R[R.unit.notna()]
U['uid'] = list(zip(U.paper, U.seq)); uidx = {u: i for i, u in enumerate(U.uid)}
for c in ['junction', 'bulge', 'cdr12', 'gap']:
    sub = R[R.cls == c]
    U[f'y_{c}'] = False; U.loc[[uidx[u] for u in sub.unit], f'y_{c}'] = True
    sig = sub.groupby('unit').agg(imgt0=('imgt0', 'min'), sgn=('delta', lambda d: int(np.sign(np.nanmedian(d))) if d.notna().any() else 0))
    U[f'sig0_{c}'] = U.uid.map(sig.imgt0); U[f'sgn_{c}'] = U.uid.map(sig.sgn)
P('units (paper x antibody):', len(U), ' erroneous units:', {c: int(U[f'y_{c}'].sum()) for c in ['junction', 'bulge', 'cdr12', 'gap']})

# 1. germline concentration
P('\n== 1. Are these errors concentrated in particular germline genes? (unit level)')
rows = []
for c, gk in [('junction', 'J'), ('bulge', 'J'), ('cdr12', 'V'), ('gap', 'V')]:
    g = U.groupby(gk).agg(units=('uid', 'size'), err=(f'y_{c}', 'sum'), em=('EM', 'mean'))
    g = g[g.units >= 20]; p0 = g.err.sum() / g.units.sum()
    from scipy.stats import chi2_contingency
    chi = chi2_contingency(np.c_[g.err, g.units - g.err])[1] if (g.err > 0).sum() > 1 else np.nan
    top = g.assign(rate=100 * g.err / g.units).sort_values('err', ascending=False).head(5)
    P(f'\n{c} ({gk} gene): {int(g.err.sum())} erroneous units of {int(g.units.sum())} ({100*p0:.2f}%); heterogeneity p={chi:.2g}')
    P(top.round(2).to_string())
    for gg, r in top.iterrows(): rows.append(dict(cls=c, gene=gg, units=r.units, err=r.err, rate=round(r.rate, 2), em_share=round(r.em, 2)))
pd.DataFrame(rows).to_csv(f'{O}/lineage_germline_top.csv', index=False)

# 2. lineage predecessors
al = Align.PairwiseAligner(mode='global', match_score=1, mismatch_score=0, open_gap_score=-1, extend_gap_score=-0.5)
_id = {}
def ident(a, b):
    k = (a, b) if a < b else (b, a)
    if k not in _id:
        s = al.score(a, b); _id[k] = s / max(len(a), len(b))
    return _id[k]
WIN = pd.Timedelta(days=3 * 365)
out = []
for c, gk in [('junction', 'J'), ('bulge', 'J'), ('cdr12', 'V'), ('gap', 'V')]:
    E = U[U[f'y_{c}']]
    for i, u in U.iterrows():
        cand = E[(E.seq != u.seq) & (E.ctype == u.ctype) & (E.rel < u.dep) & (E.rel >= u.dep - WIN)]
        if c == 'bulge': cand = cand[(cand[f'sig0_{c}'] - 118).abs() <= 3]
        hit = {f'{t}_{l}': False for t in ['close', 'vj', 'germline', 'placebo'] for l in ['same', 'other']}
        for e in cand.itertuples():
            lab = 'same' if e.lab == u.lab else 'other'
            same_g = getattr(e, gk) == u[gk]
            if not same_g:
                hit[f'placebo_{lab}'] = True; continue
            if e.V == u.V and e.J == u.J:
                if ident(e.seq, u.seq) >= 0.90: hit[f'close_{lab}'] = True
                else: hit[f'vj_{lab}'] = True
            else:
                hit[f'germline_{lab}'] = True
        out.append(dict(cls=c, uid=i, y=bool(u[f'y_{c}']), stratum=f'{u[gk]}|{u.EM}|{u.period}', **hit))
    P(f'computed predecessors for {c}')
L = pd.DataFrame(out); L.to_pickle(f'{SP}/unexpl/lineage_units.pkl')

P('\n== 2. Erroneous units vs clean units: share with an erroneous predecessor of the same class (previous 3 years)')
P('   MH OR stratified by germline gene x EM x period; tiers are cumulative within lab type (close, or same V+J, or same germline gene)')
res = []
for c in ['junction', 'bulge', 'cdr12', 'gap']:
    d = L[L.cls == c].copy()
    for lab in ['other', 'same']:
        d['close'] = d[f'close_{lab}']; d['vjplus'] = d.close | d[f'vj_{lab}']; d['gplus'] = d.vjplus | d[f'germline_{lab}']; d['plac'] = d[f'placebo_{lab}']
        for tier, col in [('close relative (>=90%)', 'close'), ('same V+J gene', 'vjplus'), ('same germline gene', 'gplus'), ('placebo: other germline', 'plac')]:
            tabs = []
            for s, g in d.groupby('stratum'):
                t = np.array([[((g.y) & g[col]).sum(), ((g.y) & ~g[col]).sum()], [((~g.y) & g[col]).sum(), ((~g.y) & ~g[col]).sum()]])
                if t[0].sum() and t[1].sum(): tabs.append(t)
            orr = lo = hi = p = np.nan
            if tabs:
                st = StratifiedTable(tabs)
                try:
                    orr = st.oddsratio_pooled; lo, hi = st.oddsratio_pooled_confint(); p = st.test_null_odds().pvalue
                except Exception: pass
            res.append(dict(cls=c, lab=lab, tier=tier, err_units=int(d.y.sum()), err_with_pred=int((d.y & d[col]).sum()),
                            pct_err=round(100 * (d.y & d[col]).sum() / max(1, d.y.sum()), 1),
                            pct_clean=round(100 * ((~d.y) & d[col]).sum() / max(1, (~d.y).sum()), 1),
                            MH_OR=round(orr, 2), lo=round(lo, 2), hi=round(hi, 2), p=p))
T = pd.DataFrame(res); P(T.to_string(index=False)); T.to_csv(f'{O}/lineage_predecessors.csv', index=False)
log.close()
