"""T6 VH error predicts VL error in the same entry (case_files_DE DE3; full set: VH CDR3/FR4 slip -> VL first-occurrence
error 34% vs 1.3%, Fab-unit/paper OR 15-25; any VH error -> any VL error OR 13-18).
Universe: entries released in the paper window with >=1 scanned VH (non-VHH) and >=1 scanned VL (kappa/lambda) chain,
amyloid excluded; EM and X-ray separately.
Exposure: a VH chain of the entry carries a paper-confirmed error (strict; or incl copies); D/E-only version.
Reference: all other window Fab entries (as in DE3), and, as sensitivity, only entries whose VH chains have no run.
Outcomes (VL chains of the same entry):
  vl_paper   VL carries a paper-confirmed error (strict / incl copies) -> both chains confirmed by conkit AND SAbR/ANARCI
  vl_first   VL carries any deeper first-occurrence error (our detector, not filtered by conkit)
Units: entry; Fab-in-paper unit (paper x VH seq x VL seq, any entry); paper.
Heterogeneity: the same entry-level OR for the rest of the deeper set (VH rest_case error -> VL err_first; all years,
entries without any paper-confirmed chain), z-test of log ORs (Haldane).
Output: tables/vh_vl_tests.csv"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/model_fingerprint/scripts')
from common import TAB, load_frame, fisher, logit, het_z, mde_or_binary, ybin_win, ybin

D = load_frame(); D = D[~D.amyloid].copy()
PAPER_S = ['paper_strict']; PAPER_I = ['paper_strict', 'paper_loose', 'paper_copy']
D['isH'] = (D.ctype == 'H') & (D.vhh != 1); D['isL'] = D.ctype.isin(['K', 'L'])
D['DE'] = D.paper_modes.fillna('').str.contains('Dplus|Dminus|E_FR4')


def entry_table(sub):
    rows = []
    for p, g in sub.groupby('pdb'):
        H = g[g.isH]; L = g[g.isL]
        if not len(H) or not len(L): continue
        rows.append(dict(pdb=p, method=g.method.iloc[0], res=g.res.iloc[0], year=g.year.iloc[0], cluster=g.cluster.iloc[0],
                         window=bool(g.in_paper_window.iloc[0]), flagged=bool(g.paper_flagged_entry.max()),
                         H_strict=H.grp.isin(PAPER_S).any(), H_incl=H.grp.isin(PAPER_I).any(),
                         H_strict_DE=(H.grp.isin(PAPER_S) & H.DE).any(), H_rest=(H.grp == 'rest_case').any(),
                         H_any_run=H.err_any.any(), H_first=H.err_first.any(),
                         L_strict=L.grp.isin(PAPER_S).any(), L_incl=L.grp.isin(PAPER_I).any(), L_first=L.err_first.any(),
                         L_any_paper=L.grp.isin(PAPER_I).any(), any_paper=g.grp.isin(PAPER_I).any(),
                         fab_units=';'.join(sorted({f'{g.paper.iloc[0]}|{h}|{l}' for h in H.vseq.dropna().unique() for l in L.vseq.dropna().unique()})) if H.vseq.notna().any() else p))
    return pd.DataFrame(rows)


E = entry_table(D)
E['ybin_w'] = ybin_win(E.year.values); E['ybin'] = ybin(E.year.values)
res = []


def run(lab, Ew, expo, outc, ref_mask=None, meth='all'):
    x = Ew if meth == 'all' else Ew[Ew.method == meth]
    if ref_mask is not None:
        x = x[x[expo] | ref_mask.reindex(x.index)]
    a = int((x[expo] & x[outc]).sum()); b = int((x[expo] & ~x[outc]).sum()); c = int((~x[expo] & x[outc]).sum()); d = int((~x[expo] & ~x[outc]).sum())
    o = dict(test=lab, method=meth, exposure=expo, outcome=outc, exposed=f'{a}/{a + b}', unexposed=f'{c}/{c + d}',
             exp_rate=a / max(a + b, 1), unexp_rate=c / max(c + d, 1))
    f = fisher(a, b, c, d); o.update(entry_OR=f['OR'], entry_lo=f['lo'], entry_hi=f['hi'], entry_p=f['p'], entry_se=f['se'])
    o['mde_OR_entries'] = mde_or_binary(a + b, c + d, c / max(c + d, 1), +1)
    if min(a, b, c) > 0 and x[x[expo]].cluster.nunique() >= 3:
        try:
            xx = x.assign(yy=x[outc].astype(int), ee=x[expo].astype(int))
            r = logit(xx, 'yy', 'ee', ['C(ybin_w)', 'res'] + (['C(method)'] if meth == 'all' else []), 'cluster')
            if r['se'] < 8: o.update(adj_OR=r['OR'], adj_lo=r['lo'], adj_hi=r['hi'], adj_p=r['p'])
        except Exception:
            pass
    # Fab-in-paper units
    u = x.assign(fu=x.fab_units.str.split(';')).explode('fu').groupby('fu').agg(e=(expo, 'max'), y=(outc, 'max'), cl=('cluster', 'first'))
    a2 = int((u.e & u.y).sum()); b2 = int((u.e & ~u.y).sum()); c2 = int((~u.e & u.y).sum()); d2 = int((~u.e & ~u.y).sum())
    f2 = fisher(a2, b2, c2, d2)
    o.update(unit_exposed=f'{a2}/{a2 + b2}', unit_unexposed=f'{c2}/{c2 + d2}', unit_OR=f2['OR'], unit_lo=f2['lo'], unit_hi=f2['hi'], unit_p=f2['p'])
    # paper level: exposed paper = any exposed entry; outcome among the exposed entries of exposed papers / any entry otherwise
    pe = x.groupby('cluster').apply(lambda g: pd.Series({'e': g[expo].any(), 'y': (g[g[expo]][outc].any() if g[expo].any() else g[outc].any())}))
    a3 = int((pe.e & pe.y).sum()); b3 = int((pe.e & ~pe.y).sum()); c3 = int((~pe.e & pe.y).sum()); d3 = int((~pe.e & ~pe.y).sum())
    f3 = fisher(a3, b3, c3, d3)
    o.update(paper_exposed=f'{a3}/{a3 + b3}', paper_unexposed=f'{c3}/{c3 + d3}', paper_OR=f3['OR'], paper_lo=f3['lo'], paper_hi=f3['hi'], paper_p=f3['p'],
             mde_OR_papers=mde_or_binary(a3 + b3, c3 + d3, c3 / max(c3 + d3, 1), +1))
    res.append(o); return o


W = E[E.window].copy()
for meth in ['EM', 'X-ray', 'all']:
    run('T6 VH paper error -> VL paper error (strict)', W, 'H_strict', 'L_strict', meth=meth)
    run('T6 VH paper error -> VL any first-occurrence error', W, 'H_strict', 'L_first', meth=meth)
    run('T6 VH paper error -> VL any first-occurrence error; reference = VH without any run', W, 'H_strict', 'L_first', ref_mask=~W.H_any_run, meth=meth)
    run('T6 VH paper error (incl copies) -> VL paper error (incl copies)', W, 'H_incl', 'L_incl', meth=meth)
    run('T6 VH paper error (incl copies) -> VL any first-occurrence error', W, 'H_incl', 'L_first', meth=meth)
    run('T6 VH paper CDR3/FR4 (D/E) error -> VL any first-occurrence error', W, 'H_strict_DE', 'L_first', meth=meth)
    # full-set style reference on the rest of the deeper set (entries without any paper-confirmed chain), all years
    Rr = E[~E.any_paper]
    o = run('T6 [rest of deeper set, all years] VH rest first-occurrence error -> VL first-occurrence error', Rr, 'H_rest', 'L_first', meth=meth)
    ps = [r for r in res if r['test'] == 'T6 VH paper error -> VL any first-occurrence error' and r['method'] == meth][0]
    o['het_p_vs_paper'] = het_z(np.log(ps['entry_OR']), ps['entry_se'], np.log(o['entry_OR']), o['entry_se'])
    o2 = run('T6 [rest of deeper set, in window] VH rest first-occurrence error -> VL first-occurrence error', Rr[Rr.window], 'H_rest', 'L_first', meth=meth)
    o2['het_p_vs_paper'] = het_z(np.log(ps['entry_OR']), ps['entry_se'], np.log(o2['entry_OR']), o2['entry_se'])
R = pd.DataFrame(res); R.to_csv(f'{TAB}/vh_vl_tests.csv', index=False)
E.to_csv(f'{TAB}/vh_vl_entries.csv.gz', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 75)
f = lambda v: f'{v:.3g}'
for _, r in R.iterrows():
    s = f"{r.test[:85]:85s} {r.method:5s} | entry {r.exposed} vs {r.unexposed} OR {f(r.entry_OR)} ({f(r.entry_lo)}-{f(r.entry_hi)}) p={f(r.entry_p)}"
    if pd.notna(r.get('adj_OR')): s += f" adj {f(r.adj_OR)} ({f(r.adj_lo)}-{f(r.adj_hi)}) p={f(r.adj_p)}"
    s += f" | unit {r.unit_exposed} vs {r.unit_unexposed} OR {f(r.unit_OR)} p={f(r.unit_p)} | paper {r.paper_exposed} vs {r.paper_unexposed} OR {f(r.paper_OR)} ({f(r.paper_lo)}-{f(r.paper_hi)}) p={f(r.paper_p)}"
    if pd.notna(r.get('het_p_vs_paper')): s += f" | het p={f(r.het_p_vs_paper)}"
    print(s)
