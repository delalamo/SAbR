"""Structural "ordinary antibody" tests on paper-confirmed case Fvs vs paper-window control Fvs (EM and X-ray):
 (a) deposited VH-VL orientation (ABangle): Mahalanobis distance from the X-ray reference (All_Angles.dat, as
     struct_unusual s09), outside the 99% ellipsoid, the six parameters, core-set RMSD, Mahalanobis in Fvs with
     undistorted cores (core RMSD < control 90th percentile, per method);
 (b) predicted (ImmuneBuilder, sequence-intrinsic) orientation: Mahalanobis, predicted core RMSD, |deposited-predicted|;
 (c) predicted CDR-H3 base kinked (VH and VHH separately), deposited H3 base kinked (context);
 (d) ImmuneBuilder ensemble error at the error site (controls: same IMGT positions, same chain class), whole chain, CDR3.
Unit: Fv (or VHH) instance; inference clustered by paper (cluster-robust), plus one vote per paper per group (PW).
Covariates: year bin (window), resolution, method (pooled), light-chain type.
Rest comparator (heterogeneity): struct_unusual (deeper, EM, 2018-2026) case Fvs/chains that are NOT paper-confirmed vs
struct_unusual controls, same model; z-test of the two estimates (EM only, the only method struct_unusual has).
Output: tables/struct_tests.csv, tables/struct_fv_frame.csv, tables/struct_chain_frame.csv"""
import sys, os, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/scripts')
sys.path.insert(0, os.path.dirname(__file__))
from common import abangle, dihedral, C as SUC  # noqa: E402
from aoutil import SP, TAB, DEEP, CACHE, logit_or, lin_diff, z_het, fisher, ybin_win, ybin_full, f_or, f_diff, mde_or  # noqa

A6 = ['HL', 'HC1', 'LC1', 'HC2', 'LC2', 'dc']
ref = pd.read_csv(f'{SUC}/abangle_src/ABangle/data/All_Angles.dat', sep='\t'); ref = ref[ref.Method == 'X-RAY']
mu = ref[A6].mean().values; ci = np.linalg.inv(np.cov(ref[A6].values.T))
THR = np.sqrt(stats.chi2.ppf(0.99, 6))
maha = lambda X: np.sqrt(np.einsum('ij,jk,ik->i', X - mu, ci, X - mu))
IB = f'{CACHE}/ib'
FWPOS = set(range(1, 27)) | set(range(39, 56)) | set(range(66, 105)) | set(range(118, 129))

F = pd.read_csv(f'{TAB}/fv_abangle_window.csv', dtype={'H': str, 'L': str})
F = F[(F.role == 'control') | F.incl].copy()
F['strict'] = F.strict.astype(bool); F['incl'] = F.incl.astype(bool)
F['yb'] = ybin_win(F.year)
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'chain', 'vseq', 'ctype'], dtype={'chain': str})
seq = dict(zip(zip(d.pdb, d.chain), d.vseq))
F['Hseq'] = [seq.get((p, h)) for p, h in zip(F.pdb, F.H)]
F['Lseq'] = [seq.get((p, l)) if isinstance(l, str) else '' for p, l in zip(F.pdb, F.L)]
jobs = pd.read_csv(f'{TAB}/ib_jobs.csv').fillna({'Lseq': ''})
job_by_seq = {(r.Hseq, r.Lseq): r.fv_id for r in jobs.itertuples() if r.grp == 'case'}
job_ctrl = {r.fv_id for r in jobs.itertuples() if r.grp == 'control'}


def pred_file(r):
    if r.role == 'case':
        fid = job_by_seq.get((r.Hseq, r.Lseq if r.kind == 'Fv' else ''))
    else:
        fid = f'c_{r.pdb}_{r.H}' + (f'_{r.L}' if r.kind == 'Fv' else '')
        fid = fid if fid in job_ctrl else None
    if fid and os.path.exists(f'{IB}/{fid}.pdb'):
        return f'{IB}/{fid}.pdb', f'{IB}/{fid}_err.npy'
    return None, None


def read_pred(fn):
    out, order = {}, []
    for line in open(fn):
        if line.startswith('ATOM'):
            ch = line[21]; key = (int(line[22:26]), line[26].strip()); an = line[12:16].strip()
            if (ch, key) not in out:
                out[(ch, key)] = {}; order.append((ch, key))
            out[(ch, key)][an] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return out, order


# ---- sites: paper-confirmed run of each case chain (IMGT window of the matched SAbR/ANARCI run)
E = pd.read_csv(f'{SP}/paperconf/paper_errors_annotated.csv', dtype={'chain': str})
E = E[E.confirmed & E.status.isin(['confirmed first occurrence', 'confirmed, loose-ledger doubt', 'copy of an earlier model', 'near copy (ledger)'])]
SITES = {k: g[['imgt0', 'imgt1']].drop_duplicates().values.tolist() for k, g in E.groupby(['pdb', 'chain'])}
STRICT_CH = set(zip(E[E.status == 'confirmed first occurrence'].pdb, E[E.status == 'confirmed first occurrence'].chain))

# ---- per-Fv prediction metrics and per-chain metrics
fv_rows, ch_rows, err_store = [], [], {}
for r in F.itertuples():
    o = dict(pdb=r.pdb, H=r.H, L=r.L)
    p, ep = pred_file(r)
    if p is None:
        fv_rows.append(o); continue
    pred, order = read_pred(p)
    err = np.load(ep) if os.path.exists(ep) else None
    e = dict(zip(order, err)) if err is not None else {}
    caH = {k[0]: v['CA'] for (c, k), v in pred.items() if c == 'H' and k[1] == '' and 'CA' in v}
    if r.kind == 'Fv':
        caL = {k[0]: v['CA'] for (c, k), v in pred.items() if c == 'L' and k[1] == '' and 'CA' in v}
        a = abangle(caH, caL, r.Lct)
        if a:
            o.update({f'pred_{k}': a[k] for k in A6}); o['pred_core_rms'] = (a['rmsH'] + a['rmsL']) / 2
    fv_rows.append(o)
    chains = [(r.H, 'H', 'H' if r.kind == 'Fv' else 'VHH')] + ([(r.L, 'L', r.Lct)] if r.kind == 'Fv' else [])
    for ch, pch, cls in chains:
        c = dict(pdb=r.pdb, chain=ch, cls=cls, role=r.role, kind=r.kind, method=r.method, res=r.res, year=r.year, paper=r.paper,
                 case_chain=(r.pdb, ch) in SITES, strict_chain=(r.pdb, ch) in STRICT_CH)
        if pch == 'H' and all(q in caH for q in (115, 116, 117, 118)):
            tau = dihedral(caH[115], caH[116], caH[117], caH[118]); c['pred_h3_tau'] = tau; c['pred_h3_kinked'] = float(-30 <= tau <= 100)
        if e:
            ev = {k: v for (cc, k), v in e.items() if cc == pch}
            c['pred_err_chain'] = float(np.mean(list(ev.values())))
            c['pred_err_cdr3'] = float(np.mean([v for k, v in ev.items() if 105 <= k[0] <= 117]))
            err_store[(r.pdb, ch)] = (cls, ev)
            if (r.pdb, ch) in SITES:
                sv = [v for k, v in ev.items() if any(a - 2 <= k[0] <= b + 2 for a, b in SITES[(r.pdb, ch)])]
                c['pred_err_site'] = float(np.mean(sv)) if sv else np.nan
        ch_rows.append(c)
PF = pd.DataFrame(fv_rows)
F = F.merge(PF, on=['pdb', 'H', 'L'], how='left')
CH = pd.DataFrame(ch_rows).drop_duplicates(['pdb', 'chain', 'role'])
# matched-site ensemble error for controls: mean over control chains of the same class at each case site
ctrl_err = {k: v for k, v in err_store.items() if k not in SITES}
site_rows = []
for (p, ch), sites in SITES.items():
    if (p, ch) not in err_store:
        continue
    cls, ev = err_store[(p, ch)]
    win = lambda k: any(a - 2 <= k[0] <= b + 2 for a, b in sites)
    cv = [np.mean([v for k, v in e2.items() if win(k)]) for (pp, cc), (cl2, e2) in ctrl_err.items() if cl2 == cls]
    cv = [x for x in cv if np.isfinite(x)]
    meta = CH[(CH.pdb == p) & (CH.chain == ch)].iloc[0]
    site_rows.append(dict(pdb=p, chain=ch, cls=cls, method=meta.method, paper=meta.paper, strict=(p, ch) in STRICT_CH,
                          site_err=np.mean([v for k, v in ev.items() if win(k)]), ctrl_site_err=np.mean(cv), n_ctrl=len(cv),
                          ctrl_site_sd=np.std(cv)))
SE = pd.DataFrame(site_rows)
if len(SE):
    SE['diff'] = SE.site_err - SE.ctrl_site_err

# ---- deposited orientation metrics
ok = F.kind.eq('Fv') & F[[f'{a}_sabr' for a in A6]].notna().all(1) & (F.cdist <= 28)
F['maha_dep'] = np.nan; F.loc[ok, 'maha_dep'] = maha(F.loc[ok, [f'{a}_sabr' for a in A6]].values)
F['outlier99'] = (F.maha_dep > THR).astype(float).where(F.maha_dep.notna())
F['core_rms'] = F[['rmsH_sabr', 'rmsL_sabr']].mean(1)
okp = F[[f'pred_{a}' for a in A6]].notna().all(1)
F['pred_maha'] = np.nan; F.loc[okp, 'pred_maha'] = maha(F.loc[okp, [f'pred_{a}' for a in A6]].values)
for a in ['LC2', 'dc', 'HL']:
    F[f'absd_{a}'] = (F[f'{a}_sabr'] - F[f'pred_{a}']).abs()
F['undistorted'] = np.nan
for m in ['EM', 'X-ray']:
    q90 = F[(F.role == 'control') & (F.method == m)].core_rms.quantile(0.9)
    F.loc[F.method == m, 'undistorted'] = (F.loc[F.method == m, 'core_rms'] < q90).astype(float)
F.to_csv(f'{TAB}/struct_fv_frame.csv', index=False); CH.to_csv(f'{TAB}/struct_chain_frame.csv', index=False)
SE.to_csv(f'{TAB}/struct_site_err.csv', index=False)

# ---- rest comparator from struct_unusual (deeper, EM): cases not paper-confirmed vs struct_unusual controls
SU = f'{DEEP}/struct_unusual/tables'
paper_keys = set(zip(E.pdb, E.chain))
fo = pd.read_csv(f'{SU}/fv_orientation.csv', dtype={'H': str, 'L': str})
fo['paper'] = [any((p, c) in paper_keys for c in (h, l)) for p, h, l in zip(fo.pdb, fo.H, fo.L)]
fo = fo[~fo.paper].copy(); fo['y'] = fo.fv_case.astype(int)
fo['core_rms'] = fo[['rmsH_sabr', 'rmsL_sabr']].mean(1)
pf = pd.read_csv(f'{SU}/pred_compare_fv.csv', dtype={'H': str, 'L': str})
okq = pf[[f'pred_{a}' for a in A6]].notna().all(1)
pf['pred_maha'] = np.nan; pf.loc[okq, 'pred_maha'] = maha(pf.loc[okq, [f'pred_{a}' for a in A6]].values)
fo = fo.merge(pf[['pdb', 'H', 'L', 'pred_maha', 'pred_core_rms', 'd_LC2', 'd_dc']], on=['pdb', 'H', 'L'], how='left')
fo['absd_LC2'] = fo.d_LC2.abs(); fo['absd_dc'] = fo.d_dc.abs()
smp = pd.read_csv(f'{SU}/sample.csv', dtype={'chain': str})
fo = fo.merge(smp.drop_duplicates('pdb')[['pdb', 'paper']].rename(columns={'paper': 'pp'}), on='pdb', how='left')
fo['pp'] = fo.pp.fillna(fo.pdb)
q90 = fo[fo.y == 0].core_rms.quantile(0.9); fo['undistorted'] = (fo.core_rms < q90).astype(float)
pc = pd.read_csv(f'{SU}/pred_compare_chain.csv', dtype={'chain': str})
pc = pc[~pd.Series([(p, c) in paper_keys for p, c in zip(pc.pdb, pc.chain)], index=pc.index)].copy()
pc = pc.merge(smp[['pdb', 'chain', 'ybin', 'res', 'paper']].drop_duplicates(['pdb', 'chain']), on=['pdb', 'chain'], how='left')
pc['y'] = pc.role_cc.eq('case').astype(int); pc['pp'] = pc.paper.fillna(pc.pdb)
pc['ybin'] = pc.ybin.fillna(pc.groupby('pdb').ybin.transform('first')); pc['res'] = pc.res.fillna(pc.groupby('pdb').res.transform('first'))

# ---- tests
rows = []


def add(trend, metric, kind, d, ycol, covs, rest=None, note=''):
    """kind 'cont': adjusted difference of metric (case - control); 'bin': OR of metric (outcome) for case vs control."""
    for meth in ['all', 'EM', 'X-ray']:
        for cs in ['strict', 'incl']:
            dd = d[(d.role == 'control') | d[cs]].copy()
            if meth != 'all':
                dd = dd[dd.method == meth]
            dd['y'] = (dd.role == 'case').astype(int)
            dd = dd.dropna(subset=[metric])
            n1, n0 = int(dd.y.sum()), int((dd.y == 0).sum())
            r = dict(trend=trend, metric=metric, method=meth, set=cs, n1=n1, n0=n0, papers1=dd[dd.y == 1].paper.nunique(),
                     v1=dd[dd.y == 1][metric].mean(), v0=dd[dd.y == 0][metric].mean(), note=note)
            if n1 < 3:
                rows.append(r); continue
            cv = [c for c in covs if not (meth != 'all' and c == 'C(method)')]
            dd['w'] = 1.0 / dd.groupby(['paper', 'y']).y.transform('size')
            if kind == 'cont':
                r['crude_p'] = stats.mannwhitneyu(dd[dd.y == 1][metric], dd[dd.y == 0][metric]).pvalue
                m = lin_diff(dd, metric, cv, y='y', groups='paper'); pw = lin_diff(dd, metric, cv, y='y', groups='paper', weights='w')
            else:
                k1, k0 = int(dd[dd.y == 1][metric].sum()), int(dd[dd.y == 0][metric].sum())
                r.update(k1=k1, k0=k0)
                fz = fisher(k1, n1, k0, n0); r.update({f'crude_{k}': v for k, v in fz.items()})
                r['mde_OR80'] = mde_or(n1, k0 / n0, n0) if 0 < k0 < n0 else np.nan
                r['mde_OR80_low'] = mde_or(n1, k0 / n0, n0, lower=True) if 0 < k0 < n0 else np.nan
                if k1 in (0, n1) or k0 in (0, n0):
                    rows.append(r); continue
                dd['m'] = dd[metric]
                m = logit_or(dd, 'y', cv, y='m', groups='paper'); pw = logit_or(dd, 'y', cv, y='m', groups='paper', weights='w')
            r.update({f'M_{k}': v for k, v in m.items() if k in ('OR', 'diff', 'lo', 'hi', 'p', 'b', 'se', 'err')})
            r.update({f'PW_{k}': v for k, v in pw.items() if k in ('OR', 'diff', 'lo', 'hi', 'p', 'b', 'se', 'err')})
            if rest is not None and meth == 'EM':
                rd, rcov = rest
                rd = rd.dropna(subset=[metric]).copy()
                if kind == 'cont':
                    rm = lin_diff(rd, metric, rcov, y='y', groups='pp')
                else:
                    rd['m'] = rd[metric]; rm = logit_or(rd, 'y', rcov, y='m', groups='pp')
                r.update({f'REST_{k}': v for k, v in rm.items() if k in ('OR', 'diff', 'lo', 'hi', 'p', 'b', 'se')})
                r['REST_n1'] = int(rd.y.sum()); r['REST_n0'] = int((rd.y == 0).sum())
                r['REST_v1'] = rd[rd.y == 1][metric].mean(); r['REST_v0'] = rd[rd.y == 0][metric].mean()
                r['het_p'] = z_het(m, rm) if 'b' in m and 'b' in rm else np.nan
            rows.append(r)


Fv = F[F.kind == 'Fv'].copy()
covF = ['C(yb)', 'res', 'C(method)', 'C(Lct)']
restF = (fo.rename(columns={'maha_sabr': 'maha_dep'}), ['C(ybin)', 'res', 'C(Lct)'])
add('VH-VL orientation (deposited)', 'maha_dep', 'cont', Fv, 'y', covF, restF)
add('VH-VL orientation (deposited)', 'outlier99', 'bin', Fv, 'y', covF, (restF[0].assign(outlier99=restF[0].outlier99.astype(float)), restF[1]))
for a in A6:
    add('VH-VL orientation (deposited), mean parameter', f'{a}_sabr', 'cont', Fv, 'y', covF, (fo, restF[1]))
add('VH-VL orientation (deposited)', 'core_rms', 'cont', Fv, 'y', covF, (fo, restF[1]))
add('VH-VL orientation: outliers explained by distorted cores', 'maha_dep', 'cont', Fv[Fv.undistorted == 1], 'y', covF,
    (restF[0][restF[0].undistorted == 1], restF[1]), note='undistorted cores only (core RMSD < control 90th pct)')
add('VH-VL orientation: outliers explained by distorted cores', 'outlier99', 'bin', Fv[Fv.undistorted == 1], 'y', covF,
    (restF[0][restF[0].undistorted == 1].assign(outlier99=lambda x: x.outlier99.astype(float)), restF[1]), note='undistorted cores only')
add('predicted orientation (sequence-intrinsic)', 'pred_maha', 'cont', Fv, 'y', covF, (fo, restF[1]))
add('predicted orientation (sequence-intrinsic)', 'pred_core_rms', 'cont', Fv, 'y', covF, (fo, restF[1]))
add('deposited minus predicted orientation', 'absd_LC2', 'cont', Fv, 'y', covF, (fo, restF[1]))
add('deposited minus predicted orientation', 'absd_dc', 'cont', Fv, 'y', covF, (fo, restF[1]))
# chain level: predicted / deposited H3 base, ensemble error
CH['yb'] = ybin_win(CH.year); CH['strict'] = CH.strict_chain | (CH.role == 'control') & False
CH['incl'] = CH.case_chain
CHc = CH[(CH.role == 'control') | CH.case_chain].copy()
restVH = (pc[pc.cclass == 'H'].assign(y=lambda x: x.y), ['C(ybin)', 'res'])
restVHH = (pc[pc.cclass == 'VHH'], ['C(ybin)', 'res'])
add('predicted CDR-H3 base kinked, VH', 'pred_h3_kinked', 'bin', CHc[CHc.cls == 'H'], 'y', ['C(yb)', 'res', 'C(method)'], restVH)
add('predicted CDR-H3 base kinked, VHH', 'pred_h3_kinked', 'bin', CHc[CHc.cls == 'VHH'], 'y', ['C(yb)', 'res', 'C(method)'], restVHH)
add('ImmuneBuilder ensemble error, whole chain', 'pred_err_chain', 'cont', CHc, 'y', ['C(yb)', 'res', 'C(method)', 'C(cls)'],
    (pc, ['C(ybin)', 'res', 'C(cclass)']))
add('ImmuneBuilder ensemble error, CDR3', 'pred_err_cdr3', 'cont', CHc, 'y', ['C(yb)', 'res', 'C(method)', 'C(cls)'],
    (pc, ['C(ybin)', 'res', 'C(cclass)']))
# deposited H3 base kinked (from s02, SAbR numbering) for context
dh = F[['pdb', 'H', 'h3_kinked', 'role', 'strict', 'incl', 'method', 'res', 'yb', 'paper', 'kind']].rename(columns={'H': 'chain'})
dh['cls'] = np.where(dh.kind == 'VHH', 'VHH', 'H')
dh['strict'] = [(p, c) in STRICT_CH for p, c in zip(dh.pdb, dh.chain)] ; dh['incl'] = [(p, c) in SITES for p, c in zip(dh.pdb, dh.chain)]
dh = dh[(dh.role == 'control') | dh.incl].drop_duplicates(['pdb', 'chain', 'role'])
add('deposited CDR-H3 base kinked, VH (context)', 'h3_kinked', 'bin', dh[dh.cls == 'H'], 'y', ['C(yb)', 'res', 'C(method)'])
T = pd.DataFrame(rows)
# matched-site ensemble error: one-sample test of case minus matched-control mean (cluster-robust by paper)
import statsmodels.formula.api as smf
for meth in ['all', 'EM', 'X-ray']:
    for cs in ['strict', 'incl']:
        s = SE if cs == 'incl' else SE[SE.strict]
        s = s if meth == 'all' else s[s.method == meth]
        s = s.dropna(subset=['diff'])
        if len(s) < 3:
            continue
        m = smf.ols('diff ~ 1', data=s).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(s.paper)[0]})
        w = 1 / s.groupby('paper').paper.transform('size')
        mw = smf.wls('diff ~ 1', data=s, weights=w).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(s.paper)[0]})
        b, se = m.params['Intercept'], m.bse['Intercept']
        rest = {}
        if meth == 'EM':   # struct_unusual: case site error vs matched-control error at the same positions, non-paper cases
            rr = pc.dropna(subset=['pred_err_site'])
            rest = lin_diff(rr, 'pred_err_site', ['C(ybin)', 'res', 'C(cclass)'], y='y', groups='pp')
            rest.update(v1=rr[rr.y == 1].pred_err_site.mean(), v0=rr[rr.y == 0].pred_err_site.mean(), n1=int(rr.y.sum()), n0=int((rr.y == 0).sum()))
        T = pd.concat([T, pd.DataFrame([dict(trend='ImmuneBuilder ensemble error at the error site (matched IMGT positions)', metric='site_err',
                       method=meth, set=cs, n1=len(s), n0=int(s.n_ctrl.median()), papers1=s.paper.nunique(), v1=s.site_err.mean(), v0=s.ctrl_site_err.mean(),
                       M_diff=b, M_lo=b - 1.96 * se, M_hi=b + 1.96 * se, M_p=m.pvalues['Intercept'], M_b=b, M_se=se,
                       PW_diff=mw.params['Intercept'], PW_lo=mw.conf_int().iloc[0, 0], PW_hi=mw.conf_int().iloc[0, 1], PW_p=mw.pvalues['Intercept'],
                       crude_p=stats.wilcoxon(s['diff']).pvalue if len(s) >= 6 else np.nan, note='case minus mean of same-class controls at the same IMGT positions +-2',
                       **({'REST_diff': rest['diff'], 'REST_lo': rest['lo'], 'REST_hi': rest['hi'], 'REST_p': rest['p'], 'REST_v1': rest['v1'], 'REST_v0': rest['v0'],
                           'REST_n1': rest['n1'], 'REST_n0': rest['n0'], 'het_p': z_het(dict(b=b, se=se), rest)} if 'diff' in rest else {}))])])
T.to_csv(f'{TAB}/struct_tests.csv', index=False)


def fmt(r):
    if pd.notna(r.get('M_OR', np.nan)):
        e = f"OR {r.M_OR:.2f} ({r.M_lo:.2f}-{r.M_hi:.2f}) p={r.M_p:.2g}"
        pw = f"OR {r.PW_OR:.2f} ({r.PW_lo:.2f}-{r.PW_hi:.2f}) p={r.PW_p:.2g}" if pd.notna(r.get('PW_OR', np.nan)) else ''
        rest = f"OR {r.REST_OR:.2f} ({r.REST_lo:.2f}-{r.REST_hi:.2f})" if pd.notna(r.get('REST_OR', np.nan)) else ''
    elif pd.notna(r.get('M_diff', np.nan)):
        e = f"{r.M_diff:+.3f} ({r.M_lo:+.3f} to {r.M_hi:+.3f}) p={r.M_p:.2g}"
        pw = f"{r.PW_diff:+.3f} ({r.PW_lo:+.3f} to {r.PW_hi:+.3f}) p={r.PW_p:.2g}" if pd.notna(r.get('PW_diff', np.nan)) else ''
        rest = f"{r.REST_diff:+.3f} ({r.REST_lo:+.3f} to {r.REST_hi:+.3f})" if pd.notna(r.get('REST_diff', np.nan)) else ''
    elif pd.notna(r.get('crude_OR', np.nan)):
        e = f"crude OR {r.crude_OR:.2f} ({r.crude_lo:.2f}-{r.crude_hi:.2f}) p={r.crude_p:.2g}"; pw = ''; rest = ''
    else:
        e = pw = rest = ''
    v = f"{r.v1:.3f} vs {r.v0:.3f}" if pd.notna(r.v1) else ''
    if pd.notna(r.get('k1', np.nan)):
        v = f"{int(r.k1)}/{r.n1} vs {int(r.k0)}/{r.n0}"
    return pd.Series(dict(trend=r.trend[:45], metric=r.metric, meth=r.method, set=r.set, n=f"{r.n1}/{r.n0} ({r.papers1}p)", values=v, est=e, PW=pw,
                          rest_EM=rest, het=f"{r.het_p:.3f}" if pd.notna(r.get('het_p', np.nan)) else '',
                          mde=f"{r.mde_OR80:.1f}" if pd.notna(r.get('mde_OR80', np.nan)) else ''))


O = T.apply(fmt, axis=1)
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 500); pd.set_option('display.max_colwidth', 60)
open(f'{TAB}/struct_tests_compact.txt', 'w').write(O.to_string(index=False))
print(O.to_string(index=False))
print('\npredictions available: Fv', int(F.pred_maha.notna().sum()), 'chains', len(CH), 'site rows', len(SE))
