"""Independent replication of the beyond-site geometry finding with (i) a NEW cryo-EM control sample that does not
reuse the investigator's control entries and does not require the whole entry to be free of disagreement runs (only the
control chain itself must have none), and (ii) an X-ray case-control set (X-ray was not analysed by the investigator).

Metrics are computed with independent code straight from the wwPDB validation XML (per-residue rama / rota / clash)
and the SAbR scan records (author numbering, ANARCI and SAbR IMGT positions, SEQRES index) -- no coordinates needed.
'Outside' = V residues with a Ramachandran classification, excluding: the error runs (cases) or the matched case's
IMGT run ranges (controls) +-3, every residue where ANARCI and SAbR disagree +-3, and residues within 3 of a chain
break or the ends of the built V region (the strict 'E3' definition of v02).

Usage: python3 v03_independent_controls.py [select|fetch|metrics|stats|all]
"""
import gzip, json, os, subprocess, sys, warnings, xml.etree.ElementTree as ET
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SU = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
CU = f'{SP}/deeper/cache/struct_unusual/vrpt'
CV = f'{SP}/deeper/cache/verify_bsb/vrpt'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_beyond_site_backbone_confounding'
TAB = f'{OUT}/tables'
os.makedirs(CV, exist_ok=True)
rng = np.random.default_rng(7)


def ybin_em(y):
    return pd.cut(y, [0, 2019, 2020, 2021, 2022, 2023, 2024, 2100],
                  labels=['<=2019', '2020', '2021', '2022', '2023', '2024', '>=2025']).astype(str)


def ybin_x(y):
    return pd.cut(y, [0, 2015, 2019, 2021, 2023, 2100], labels=['<=2015', '2016-19', '2020-21', '2022-23', '>=2024']).astype(str)


def select():
    d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str})
    d['cclass'] = np.where(d.vhh == 1, 'VHH', d.ctype)
    s = pd.read_csv(f'{SU}/sample.csv', dtype={'chain': str})
    old_ctrl_entries = set(s[s.role_cc == 'control'].pdb)
    case_entries = set(d[d.err_first == 1].pdb)
    rows = []
    # ---- EM: same case representatives as the investigator; new controls
    em = d[d.method == 'EM'].copy(); em['ybin'] = ybin_em(em.year)
    unit_err = em.groupby('unit').err_first.max()
    pool = em[(em.err_any == 0) & ~em.unit.isin(unit_err[unit_err == 1].index) & ~em.pdb.isin(old_ctrl_entries)
              & ~em.pdb.isin(case_entries)]
    pool = pool.sort_values(['pdb', 'chain']).groupby(['unit', 'pdb']).head(1)
    cases = s[s.role_cc == 'case']
    used_u, per_entry = set(), {}
    out = []
    for k in range(2):
        for i in rng.permutation(len(cases)):
            c = cases.iloc[i]
            cand = pool[(pool.cclass == c.cclass) & (pool.ybin == c.ybin) & ((pool.res - c.res).abs() <= 0.25)
                        & ~pool.unit.isin(used_u)]
            cand = cand[[per_entry.get(p, 0) < 2 for p in cand.pdb]]
            if not len(cand):
                continue
            pick = cand.iloc[rng.integers(len(cand))]
            used_u.add(pick.unit); per_entry[pick.pdb] = per_entry.get(pick.pdb, 0) + 1
            out.append(dict(arm='EM', role_cc='control', set_id=c.set_id, unit=pick.unit, pdb=pick.pdb,
                            chain=pick.chain, cclass=pick.cclass, ybin=pick.ybin, res=pick.res, year=pick.year))
    for _, c in cases.iterrows():
        out.append(dict(arm='EM', role_cc='case', set_id=c.set_id, unit=c.unit, pdb=c.pdb, chain=c.chain,
                        cclass=c.cclass, ybin=c.ybin, res=c.res, year=c.year))
    # ---- X-ray: cases = first-occurrence error units (earliest entry, err_first chain); 4 controls each
    xr = d[d.method == 'X-ray'].copy(); xr['ybin'] = ybin_x(xr.year)
    xc = xr[xr.err_first == 1].sort_values(['deposit', 'pdb', 'chain']).groupby('unit', sort=False).head(1)
    uerr = xr.groupby('unit').err_first.max()
    xpool = xr[(xr.err_any == 0) & ~xr.unit.isin(uerr[uerr == 1].index) & ~xr.pdb.isin(case_entries)]
    xpool = xpool.sort_values(['pdb', 'chain']).groupby(['unit', 'pdb']).head(1)
    used_u, per_entry = set(), {}
    for k in range(4):
        for i in rng.permutation(len(xc)):
            c = xc.iloc[i]
            cand = xpool[(xpool.cclass == c.cclass) & (xpool.ybin == c.ybin) & ((xpool.res - c.res).abs() <= 0.2)
                         & ~xpool.unit.isin(used_u)]
            cand = cand[[per_entry.get(p, 0) < 1 for p in cand.pdb]]
            if not len(cand):
                cand = xpool[(xpool.cclass == c.cclass) & ((xpool.year - c.year).abs() <= 3)
                             & ((xpool.res - c.res).abs() <= 0.25) & ~xpool.unit.isin(used_u)]
                cand = cand[[per_entry.get(p, 0) < 1 for p in cand.pdb]]
            if not len(cand):
                continue
            pick = cand.iloc[rng.integers(len(cand))]
            used_u.add(pick.unit); per_entry[pick.pdb] = 1
            out.append(dict(arm='Xray', role_cc='control', set_id=c.unit, unit=pick.unit, pdb=pick.pdb,
                            chain=pick.chain, cclass=pick.cclass, ybin=c.ybin, res=pick.res, year=pick.year))
    for _, c in xc.iterrows():
        out.append(dict(arm='Xray', role_cc='case', set_id=c.unit, unit=c.unit, pdb=c.pdb, chain=c.chain,
                        cclass=c.cclass, ybin=c.ybin, res=c.res, year=c.year))
    S = pd.DataFrame(out)
    S.to_csv(f'{TAB}/v03_sample.csv', index=False)
    print(S.groupby(['arm', 'role_cc']).agg(n=('unit', 'size'), entries=('pdb', 'nunique'), res=('res', 'median'),
                                            year=('year', 'median')))
    print(S.groupby(['arm', 'role_cc']).cclass.value_counts().unstack())


def vpath(p):
    a = f'{CU}/{p}_validation.xml.gz'
    return a if os.path.exists(a) else f'{CV}/{p}_validation.xml.gz'


def get(p):
    fn = vpath(p)
    if os.path.exists(fn) and os.path.getsize(fn) > 0:
        return 'cached'
    url = f'https://files.rcsb.org/pub/pdb/validation_reports/{p[1:3]}/{p}/{p}_validation.xml.gz'
    r = subprocess.run(['curl', '-sS', '-f', '--retry', '3', '-o', fn + '.part', url], capture_output=True)
    if r.returncode == 0:
        os.rename(fn + '.part', fn); return 'ok'
    return 'fail'


def fetch():
    S = pd.read_csv(f'{TAB}/v03_sample.csv', dtype={'chain': str})
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(get, sorted(set(S.pdb))))
    print(pd.Series(res).value_counts().to_dict())


def parse_vrpt(args):
    p, chains = args
    out = {}
    try:
        for ev, el in ET.iterparse(gzip.open(vpath(p)), events=('end',)):
            if el.tag == 'ModelledSubgroup':
                if el.get('model', '1') == '1' and el.get('chain') in chains:
                    key = (el.get('chain'), int(el.get('resnum')), (el.get('icode') or '').strip())
                    out[key] = (el.get('rama'), el.get('rota'), sum(1 for c in el if c.tag == 'clash'))
                el.clear()
    except Exception as e:
        return p, None
    return p, out


def load_sabr(keys):
    out = {}
    for fn in ['validation', 'snapshot_rest', 'new']:
        with open(f'{SP}/sabr/out/{fn}.jsonl') as f:
            for line in f:
                i = line.find('"chain": "')
                p = line[9:13]
                ch = line[i + 10:line.index('"', i + 10)]
                if (p, ch) in keys:
                    out[(p, ch)] = json.loads(line)
    return out


def dilate(m, k):
    o = m.copy()
    for i in np.where(m)[0]:
        o[max(0, i - k):i + k + 1] = True
    return o


def metrics():
    S = pd.read_csv(f'{TAB}/v03_sample.csv', dtype={'chain': str})
    runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv', dtype={'chain': str})
    case_runs = {k: g[['imgt0', 'imgt1']].values.tolist() for k, g in runs.groupby(['pdb', 'chain'])}
    set_runs = {r.set_id: case_runs.get((r.pdb, r.chain), []) for r in S[S.role_cc == 'case'].itertuples()}
    keys = set(zip(S.pdb, S.chain))
    sab = load_sabr(keys)
    jobs = [(p, set(g.chain)) for p, g in S.groupby('pdb')]
    with ProcessPoolExecutor(2) as ex:
        VR = dict(ex.map(parse_vrpt, jobs, chunksize=4))
    rows = []
    for r in S.itertuples():
        rec = sab.get((r.pdb, r.chain)); vr = VR.get(r.pdb)
        o = dict(arm=r.arm, role_cc=r.role_cc, set_id=r.set_id, unit=r.unit, pdb=r.pdb, chain=r.chain,
                 cclass=r.cclass, ybin=r.ybin, res=r.res, year=r.year)
        if rec is None or vr is None or 'residues' not in rec:
            o['skip'] = 'no sabr' if rec is None else 'no vrpt'
            rows.append(o); continue
        res = rec['residues']
        an = np.array([x[0] for x in res]); ic = [x[1] or '' for x in res]
        si = np.array([x[3] for x in res])
        ap = np.array([np.nan if x[4] is None else x[4] for x in res], float)
        ai = [x[5] or '' for x in res]
        sp_ = np.array([np.nan if x[6] is None else x[6] for x in res], float)
        sins = [x[7] or '' for x in res]
        n = len(res)
        inV = ~np.isnan(ap) | ~np.isnan(sp_)
        site = np.zeros(n, bool)
        for a, b in set_runs.get(r.set_id, []):
            site |= ((ap >= a) & (ap <= b)) | ((sp_ >= a) & (sp_ <= b))
        site &= inV
        if not site.any():
            o['skip'] = 'site not present'
            rows.append(o); continue
        dis = inV & ((ap != sp_) | np.array([x != y for x, y in zip(ai, sins)]) | np.isnan(ap) | np.isnan(sp_))
        gap = np.r_[False, np.diff(si) > 1]
        brk = gap | np.r_[gap[1:], False]
        vi = np.where(inV)[0]; brk[vi.min()] = True; brk[vi.max()] = True
        ex0 = dilate(site, 3)
        ex3 = ex0 | dilate(dis, 3) | dilate(brk, 3)
        v = [vr.get((r.chain, int(a), c), (None, None, 0)) for a, c in zip(an, ic)]
        rama = np.array([x[0] for x in v], object); rota = np.array([x[1] for x in v], object)
        clash = np.array([x[2] > 0 for x in v])
        hasr = np.array([x is not None for x in rama])
        hasrot = np.array([x is not None for x in rota])
        o['frac_in_vrpt'] = float(np.mean([x != (None, None, 0) for x in v]))
        for lab, m in [('E0r', inV & ~ex0 & hasr), ('E3', inV & ~ex3 & hasr)]:
            nn = int(m.sum()); o[f'n_{lab}'] = nn
            if nn >= 10:
                o[f'allowed_{lab}'] = 100 * np.sum(rama[m] == 'Allowed') / nn
                o[f'nonfav_{lab}'] = 100 * np.sum(np.isin(rama[m], ['Allowed', 'OUTLIER'])) / nn
                o[f'clash_{lab}'] = 100 * clash[m].sum() / nn
                mr = m & hasrot
                o[f'rota_{lab}'] = 100 * np.sum(rota[mr] == 'OUTLIER') / max(mr.sum(), 1)
        o['n_breaks'] = int(gap[inV].sum())
        rows.append(o)
    M = pd.DataFrame(rows)
    M.to_csv(f'{TAB}/v03_unit_metrics.csv', index=False)
    print(M.groupby(['arm', 'role_cc']).agg(n=('unit', 'size'), ok=('allowed_E3', 'count'),
                                            inv=('frac_in_vrpt', 'mean')))
    print(M.skip.value_counts())


def stats_():
    M = pd.read_csv(f'{TAB}/v03_unit_metrics.csv', dtype={'chain': str})
    d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str},
                    usecols=['pdb', 'chain', 'last_author', 'paper', 'sars2', 'sw_Phenix', 'role'])
    M = M.merge(d, on=['pdb', 'chain'], how='left')
    M['author_c'] = M.last_author.fillna(M.paper).fillna(M.pdb)
    M['case'] = (M.role_cc == 'case').astype(int)
    M['sars2'] = M.sars2.astype(float)
    # compare my independent case values with the investigator's (EM)
    U = pd.read_csv(f'{SU}/unit_metrics.csv', dtype={'chain': str})
    chk = M[(M.arm == 'EM') & (M.case == 1)].merge(U[['pdb', 'chain', 'rama_allowed_out', 'clash_out']], on=['pdb', 'chain'])
    print('EM cases: my E0r vs investigator rama_allowed_out: corr',
          round(chk[['allowed_E0r', 'rama_allowed_out']].corr().iloc[0, 1], 3), 'means',
          round(chk.allowed_E0r.mean(), 2), round(chk.rama_allowed_out.mean(), 2))
    rows = []
    for arm, x0 in M.groupby('arm'):
        for y in ['allowed_E0r', 'allowed_E3', 'nonfav_E3', 'clash_E3', 'rota_E3']:
            for lab, extra, sub in [('base', '', None), ('+ SARS-CoV-2', ' + sars2', None),
                                    ('2020-2022 only', '', 'y'), ('non-SARS-CoV-2', '', 'ns')]:
                x = x0.dropna(subset=[y, 'sars2'] if 'sars2' in extra else [y])
                if sub == 'y':
                    x = x[x.year.between(2020, 2022)]
                if sub == 'ns':
                    x = x[x.sars2 == 0]
                if x.case.sum() < 5:
                    continue
                f = f'{y} ~ case + C(ybin) + res' + extra + (' + C(cclass)' if x.cclass.nunique() > 1 else '')
                m = smf.ols(f, x).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(x.author_c)[0]})
                lo, hi = m.conf_int().loc['case']
                rows.append(dict(arm=arm, metric=y, model=lab, n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()),
                                 n_ctrl_entries=x[x.case == 0].pdb.nunique(),
                                 case_mean=x[x.case == 1][y].mean(), ctrl_mean=x[x.case == 0][y].mean(),
                                 case_median=x[x.case == 1][y].median(), ctrl_median=x[x.case == 0][y].median(),
                                 effect=m.params['case'], ci_lo=lo, ci_hi=hi, p=m.pvalues['case']))
    # investigator's controls vs my new controls (EM), same metric definition: is the old control set unusually clean?
    T = pd.DataFrame(rows)
    T.to_csv(f'{TAB}/v03_models.csv', index=False)
    pd.set_option('display.width', 250)
    print(T.round(4).to_string())


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if step in ('select', 'all'):
        select()
    if step in ('fetch', 'all'):
        fetch()
    if step in ('metrics', 'all'):
        metrics()
    if step in ('stats', 'all'):
        stats_()
