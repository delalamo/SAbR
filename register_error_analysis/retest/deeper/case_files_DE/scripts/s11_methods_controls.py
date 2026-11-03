"""Methods-text comparison: papers behind D+/D-/E first-occurrence errors vs papers of fully clean cryo-EM antibody
entries (same deposition years). Europe PMC open-access XML; methods sections only. Flags are regexes."""
import sys, re, os, xml.etree.ElementTree as ET
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/scripts')
from common import *
from s03_fetch_text import fetch
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import fisher_exact
import statsmodels.formula.api as smf
CH = chains(); R = runs(); ent = pd.read_csv(f'{D}/entries.csv')
ER = pd.read_pickle(f'{CACHE}/err_struct.pkl'); bad = set(zip(ER[ER.confirm == 'not confirmed'].pdb, ER[ER.confirm == 'not confirmed'].chain))
E = R[R['mode'].isin(MODES) & (R.method == 'EM')]
E = E[[k not in bad for k in zip(E.pdb, E.chain)]]
pm = {}
for r in E.itertuples():
    d = CH[(CH.pdb == r.pdb)].doi.iloc[0]
    if isinstance(d, str) and d.startswith('10.'): pm.setdefault(d, set()).add(r.mode)
case_years = CH[CH.doi.isin(pm)].groupby('doi').year.min()
clean_e = set(ent[(~ent.err_any) & (ent.method == 'EM')].pdb)
anyerr_dois = set(CH[CH.pdb.isin(set(ent[ent.err_any].pdb))].doi.dropna())
ctl = CH[CH.pdb.isin(clean_e) & CH.doi.notna() & CH.doi.str.startswith('10.') & ~CH.doi.isin(anyerr_dois)].groupby('doi').year.min()
rng = np.random.default_rng(3); picks = []
for y, n in case_years.value_counts().items():
    pool = ctl[ctl == y].index.tolist(); rng.shuffle(pool); picks += pool[:3 * n]
dois = list(pm) + picks
with ThreadPoolExecutor(3) as ex: meta = list(ex.map(fetch, dois))
FL = {'local_or_focused_refinement': r'local(ly)?[ -]refine|focused (3D )?refine|local mask|focused classification|local (map|reconstruction)',
      'other_fab_or_homology_template': r'(fab|antibod|nanobod|vhh|heavy chain|light chain)[^.]{0,150}(pdb[^.]{0,12}\b\d[a-z0-9]{3}\b|template|homolog)|swiss-?model|homology model',
      'alphafold_or_predictor': r'alphafold|colabfold|abodybuilder|immunebuilder|igfold|rosettaantibody|nanonet',
      'rigid_body': r'rigid[- ]body',
      'modelangelo_or_auto': r'modelangelo|deeptracer|map_to_model|buccaneer|trace_and_build',
      'mutated_to_sequence': r'mutat\w+ (?:to|into) (?:the|match)|were mutated|was mutated|in silico mutat',
      'isolde': r'\bisolde\b', 'rosetta': r'\brosetta', 'namdinator_or_mdff': r'namdinator|mdff|flexible fitting',
      'variable_domain_only': r'(only|just) (the )?(variable|fv)|variable (domains?|regions?) (were|was) (built|modell?ed)|constant (domains?|regions?) (were|was) (not|omitted|removed|excluded)'}
rows = []
for d, m in zip(dois, meta):
    fn = f'{CACHE}/epmc/{re.sub(r"[^A-Za-z0-9]", "_", d)}.xml'
    if not os.path.exists(fn): rows.append(dict(doi=d, ft=False)); continue
    try: root = ET.fromstring(open(fn, 'rb').read())
    except Exception: rows.append(dict(doi=d, ft=False)); continue
    tx = lambda el: re.sub(r'\s+', ' ', ''.join(el.itertext()))
    secs = [s for s in root.iter('sec') if re.search(r'method|procedure|model|structure determination|cryo|data processing', (tx(s.find('title')) if s.find('title') is not None else '') + ' ' + (s.get('sec-type') or ''), re.I)]
    t = ' '.join(tx(s) for s in secs) if secs else (tx(root.find('.//body')) if root.find('.//body') is not None else '')
    row = dict(doi=d, ft=True, nchar=len(t))
    for k, p in FL.items(): row[k] = bool(re.search(p, t, re.I))
    rows.append(row)
P = pd.DataFrame(rows)
P['group'] = ['D/E error paper' if d in pm else 'clean control paper' for d in P.doi]
P['E'] = [('E_FR4bulge' in pm.get(d, set())) for d in P.doi]
P['Dp'] = [('Dplus_CDR3short' in pm.get(d, set())) for d in P.doi]
P['Dm'] = [('Dminus_CDR3long' in pm.get(d, set())) for d in P.doi]
P['year'] = [case_years.get(d, ctl.get(d)) for d in P.doi]
P.to_csv(f'{OUT}/tables/methods_flags_DE_vs_controls.csv', index=False)
Q = P[P.ft]
print(Q.groupby('group').size())
out = []
for sub, mask in [('any D/E', Q.group == 'D/E error paper'), ('E', Q.E), ('D+', Q.Dp), ('D-', Q.Dm)]:
    a = Q[mask]; c = Q[Q.group == 'clean control paper']
    for k in FL:
        x1, n1, x0, n0 = int(a[k].sum()), len(a), int(c[k].sum()), len(c)
        o, p = fisher_exact([[x1, n1 - x1], [x0, n0 - x0]])
        z = pd.concat([a, c]).assign(y=lambda d: d.group.eq('D/E error paper').astype(int), x=lambda d: d[k].astype(int), yb=lambda d: pd.cut(d.year, [0, 2019, 2021, 2023, 2030]).astype(str))
        try:
            mm = smf.logit('y ~ x + C(yb)', data=z).fit(disp=0); ao, ap = float(np.exp(mm.params['x'])), float(mm.pvalues['x'])
        except Exception: ao, ap = np.nan, np.nan
        out.append(dict(subset=sub, flag=k, err_papers=f'{x1}/{n1}', ctl_papers=f'{x0}/{n0}', err_pct=100 * x1 / max(n1, 1), ctl_pct=100 * x0 / max(n0, 1), fisher_OR=o, fisher_p=p, year_adj_OR=ao, year_adj_p=ap))
O = pd.DataFrame(out)
from statsmodels.stats.multitest import multipletests
O['q'] = multipletests(O.fisher_p, method='fdr_bh')[1]
O.to_csv(f'{OUT}/tables/methods_tests_DE.csv', index=False)
pd.set_option('display.width', 220); print(O.round(3).to_string())
