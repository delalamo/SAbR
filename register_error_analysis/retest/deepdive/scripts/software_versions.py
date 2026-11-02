"""Entry-level: EM entries with a first-occurrence register error vs EM entries with no disagreement run, by recorded
model-refinement / model-fitting program and version, within period strata."""
import json, numpy as np, pandas as pd
from scipy.stats import fisher_exact
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
O = '/home/user/SAbR/register_error_analysis/retest/deepdive/tables'
V = json.load(open(f'{SP}/unexpl/em_software_versions.json'))
M = pd.read_pickle(f'{SP}/unexpl/master4.pkl'); E = M[M.method == 'EM'].groupby('pdb').agg(year=('year', 'min')).reset_index()
R = pd.read_csv(f'{SP}/unexpl/runs_origin_v2.csv'); R['cls'] = R.origin.str.split(' ').str[0]
first = set(R[~R.cls.isin(['1a', '1b', '2', '4'])].pdb); anyrun = set(R.pdb)
E['case'] = E.pdb.isin(first); E = E[E.case | ~E.pdb.isin(anyrun)]
E['period'] = np.where(E.year <= 2021, '2018-21', '2022+'); E = E[E.year >= 2018]
def feats(p):
    sw = (V.get(p) or {}).get('em_software') or []
    f = {}
    for s in sw:
        cat, nm, ver = s.get('category') or '', (s.get('name') or '').lower(), (s.get('version') or '')
        if cat not in ('MODEL REFINEMENT', 'MODEL FITTING'): continue
        for tool in ['phenix', 'coot', 'isolde', 'rosetta', 'chimerax', 'chimera', 'namdinator', 'mdff', 'modelangelo', 'refmac', 'servalcat']:
            if tool in nm:
                f[tool] = True
                if tool == 'phenix' and ver[:4] in ('1.17', '1.18', '1.19', '1.20', '1.21'): f[f'phenix {ver[:4]}'] = True
                if tool == 'coot' and ver[:3] in ('0.8', '0.9'): f[f'coot {ver[:3]}'] = True
                break
    f['chimera fit, no Coot/ISOLDE/Rosetta'] = (f.get('chimera') or f.get('chimerax')) and not (f.get('coot') or f.get('isolde') or f.get('rosetta'))
    f['phenix version recorded'] = any(k.startswith('phenix 1.') for k in f)
    return f
F = pd.DataFrame([feats(p) for p in E.pdb]).fillna(False).astype(bool); E = pd.concat([E.reset_index(drop=True), F], axis=1)
rows = []
for per, g in E.groupby('period'):
    c, k = g[g.case], g[~g.case]
    for col in F.columns:
        a, b = c[col].sum(), k[col].sum()
        if a + b < 10: continue
        p = fisher_exact([[a, len(c) - a], [b, len(k) - b]])[1]
        rate = 100 * a / (a + b)
        rows.append(dict(period=per, feature=col, case_entries=int(a), clean_entries=int(b), pct_case_of_entries_using=round(rate, 1),
                         base_rate=round(100 * len(c) / len(g), 1), p=round(p, 4)))
T = pd.DataFrame(rows).sort_values(['period', 'p']); pd.set_option('display.width', 200); print(T.to_string(index=False))
T.to_csv(f'{O}/software_versions.csv', index=False)
