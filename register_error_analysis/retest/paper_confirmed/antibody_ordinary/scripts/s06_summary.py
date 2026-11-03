"""Collect the key estimates per trend into tables/summary_key_rows.csv (strict all/EM/X-ray, one vote per paper,
incl copies, recomputed full-set reference = rest, heterogeneity) for notes.md. No new statistics here."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from aoutil import TAB

S = pd.read_csv(f'{TAB}/seq_tests.csv')
T = pd.read_csv(f'{TAB}/struct_tests.csv')


def est(r, pre):
    if pd.notna(r.get(f'{pre}_OR', np.nan)):
        return f"OR {r[pre + '_OR']:.2f} ({r[pre + '_lo']:.2f}-{r[pre + '_hi']:.2f}) p={r[pre + '_p']:.2g}"
    if pd.notna(r.get(f'{pre}_diff', np.nan)):
        return f"{r[pre + '_diff']:+.3f} ({r[pre + '_lo']:+.3f} to {r[pre + '_hi']:+.3f}) p={r[pre + '_p']:.2g}"
    return ''


rows = []
for f, g in S.groupby('feature', sort=False):
    def get(c, m):
        x = g[(g.comparison == c) & (g.method == m)]
        return x.iloc[0] if len(x) else None
    for m in ['all', 'EM', 'X-ray']:
        s = get('strict', m)
        if s is None:
            continue
        val = (f"{int(s.k1)}/{s.n1} vs {int(s.k0)}/{s.n0} ({100*s.v1:.1f}% vs {100*s.v0:.1f}%)" if s.binary and pd.notna(s.get('k1'))
               else f"mean {s.v1:.3f} vs {s.v0:.3f} (n {s.n1}/{s.n0})")
        main = est(s, 'M1') or ('crude ' + est(s, 'crude'))
        i, rf, hc = get('incl', m), get('rest_full', m), get('het_case', m)
        rows.append(dict(kind='seq', trend=s.label, feature=f, method=m, papers=s.papers1, strict_values=val, strict_M1=main,
                         strict_M2=est(s, 'M2'), strict_PW=est(s, 'PW'), strict_M3=est(s, 'M3'),
                         mde_OR80=s.get('mde_OR80'), incl_M1=(est(i, 'M1') or 'crude ' + est(i, 'crude')) if i is not None else '',
                         rest_full_M1=(est(rf, 'M1') or 'crude ' + est(rf, 'crude')) if rf is not None else '',
                         rest_values=(f"{100*rf.v1:.1f}% vs {100*rf.v0:.1f}%" if rf.binary else f"{rf.v1:.3f} vs {rf.v0:.3f}") if rf is not None else '',
                         het_z_p=s.het_z_p, het_case_paper_vs_rest_win=(est(hc, 'M1') or 'crude ' + est(hc, 'crude')) if hc is not None else ''))
for (tr, mt), g in T.groupby(['trend', 'metric'], sort=False):
    for m in ['all', 'EM', 'X-ray']:
        s = g[(g.set == 'strict') & (g.method == m)]
        if not len(s):
            continue
        s = s.iloc[0]; i = g[(g.set == 'incl') & (g.method == m)]
        val = (f"{int(s.k1)}/{s.n1} vs {int(s.k0)}/{s.n0}" if pd.notna(s.get('k1', np.nan)) else f"mean {s.v1:.3f} vs {s.v0:.3f} (n {s.n1}/{s.n0})")
        rows.append(dict(kind='struct', trend=tr, feature=mt, method=m, papers=s.papers1, strict_values=val,
                         strict_M1=est(s, 'M') or ('crude ' + est(s, 'crude') if pd.notna(s.get('crude_OR', np.nan)) else ''),
                         strict_PW=est(s, 'PW'), mde_OR80=s.get('mde_OR80'),
                         incl_M1=est(i.iloc[0], 'M') if len(i) else '',
                         rest_full_M1=est(s, 'REST') if m == 'EM' else '',
                         rest_values=f"{s.REST_v1:.3f} vs {s.REST_v0:.3f}" if m == 'EM' and pd.notna(s.get('REST_v1', np.nan)) else '',
                         het_z_p=s.get('het_p', np.nan), note=s.get('note', '')))
R = pd.DataFrame(rows)
R.to_csv(f'{TAB}/summary_key_rows.csv', index=False)
pd.set_option('display.width', 400); pd.set_option('display.max_colwidth', 70); pd.set_option('display.max_rows', 500)
print(R[R.kind == 'struct'][['trend', 'feature', 'method', 'strict_values', 'strict_M1', 'strict_PW', 'incl_M1', 'rest_full_M1', 'het_z_p']].to_string(index=False))
