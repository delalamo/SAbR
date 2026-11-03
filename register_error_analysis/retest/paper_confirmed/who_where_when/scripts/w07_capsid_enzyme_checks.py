"""Capsid / enzyme follow-up on the paper-confirmed set: species of the case antibodies, leave-one-group-out
(the two capsid papers that share a co-author cluster count as one group), human antibodies only, and the
EC-style enzyme sensitivity is not re-run (needs the verifier's EC mapping); paper-level Fisher tests."""
import sys; sys.path.insert(0, '.')
from ww_common import *
from scipy.stats import fisher_exact
C = chains(); E = entry_flags(C)
f = pd.read_csv(f'{T}/w01_entry_frame.csv.gz')
sp = C.groupby('pdb').species.agg(lambda s: '|'.join(sorted(set(s.dropna().astype(str))))).rename('species')
f = f.merge(sp, on='pdb', how='left')
f['human_only'] = f.species.eq('human')
em = f[f.method == 'EM']
print('case capsid/enzyme entries:'); print(em[em.case & em.cls2.isin(['viral capsid', 'enzyme'])][['pdb', 'paper', 'cls2', 'species', 'dyear']].to_string(index=False))
print('rest_w capsid/enzyme entries:'); print(em[em.rest_w & em.cls2.isin(['viral capsid', 'enzyme'])][['pdb', 'paper', 'cls2', 'species', 'dyear']].to_string(index=False))
P = em.groupby('paper').agg(cap=('cls2', lambda s: (s == 'viral capsid').any()), enz=('cls2', lambda s: (s == 'enzyme').any()),
                            human=('human_only', 'all'), case=('case', 'max'), case_i=('case_i', 'max'), ctrl=('ctrl', 'min'), gen=('gen', 'max'), rest_w=('rest_w', 'max')).reset_index()
grp = {'10.1126/sciadv.aat7459': 'G1', '10.1016/j.chom.2020.01.003': 'G1'}   # shared co-author cluster (see w03)
rows = []
def fish(d, x, lab):
    a = d[d.case]; c = d[d.ctrl & ~d.gen & ~d.case_i]
    t = [[int(a[x].sum()), len(a) - int(a[x].sum())], [int(c[x].sum()), len(c) - int(c[x].sum())]]
    o, p = fisher_exact(t); rows.append(dict(test=lab, x=x, case=f'{t[0][0]}/{len(a)}', ctrl=f'{t[1][0]}/{len(c)}', OR=o, p=p)); print(f'{lab:60s} {x}: case {t[0][0]}/{len(a)} vs ctrl {t[1][0]}/{len(c)}  OR {o:.2f} p={p:.3g}')
for x in ['cap', 'enz']:
    fish(P, x, 'paper level, EM, window controls')
    fish(P[P.human], x, 'paper level, EM, human-only antibodies')
    fish(P[~P.paper.isin(grp)], x, 'paper level, EM, shared co-author cluster dropped')
    fish(P[~P.paper.isin(list(grp)[:1])], x, 'paper level, EM, cluster counted once')
pd.DataFrame(rows).to_csv(f'{T}/w07_capsid_enzyme_checks.csv', index=False)
