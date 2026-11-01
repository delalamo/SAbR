"""Restrict the denominator to entries released by the authors' PDB snapshot (2022-04-05).
Run after build2.py (and after the first-report build.py / build_ecod.py); then re-run analyze.py,
local_q.py, partBE.py, partD.py, prior_template.py, figures.py."""
import pandas as pd
A='/tmp/analysis'
ent=pd.read_pickle(f'{A}/entries35.pkl')
ok=set(ent.pdb[ent.release<='2022-04-05'])
for f in ['chains.pkl','ecod_domains.pkl','entries35.pkl','instances35.pkl']:
    x=pd.read_pickle(f'{A}/{f}'); x[x.pdb.isin(ok)].to_pickle(f'{A}/{f}')
