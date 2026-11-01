import json, gzip, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
O='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/tmpl2/out'; T='/tmp/tmpl'
R=pd.read_pickle(f'{O}/register_calls.pkl'); R=R[R.called].copy(); R['date']=pd.to_datetime(R.date)
Fa=pd.read_pickle(f'{O}/family_summary_arb.pkl'); real=pd.read_pickle(f'{O}/families_real.pkl')
F=pd.read_pickle(f'{T}/families.pkl').set_index('family')
# 1) after a correct structure appears, do later papers still carry the error?
rows=[]
for fid in real.family:
    g=R[R.family==fid].sort_values('date')
    S=g[g.S]; Oo=g[g.O]
    if S.empty or Oo.empty: continue
    firstS=S.date.min(); firstO=Oo.date.min()
    if firstS<firstO:  # error came first, correct structure appeared later
        after=g[(g.date>firstO)&~g.paper.isin(set(g[g.date<=firstO].paper))]
        pp=after.groupby('paper').agg(S=('S','any'))
        rows.append(dict(family=fid,years_to_first_correct=(firstO-firstS).days/365.25,later_papers=len(pp),later_with_error=int(pp.S.sum())))
X=pd.DataFrame(rows)
print('== Contested real errors where the error came first (n=%d families)'%len(X))
print('   median years until a structure with the other register was deposited: %.1f'%X.years_to_first_correct.median())
print('   papers deposited after the correct structure existed: %d; still carrying the error: %d (%.0f%%)'%(X.later_papers.sum(),X.later_with_error.sum(),100*X.later_with_error.sum()/max(1,X.later_papers.sum())))
# 2) antibody vs other
an=json.load(open('/tmp/supp/meta2/anarci_all.json'))
t=pd.read_pickle(f'{T}/errors_seqmapped.pkl'); famseq=t.groupby('family').seq.first()
def isab(s):
    return any(d['chain_type'] in 'HKL' for d in an.get(s,[]))
real['antibody']=real.family.map(lambda f: isab(famseq.get(f,'')))
for lab,s in real.groupby('antibody'):
    print('%s: %d real-error families; ≥2 labs %.0f%%; later papers carrying error %.0f%%; still in 2023+ %.0f%%'%('Antibody' if lab else 'Non-antibody',len(s),100*(s.labs_S>=2).mean(),100*s.post_papers_S.sum()/max(1,s.post_papers.sum()),100*(s.recent_S>0).mean()))
# 3) top examples
names={}
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz','rt') as fh:
    for line in fh:
        if line.startswith('>'):
            h=line[1:].split(None,3); names[h[0]]=h[3].strip() if len(h)>3 else ''
Rc=pd.read_pickle(f'{O}/register_calls_chain.pkl'); ref=Rc[Rc.is_ref].set_index('family')
real['protein']=real.family.map(lambda f: names.get(f"{ref.loc[f].pdb}_{ref.loc[f].chain}",'') if f in ref.index else '')
real['example_ref']=real.family.map(lambda f: ref.loc[f].pdb if f in ref.index else '')
cols=['protein','example_ref','labs_S','papers_S','papers_O','origin_date','origin_res','last_S','arb']
top=real.sort_values(['labs_S','papers_S'],ascending=False).head(15)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',45)
print('\n== Most widely propagated likely-real errors'); print(top[cols].to_string(index=False))
real.to_pickle(f'{O}/families_real2.pkl'); X.to_csv(f'{O}/after_correct.csv',index=False)
# overall entries in sample carrying likely-real errors
print('\nentries in sample carrying a likely-real error:',R[R.family.isin(real.family)&R.S].pdb.nunique())
