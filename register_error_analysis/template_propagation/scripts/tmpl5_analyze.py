import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
T='/tmp/tmpl'; O='/tmp/tmpl/out'
import os; os.makedirs(O,exist_ok=True)
R=pd.read_pickle(f'{T}/register_calls.pkl')
F=pd.read_pickle(f'{T}/families.pkl').set_index('family')
irm=json.load(open(f'{T}/irm_selected.json'))
R['S']=R.call=='same as flagged'; R['O']=R.call=='other register'; R['called']=R.S|R.O
R['date']=pd.to_datetime(R.date)
print('== call distribution'); print(R.call.value_counts())
# ---------- validation vs conkit flags within the authors' snapshot (3-5 A, released by 2022-04-05; release ~ we only have deposit here)
snap=R[(R.res>=3.0)&(R.res<=5.0)&(R.date<='2022-03-01')&R.method.isin(['X-RAY DIFFRACTION','ELECTRON MICROSCOPY'])&~R.is_ref&R.called]
print('\n== validation: in-snapshot entries of the same family (excluding reference)')
print(pd.crosstab(snap.flagged,snap.S,margins=True).rename(columns={True:'same register as flagged',False:'other register'}))
# ---------- family level
rows=[]
for fid,g in R[R.called].groupby('family'):
    g=g.sort_values('date'); S=g[g.S]; Ot=g[g.O]
    if S.empty: continue
    o=S.iloc[0]; origin_paper=o.paper; od=o.date
    prior_O=Ot[Ot.date<od]
    post=g[(g.date>od)&(g.paper!=origin_paper)]
    # papers rather than entries for propagation
    postp=post.groupby('paper').agg(S=('S','any'),O=('O','any'),date=('date','min'),lab=('last_author','first'))
    hi=g[g.res<=2.5]
    rows.append(dict(family=fid,n_called=len(g),papers=g.paper.nunique(),papers_S=S.paper.nunique(),labs_S=S.last_author.nunique(),
        papers_O=Ot.paper.nunique(),origin_date=od,origin_res=o.res,origin_method=o.method,origin_flagged=o.flagged,
        first_entry_status='S' if g.iloc[0].S else 'O',prior_O_papers=prior_O.paper.nunique(),
        post_papers=len(postp),post_papers_S=int(postp.S.sum()),post_papers_O=int((postp.O&~postp.S).sum()),
        post_labs_S=postp[postp.S].lab.nunique(),
        hires_S=int(hi.S.sum()),hires_O=int(hi.O.sum()),recent_S=int(g[(g.date>='2023-01-01')].S.sum()),recent_O=int(g[(g.date>='2023-01-01')].O.sum()),
        last_S=S.date.max(),span_years=(S.date.max()-od).days/365.25))
Fm=pd.DataFrame(rows).merge(F[['s0','s1','shift','n_pdb_entries']],left_on='family',right_index=True)
Fm.to_pickle(f'{O}/family_summary.pkl')
n=len(Fm); print(f'\n== {n} families where the flagged register was confirmed in >=1 entry, with >=2 papers on this sequence')
Fm['class']=np.select([
    (Fm.first_entry_status=='S')&(Fm.post_papers_S>0),
    (Fm.first_entry_status=='S')&(Fm.post_papers_S==0),
    (Fm.first_entry_status=='O')&(Fm.post_papers_S>0),
    (Fm.first_entry_status=='O')&(Fm.post_papers_S==0)],
    ['Present in first structure; carried into later papers','Present in first structure; later papers differ','Introduced after a correct structure existed; carried into later papers','Introduced after a correct structure existed; confined to one paper'],'other')
print(Fm['class'].value_counts().to_string())
print('\nfamilies where shifted register appears in >=2 papers: %d (%.0f%%); >=2 labs: %d'%((Fm.papers_S>=2).sum(),100*(Fm.papers_S>=2).mean(),(Fm.labs_S>=2).sum()))
print('families with shifted register still present in entries deposited 2023+: %d (%.0f%%)'%((Fm.recent_S>0).sum(),100*(Fm.recent_S>0).mean()))
# propagation probability at paper level
tot=Fm.post_papers.sum(); s=Fm.post_papers_S.sum()
print('\n== later papers (after the error first appeared, different paper): %d; carrying the shifted register: %d (%.0f%%)'%(tot,s,100*s/tot))
# split by whether a correct structure already existed before the error
for lab,sub in [('no correct structure before error',Fm[Fm.prior_O_papers==0]),('correct structure existed before error',Fm[Fm.prior_O_papers>0])]:
    print('   %s: %d families; later papers %d, carrying shift %.0f%%'%(lab,len(sub),sub.post_papers.sum(),100*sub.post_papers_S.sum()/max(1,sub.post_papers.sum())))
# high-res arbitration
print('\n== high-resolution (<=2.5 A) structures in family:')
print(' families with hi-res structure:',((Fm.hires_S+Fm.hires_O)>0).sum(),
      '| hi-res only in shifted register:',((Fm.hires_S>0)&(Fm.hires_O==0)).sum(),
      '| only other register:',((Fm.hires_O>0)&(Fm.hires_S==0)).sum(),'| both:',((Fm.hires_S>0)&(Fm.hires_O>0)).sum())
# ---------- inheritance via recorded starting model
R2=R[R.called].copy()
status={(r.family,r.pdb):('S' if r.S else 'O') for r in R2.itertuples()}
fam_entries=R2.groupby('family').pdb.apply(set).to_dict()
orig=Fm.set_index('family')
rec=[]
for r in R2.itertuples():
    if r.family not in orig.index or r.is_ref: continue
    o=orig.loc[r.family]
    if r.date<=o.origin_date: continue
    cited=[m['acc'] for m in irm.get(r.pdb,[]) if m['acc']]
    cs=[status.get((r.family,c)) for c in cited if (r.family,c) in status]
    rec.append(dict(family=r.family,pdb=r.pdb,paper=r.paper,S=r.S,has_irm=bool(cited),
        cites=('cites shifted template' if 'S' in cs and 'O' not in cs else 'cites other-register template' if 'O' in cs and 'S' not in cs else 'cites both' if cs else ('cites unrelated/unknown model' if cited else 'no starting model recorded'))))
I=pd.DataFrame(rec)
print('\n== later entries (after first appearance of the shift): register by recorded starting model')
print(I.groupby('cites').agg(entries=('S','size'),pct_shifted=('S',lambda x:round(100*x.mean(),1))).to_string())
I.to_pickle(f'{O}/inheritance.pkl')
R.to_pickle(f'{O}/register_calls.pkl')
