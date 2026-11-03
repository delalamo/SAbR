"""Re-count SN3 after (i) classifying each positive by the high-resolution check (v05 + the bovine check of
../antigen_competition_capsid_target_confounding), (ii) matching the expectation additionally on species group,
nanobody and atypical CDR length, and (iii) collapsing to papers and labs."""
import numpy as np, pandas as pd, pickle
from scipy import stats
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SN='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN3_rare_column_in_window_confounding'
R=pd.read_csv(f'{OUT}/v01_unit_modes.csv')
# classification of the 14 positives (evidence in v02_case_numbering.log, v05_hires_scan.csv, bovine_hires_check.csv)
CLS={'8c8t':('tool artifact','FR3 85A-G insertion: same run 83-85x / SAbR 82A..85 in 4/4 X-ray 1.6-2.6 A (5jxa, 4s1q, 7sx7, 4jb9); class already treated as convention artifact by the pipeline'),
     '7cws':('tool artifact','H 4A (EVQLLEQSG): same run 3-4A / SAbR 2..4 in 5/5 X-ray 1.75-2.5 A (4n0y, 1rhh, 4g6a, 6uyd, 6wo3)'),
     '8fsj':('tool artifact','H 4A (EVQLLEQSG): as above'),
     '7nk2':('tool artifact','H 73: same run 70-73 / SAbR 66..72 in 9wmg (1.96 A)'),
     '8bw0':('tool artifact','H 70A: run 69-70A / SAbR 68A..70 in 9lad (2.02 A); EM run longer (63-70A)'),
     '8r4b':('tool artifact','H 67A-C (3-residue CDR2/FR3 insertion): run 61-67C / SAbR 60A..67 in 6oca (2.11 A)'),
     '8ult':('model deviation','same V domain in 8uki (1.75 A X-ray) numbered identically by both tools; EM chain 8ulu L slips'),
     '7d3l':('model deviation, no register shift (D0)','bovine ultralong CDR3; 15/15 high-res bovine VH agree (other verifier); CDR3-internal'),
     '7d3m':('model deviation','bovine ultralong CDR3 (same lab); unit has D0 and Dminus runs'),
     '8grr':('model deviation','bovine CDR3 (same FMDV lab, Kun L.)'),
     '7t6x':('probable model deviation','lambda extra FR1 residue at IMGT 10 (run ANARCI 2-10 vs SAbR 1-9, N-terminus unbuilt); a related lambda with an IMGT-10 residue (9nj9 L, 2.6 A EM) and 2 X-ray lambdas (4r26, 4fqc) give no run, but none has the identical insertion'),
     '22ge':('untestable, insertion-placement form','rabbit 4-residue FR1 insertion: ANARCI 9A-C/10 vs SAbR 8A-D/9 (offset of one column)'),
     '12iu':('untestable, insertion-placement form','kappa 3-residue insertion placed in CDR2 (ANARCI) vs FR3 71A-C (SAbR); no high-res relative')}
U=pd.read_csv(f'{SN}/units.csv.gz',low_memory=False,usecols=['unit','pdb'])
R['cls']=[CLS.get(p,('',''))[0] if o else '' for p,o in zip(R.pdb,R.obs)]
R['evidence']=[CLS.get(p,('',''))[1] if o else '' for p,o in zip(R.pdb,R.obs)]
R.loc[R.obs,['mode','pdb','paper','last_author','cls','evidence']].to_csv(f'{OUT}/v07_positive_classification.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',90)
print(R.loc[R.obs,['mode','pdb','last_author','dep_year','cls']].to_string())
E='exp_clean_EM (investigator)'
def oe(x,lab,obsmask=None):
    o=int((x.obs if obsmask is None else obsmask).sum()); e=x[E].sum()
    p=stats.poisson.sf(o-1,e) if o>0 else 1.0
    print(f'{lab:70s} n={len(x):3d} obs={o:2d} ({o/len(x):.3f}) exp={e:.2f} ({e/len(x):.4f}) O/E={o/e if e else np.nan:5.1f} p={p:.2e}')
    return dict(test=lab,n=len(x),obs=o,obs_frac=o/len(x),exp=e,exp_frac=e/len(x),ratio=o/e if e else np.nan,p=p)
T=[]
T.append(oe(R,'as published (all 199 unit-modes)'))
art=R.cls=='tool artifact'
T.append(oe(R[~art],'drop the 6 positives reproduced as tool disagreement at high resolution'))
T.append(oe(R[~art],'  ... and count only confirmed model deviations as positive',obsmask=R[~art].cls.str.startswith('model deviation')))
T.append(oe(R[~art],'  ... confirmed model deviations with a register shift (exclude D0)',obsmask=R[~art].cls.eq('model deviation')))
bov=R.last_author.isin(['Lou, Z.','Kun, L.'])
T.append(oe(R[~art&~bov],'drop artifacts and the bovine FMDV lab (Lou Z./Kun L.)'))
T.append(oe(R[~art&~bov],'  ... confirmed model deviations only',obsmask=R[~art&~bov].cls.str.startswith('model deviation')))
T.append(oe(R[~art],'drop artifacts; confirmed + probable model deviations',obsmask=R[~art].cls.str.contains('model deviation')))
T.append(oe(R[~art&~bov],'drop artifacts and bovine lab; confirmed + probable model deviations',obsmask=R[~art&~bov].cls.str.contains('model deviation')))
# ---- expectation matched also on species group, nanobody, atypical CDR length (any of L1/L2/L3 atypical)
SNU=pd.read_csv(f'{SN}/units.csv.gz',low_memory=False).set_index('unit')
db=pickle.load(open(f'{SP}/deeper/cache/seq_novelty/abdb.pkl','rb')); D,M,keys=db['D'],db['M'],db['keys']
pos=np.array([k[0] for k in keys]); OCC={ct:((M[(D.ctype==ct).values]>0).mean(0)) for ct in 'HKL'}
SNU['yb']=pd.cut(SNU.dep_year,[0,2018,2019,2020,2021,2022,2023,2100],labels=False)
SNU['spg']=np.where(SNU.q_species_all.isin(['human','mouse']),'hm',np.where(SNU.vhh_u.fillna(0)>0,'vhh','other'))
SNU['atyp']=(SNU[['L1_atyp','L2_atyp','L3_atyp']].fillna(0).sum(1)>0).astype(int)
ctrl=SNU[(SNU.method_u=='EM')&SNU.clean&(SNU.dbrow>=0)&(SNU.dbrow<len(M))]
W=pd.read_csv(f'{SN}/unit_mode_windows.csv')
ex=[]
for i,r in R.iterrows():
    u=SNU.loc[r.unit]; w=W[(W.unit==r.unit)&(W['mode']==r['mode'])].iloc[0]
    a=int(w.i0)-2; b=min(int(w.i1),int(w.i0)+24)+2; win=(pos>=a)&(pos<=b)
    cc=ctrl[(ctrl.ctype==u.ctype)&(ctrl.spg==u.spg)&(ctrl.atyp==u.atyp)]
    cc2=cc[cc.yb==u.yb] if (cc.yb==u.yb).sum()>=30 else cc
    occ=OCC[u.ctype]<0.01
    ex.append(np.mean([((M[x]>0)&occ&win).any() for x in cc2.dbrow.values]) if len(cc2) else np.nan)
R['exp_matched_species_atyp']=ex
E='exp_matched_species_atyp'
print('\nExpectation matched on chain type, year bin (if >=30 controls), species group (human/mouse, nanobody, other) and atypical CDR length:')
print(R.groupby(R.obs)[E].describe())
T.append(oe(R.dropna(subset=[E]),'species+atypical-length matched, as published'))
T.append(oe(R[~art].dropna(subset=[E]),'species+atypical-length matched, artifacts dropped'))
T.append(oe(R[~art].dropna(subset=[E]),'species+atypical-length matched, confirmed model deviations only',obsmask=R[~art].dropna(subset=[E]).cls.str.startswith('model deviation')))
# ---- paper / lab collapse
P=R.groupby('paper').agg(obs=('obs','max'),lab=('last_author','first')).reset_index()
print('\npapers with >=1 positive:',int(P.obs.sum()),'of',len(P),'; labs:',R[R.obs].last_author.nunique(),
      '; confirmed model-deviation papers:',R[R.cls.str.startswith('model deviation')].paper.nunique(),
      'labs:',sorted(R[R.cls.str.startswith('model deviation')].last_author.unique()))
pd.DataFrame(T).to_csv(f'{OUT}/v07_recount.csv',index=False)
R.to_csv(f'{OUT}/v07_unit_modes_classified.csv',index=False)
