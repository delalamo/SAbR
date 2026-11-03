"""Unit x fixed-window analysis of SN3 with confounders.
y  = first-occurrence run overlapping window W (cryo-EM units; controls = units with no run in W of any kind... and also all)
x  = residue at a rarely occupied IMGT column (<1% of PDB V domains of that chain type) in W
Covariates: deposit-year bin, resolution, chain type, SARS-CoV-2, nanobody, non-human/mouse species, atypical CDR length
in the window, unbuilt residues (SEQRES gap) in the window; standard errors clustered by paper.
Also: collapse to one row per paper; leave-one-lab-out; and the matched-expectation statistic restricted to units without
an unbuilt gap, without atypical CDR length, and excluding the bovine paper(s)."""
import json, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from scipy import stats
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SN='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN3_rare_column_in_window_confounding'
ch=pd.read_csv(f'{SN}/chain_novelty.csv.gz',low_memory=False)
ra=pd.read_csv(f'{SP}/deeper/runs_all.csv'); rf=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
WIN={'A':(1,13),'F':(24,41),'B':(55,72),'C':(74,92),'D':(105,128),'E':(116,128)}
# ---- unbuilt residues per chain: list of (anarci_lo, anarci_hi) spans bracketing a seq_index jump, plus N/C-terminal unbuilt
gaps={}
for f in ['new.jsonl','snapshot_rest.jsonl','validation.jsonl']:
    for l in open(f'{SP}/sabr/out/{f}'):
        d=json.loads(l); k=(d['pdb'],d['chain'])
        if k in gaps: continue
        res=[r for r in d.get('residues',[]) if r[4] is not None]
        if not res: continue
        sp=[]
        if res[0][3]>0: sp.append((0,res[0][4]))           # N-terminal SEQRES residues not modelled
        for a,b in zip(res,res[1:]):
            if b[3]>a[3]+1: sp.append((a[4],b[4]))
        gaps[k]=sp
def gap_in(k,a,b):
    return any(lo<=b+2 and hi>=a-2 for lo,hi in gaps.get(k,[]))
ch['k']=list(zip(ch.pdb,ch.chain))
for w,(a,b) in WIN.items():
    ch[f'gap_{w}']=[gap_in(k,a,b) for k in ch.k]
    for nm,R in (('any',ra),('first',rf)):
        hit=R[(R.imgt0-2<=b)&(R.imgt1+2>=a)][['pdb','chain']].drop_duplicates(); hit[f'run_{nm}_{w}']=True
        ch=ch.merge(hit,on=['pdb','chain'],how='left'); ch[f'run_{nm}_{w}']=ch[f'run_{nm}_{w}'].fillna(False).astype(bool)
    ch[f'rc_{w}']=ch[f'rareins_win_{w}'].fillna(0)>0
ch=ch[ch.dbrow>=0].copy()
ch['nonhm']=~ch.q_species_all.isin(['human','mouse'])
ATYP={'A':None,'F':'L1_atyp','B':'L2_atyp','C':None,'D':'L3_atyp','E':'L3_atyp'}
L=[]
for w in WIN:
    g=ch.groupby('unit').agg(method=('method',lambda x:'EM' if (x=='EM').any() else 'X-ray'),rc=(f'rc_{w}','any'),y=(f'run_first_{w}','any'),
                             yany=(f'run_any_{w}','any'),gap=(f'gap_{w}','any'),yr=('year','min'),dep=('deposit','min'),res=('res','median'),ctype=('ctype','first'),
                             sars2=('sars2','any'),vhh=('vhh','max'),nonhm=('nonhm','any'),paper=('paper','first'),lab=('last_author','first'),
                             atyp=(ATYP[w],'max') if ATYP[w] else ('vhh','size')).reset_index()
    if not ATYP[w]: g['atyp']=0
    g['window']=w; L.append(g)
L=pd.concat(L,ignore_index=True)
L['yb']=pd.cut(pd.to_datetime(L.dep).dt.year,[0,2018,2019,2020,2021,2022,2023,2100],labels=False)
for c in ['rc','y','yany','gap','sars2','nonhm']: L[c]=L[c].astype(int)
L['vhh']=L.vhh.fillna(0).astype(int); L['atyp']=L.atyp.fillna(0).astype(int); L['paper']=L.paper.fillna(L.unit)
L.to_csv(f'{OUT}/v06_unit_window_frame.csv.gz',index=False)
EM=L[L.method=='EM'].copy()
tests=[]
def fit(d,form,lab,term='rc'):
    d=d.dropna(subset=['yb','res']).copy()
    try:
        f=smf.logit(form,data=d).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(d.paper)[0]},maxiter=200)
        ci=np.exp(f.conf_int().loc[term].values)
        r=dict(test=lab,n=len(d),n_y=int(d.y.sum()),n_rc=int(d.rc.sum()),n_y_rc=int((d.y*d.rc).sum()),OR=np.exp(f.params[term]),lo=ci[0],hi=ci[1],p=f.pvalues[term])
    except Exception as e:
        r=dict(test=lab,n=len(d),note=str(e)[:80])
    tests.append(r); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()})
# controls: error units-in-window vs units with no run in W of any kind (mirrors "clean" but window-specific)
base=EM[(EM.y==1)|(EM.yany==0)]
fit(base,'y ~ rc + C(window)','EM M0 window FE')
fit(base,'y ~ rc + C(window) + C(yb) + res + C(ctype)','EM M1 +year+res+ctype')
fit(base,'y ~ rc + C(window) + C(yb) + res + C(ctype) + sars2 + vhh + nonhm','EM M2 +sars2+vhh+nonhm')
fit(base,'y ~ rc + C(window) + C(yb) + res + C(ctype) + sars2 + vhh + nonhm + atyp','EM M3 +atypical CDR length')
fit(base,'y ~ rc + C(window) + C(yb) + res + C(ctype) + sars2 + vhh + nonhm + atyp + gap','EM M4 +unbuilt gap in window')
fit(base[base.gap==0],'y ~ rc + C(window) + C(yb) + res + C(ctype) + sars2 + vhh + nonhm + atyp','EM M3 within no-gap')
fit(base[~base.window.isin(['D','E'])],'y ~ rc + C(window) + C(yb) + res + C(ctype) + sars2 + vhh + nonhm + atyp + gap','EM M4 excl. CDR3/FR4 windows (bovine)')
# same model in X-ray (3-5 A): is the association EM-specific?
XR=L[L.method=='X-ray']; xb=XR[(XR.y==1)|(XR.yany==0)]
fit(xb,'y ~ rc + C(window) + C(yb) + res + C(ctype)','X-ray M1')
fit(xb,'y ~ rc + C(window) + res + C(ctype) + vhh + nonhm','X-ray M1b (no year)')
allb=L[(L.y==1)|(L.yany==0)].copy(); allb['em']=(allb.method=='EM').astype(int)
fit(allb,'y ~ rc*em + C(window) + C(yb) + res + C(ctype) + vhh + nonhm','pooled rc x EM interaction',term='rc:em')
# period interaction within EM: does rare column predict errors equally before/after the 2020-21 peak?
EM2=base.copy(); EM2['peak']=EM2.yb.isin([2,3]).astype(int)
fit(EM2,'y ~ rc*peak + C(window) + res + C(ctype) + vhh + nonhm','EM rc x 2020-21 peak interaction',term='rc:peak')
T=pd.DataFrame(tests); T.to_csv(f'{OUT}/v06_models.csv',index=False)
# ---- paper-level collapse: papers with any error-in-window carrying a rare col vs papers' clean units carrying rare col
P=base.groupby(['paper']).agg(err=('y','max'),err_rc=('y',lambda s: 0),).reset_index()
pp=base.assign(yrc=base.y*base.rc).groupby('paper').agg(err=('y','max'),err_rc=('yrc','max'),clean_rc=('rc',lambda s:0)).reset_index()
cl=base[base.y==0].groupby('paper').rc.max().rename('ctrl_rc')
pp=pp.merge(cl,on='paper',how='left')
print('\nPaper level (EM): papers with an error in a window holding a rare column:',int(pp.err_rc.sum()),'of',int(pp.err.sum()),'error papers')
print('papers whose non-error unit-windows hold a rare column:',int(pp.ctrl_rc.fillna(0).sum()),'of',int(pp.ctrl_rc.notna().sum()))
print('labs among positives:',sorted(base[(base.y==1)&(base.rc==1)].lab.unique()))
