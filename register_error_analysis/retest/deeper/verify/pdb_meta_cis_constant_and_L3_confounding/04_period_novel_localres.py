"""(a) period strata; (b) novel-V-only constant-domain test with/without the Zhou paper; (c) L3-length / species adjustment
for Pro115; (d) map-intrinsic local resolution (EMDB validation) at the C-kappa Tyr-Pro and kappa Pro115 sites."""
import json, re, sys, warnings, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
ZHOU='10.1038/s41422-021-00487-9'
X=pd.read_csv('03_canonical_sites.csv.gz')
C=pd.read_csv(f'{SP}/deeper/chains.csv.gz',usecols=['pdb','chain','L3'])
X=X.merge(C,on=['pdb','chain'],how='left')
X['period']=np.where(X.year<=2019,'<=2019',np.where(X.year<=2021,'2020-21','>=2022'))
out=[]
def add(label,s,feat,cov=['C(ybin)','res']):
    a=fit_feature(s.assign(x=s[feat].astype(float)),'x','bin',cov)
    out.append({'test':label,'n_err':a['n_err'],'n_err_papers':s[s.y==1].cluster.nunique(),'n_ctrl':a['n_ctrl'],'err':a.get('err_val'),'ctrl':a.get('ctrl_val'),
                'OR':a.get('OR'),'p':a.get('p'),'p_fisher':a.get('p_unadj'),'note':a.get('note')})
CK=X[(X.ctype=='K')&(X.ck_YP_modelled==1)]
P115=X[(X.ctype=='K')&(X.k_P115_present==1)&~(X.mode_D_true&(X.y==1))]
ALLC=X[X.const_ok]
for per in ['<=2019','2020-21','>=2022']:
    for nm,s in [('C-kappa Tyr-Pro cis | modelled',CK),('kappa Pro115 cis (no CDR3/FR4 error)',P115),('any constant cis (pooled)',ALLC)]:
        cov=['C(ybin)','res']+(['C(ctype)'] if 'pooled' in nm else [])
        add(f'{nm} | {per}',s[s.period==per],'cis' if False else {'C-kappa Tyr-Pro cis | modelled':'ck_YP_cis','kappa Pro115 cis (no CDR3/FR4 error)':'k_P115_cis','any constant cis (pooled)':'const_cis_any'}[nm],cov)
        add(f'{nm} | {per} | drop Zhou',s[(s.period==per)&(s.cluster!=ZHOU)],{'C-kappa Tyr-Pro cis | modelled':'ck_YP_cis','kappa Pro115 cis (no CDR3/FR4 error)':'k_P115_cis','any constant cis (pooled)':'const_cis_any'}[nm],cov)
N=ALLC[ALLC.novel_v==True]
add('NOVEL V only: any constant cis (pooled)',N,'const_cis_any',['C(ybin)','res','C(ctype)'])
add('NOVEL V only: any constant cis (pooled) | drop Zhou',N[N.cluster!=ZHOU],'const_cis_any',['C(ybin)','res','C(ctype)'])
add('kappa Pro115 cis, L3 = 9 only',P115[P115.L3==9],'k_P115_cis')
add('kappa Pro115 cis, L3 = 9 only | drop Zhou',P115[(P115.L3==9)&(P115.cluster!=ZHOU)],'k_P115_cis')
add('kappa Pro115 cis, + species + L3 class',P115.assign(L3c=np.where(P115.L3==9,'9','other')),'k_P115_cis',['C(ybin)','res','C(species)','C(L3c)'])
add('C-kappa Tyr-Pro cis | modelled, + species',CK,'ck_YP_cis',['C(ybin)','res','C(species)'])
add('C-kappa Tyr-Pro cis | modelled, + species | drop Zhou',CK[CK.cluster!=ZHOU],'ck_YP_cis',['C(ybin)','res','C(species)'])
# (d) local resolution at the site from EMDB validation (map-intrinsic)
va={}
for l in open(f'{SP}/deeper/cache/pdb_meta/emdb_va.jsonl'):
    x=json.loads(l)
    if x.get('status')=='ok' and x.get('res_lr'): va[x['pdb']]=x
three={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def site_lr(pdb,chain,motif,offset,after=100):
    x=va.get(pdb)
    if x is None: return np.nan
    rr=[(k,v) for k,v in x['res_lr'] if k.split(':')[0]==chain]
    if not rr: return np.nan
    s=''.join(three.get(k.split(' ')[-1],'X') for k,v in rr)
    for m in re.finditer(motif,s):
        if m.start()>=after:
            v=rr[m.start()+offset][1]; return float(v)-float(x.get('resolution') or np.nan)
    return np.nan
CK=CK.copy(); CK['site_lr_minus_res']=[site_lr(r.pdb,r.chain,'FYP',2) for r in CK.itertuples()]
P=P115.copy()
# Pro115 site: locate by the V-region CDR-L3 context is hard without numbering; use the kappa 'FG[QG]GT' FR4 motif: Pro115 is 3 residues before W? use seq context from SEQRES
def p115_lr(r):
    x=va.get(r.pdb)
    if x is None or not isinstance(r.vseq,str): return np.nan
    m=re.search(r'FG[A-Z]GT',r.vseq)
    if not m: return np.nan
    ctx=r.vseq[m.start()-4:m.start()+5]  # covers Pro115 at ~ -3/-2 positions
    rr=[(k,v) for k,v in x['res_lr'] if k.split(':')[0]==r.chain]
    s=''.join(three.get(k.split(' ')[-1],'X') for k,v in rr)
    i=s.find(ctx)
    if i<0: return np.nan
    vals=[float(v) for k,v in rr[i:i+len(ctx)]]
    return float(np.mean(vals))-float(x.get('resolution') or np.nan)
V=pd.read_csv(f'{SP}/deeper/chains.csv.gz',usecols=['pdb','chain','vseq'])
P['site_lr_minus_res']=[p115_lr(r) for r in P.itertuples()]
lr=[]
for nm,s,f in [('C-kappa Tyr-Pro site',CK,'ck_YP_cis'),('kappa Pro115 site (CDR-L3/FR4 window)',P,'k_P115_cis')]:
    t=s.dropna(subset=['site_lr_minus_res'])
    e=t[t.y==1].site_lr_minus_res; c=t[t.y==0].site_lr_minus_res
    lr.append({'site':nm,'n_err':len(e),'n_ctrl':len(c),'err_median':e.median(),'ctrl_median':c.median(),'p_MW':stats.mannwhitneyu(e,c).pvalue if len(e)>2 else np.nan})
    a=fit_feature(t.assign(x=t[f].astype(float)),'x','bin',['C(ybin)','res','site_lr_minus_res'])
    add(f'{nm}: cis, + local resolution at site (EMDB-VA subset)',t,f,['C(ybin)','res','site_lr_minus_res'])
    add(f'{nm}: cis, EMDB-VA subset without local-res adjustment',t,f,['C(ybin)','res'])
    # does local resolution predict cis in controls?
    cc=t[t.y==0]
    lr[-1]['ctrl_cis_vs_lr_spearman']=stats.spearmanr(cc[f],cc.site_lr_minus_res).correlation if len(cc)>5 else np.nan
R=pd.DataFrame(out); R.to_csv('04_period_novel_localres_tests.csv',index=False)
L=pd.DataFrame(lr); L.to_csv('04_local_resolution_at_sites.csv',index=False)
pd.set_option('display.width',250); print(R.to_string(index=False,float_format=lambda v:f'{v:.3g}')); print(L.to_string(index=False,float_format=lambda v:f'{v:.3g}'))
