"""Is the Pro115 deficit a local consequence after all? For kappa chains with Pro115 (error chains whose runs are not
CDR3/FR4 modes, and clean controls): any SAbR != ANARCI residue (sub-threshold disagreement) in IMGT 105-118, and any cis
peptide displaced to IMGT 113-117 other than 115."""
import json, sys, warnings, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
ZHOU='10.1038/s41422-021-00487-9'
X=pd.read_csv('03_canonical_sites.csv.gz')
P=X[(X.ctype=='K')&(X.k_P115_present==1)&~(X.mode_D_true&(X.y==1))].copy()
want=set(zip(P.pdb,P.chain)); rec={}
for fn in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        r=json.loads(l); k=(r['pdb'],r['chain'])
        if k in want and k not in rec: rec[k]=r.get('residues') or []
def l3dis(k):
    rr=rec.get(k)
    if not rr: return np.nan
    w=[x for x in rr if x[4] is not None and 105<=x[4]<=118]
    if not w: return np.nan
    return int(any((x[4],x[5] or '')!=(x[6],x[7] or '') for x in w))
P['l3_disagree']=[l3dis((r.pdb,r.chain)) for r in P.itertuples()]
P['cis_displaced']=[int(any(t.endswith('PRO') is False and t[:3].rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ') in ('113','114','116','117') for t in str(s).split(';') if t and t!='nan')) for s in P.v_cis_pos]
P['any_v_cis']=P.v_cis_n>0
out=[]
for lab,s in [('all',P),('drop Zhou',P[P.cluster!=ZHOU])]:
    e=s[s.y==1]; c=s[s.y==0]
    out.append({'set':lab,'n_err':len(e),'err_L3_subthreshold_disagree':f'{int(e.l3_disagree.sum())}/{e.l3_disagree.notna().sum()}',
                'ctrl_L3_subthreshold_disagree':f'{int(c.l3_disagree.sum())}/{c.l3_disagree.notna().sum()}',
                'err_cis_displaced_113-117':int(e.cis_displaced.sum()),'ctrl_cis_displaced_rate':round(c.cis_displaced.mean(),4),
                'err_any_V_cis':f'{int(e.any_v_cis.sum())}/{len(e)}','ctrl_any_V_cis':f'{c.any_v_cis.mean():.3f}'})
    # restrict to chains with complete agreement over IMGT 105-118
    t=s[s.l3_disagree==0]
    a=fit_feature(t.assign(x=t.k_P115_cis.astype(float)),'x','bin',['C(ybin)','res'])
    out[-1].update({'Pro115_cis_err_if_L3_agree':a.get('err_val'),'Pro115_cis_ctrl_if_L3_agree':a.get('ctrl_val'),'OR':a.get('OR'),'p':a.get('p')})
R=pd.DataFrame(out); R.to_csv('05_L3_local_disagreement.csv',index=False)
pd.set_option('display.width',250); print(R.T.to_string())
print(P[P.y==1][['pdb','chain','cluster','modes','l3_disagree','v_cis_pos','k_P115_cis']].to_string())
