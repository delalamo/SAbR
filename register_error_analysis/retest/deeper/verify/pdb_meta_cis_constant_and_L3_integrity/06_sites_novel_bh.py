"""Canonical-site (Ck FYP / CH1 FPEP) version of the constant-domain test, cis offsets (register check), the novel-V
sub-claim without the dominant paper and with entry all-trans adjustment, and BH across this verifier's tests."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature
from statsmodels.stats.multitest import multipletests
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_constant_and_L3_integrity'
T='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
D=pd.read_csv(f'{OUT}/frame_em.csv.gz')
ecis={}
for l in open(f'{SP}/deeper/cache/pdb_meta/rcsb_meta.jsonl'):
    e=json.loads(l); ecis[e['rcsb_id'].lower()]=(e.get('rcsb_entry_info') or {}).get('cis_peptide_count')
D['other_cis']=D.pdb.map(ecis)-D.n_cis_chain; D['other_alltrans']=(D.other_cis<=0).astype(float)
D=D.merge(pd.read_csv(f'{T}/chain_novel_v.csv.gz')[['pdb','chain','novel_v']],on=['pdb','chain'],how='left')
ZHOU='10.1038/s41422-021-00487-9'
rows=[]
def rec(lbl,S,feat,cov):
    r=fit_feature(S,feat,'bin',cov); d={'test':lbl,**{k:r.get(k) for k in ['n_err','n_ctrl','err_val','ctrl_val','OR','lo','hi','p','note']}}; rows.append(d)
    print(lbl,'|',{k:d[k] for k in ['err_val','ctrl_val','OR','p']})
# canonical sites, both residues modelled
D['site_all_cis']=np.where(D.site_mod_n>0,(D.site_cis_n>0).astype(float),np.nan)
for ct,lbl in [('K','Ck Tyr-Pro (FYP) site modelled'),('H','CH1 Phe-Pro/Glu-Pro (FPEP) sites modelled')]:
    S=D[(D.ctype==ct)]
    rec(f'{lbl}: any canonical site cis',S,'site_all_cis',['C(ybin)','res'])
    rec(f'{lbl}: any canonical site cis, drop Zhou',S[S.cluster!=ZHOU],'site_all_cis',['C(ybin)','res'])
# offsets of constant-domain cis relative to V end (register check)
for ct in 'KH':
    for y in [1,0]:
        s=D[(D.ctype==ct)&D.const_ok&(D.y==y)].const_cis_pos.dropna().astype(str).str.split(';').explode().astype(int)
        print(ct,'error' if y else 'clean','cis offsets from V end (top):',s.value_counts().head(5).to_dict())
# novel V sub-claim
N=D[D.const_ok&(D.novel_v==True)]
rec('novel V: any constant cis (investigator def)',N,'ccis',['C(ybin)','res','C(ctype)'])
rec('novel V: drop Zhou',N[N.cluster!=ZHOU],'ccis',['C(ybin)','res','C(ctype)'])
rec('novel V: + entry all-trans elsewhere',N,'ccis',['C(ybin)','res','C(ctype)','other_alltrans'])
rec('novel V: Ck only, drop Zhou',N[(N.cluster!=ZHOU)&(N.ctype=='K')],'ccis',['C(ybin)','res'])
pd.DataFrame(rows).to_csv(f'{OUT}/06_sites_novel.csv',index=False)
# which non-Zhou Ck error chains lack cis, and is their entry all-trans?
e=D[(D.y==1)&D.const_ok&(D.cluster!=ZHOU)&(D.ctype=='K')]
print(e[['pdb','chain','last_author','modes','n_cis_chain','n_cis_const','other_cis','res','deposit']].to_string())
# BH across this verifier's tests
A=pd.concat([pd.read_csv(f'{OUT}/04_robust.csv').assign(src='04'),pd.read_csv(f'{OUT}/05_pro115_tests.csv').assign(src='05'),pd.DataFrame(rows).assign(src='06')],ignore_index=True)
ok=A.p.notna(); A.loc[ok,'q_BH_verifier']=multipletests(A.loc[ok,'p'],method='fdr_bh')[1]
A.to_csv(f'{OUT}/all_tests_bh.csv',index=False)
print(A[['src','set','test','err_val','ctrl_val','OR','p','q_BH_verifier']].to_string())
