"""Reproduce the constant-domain / Pro115 cis numbers from the investigator's chain table and list the error chains."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature, ybin
T='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_constant_and_L3_confounding'
D=pd.read_csv(f'{T}/chain_cis_sheet.csv.gz')
D=D.merge(pd.read_csv(f'{T}/chain_novel_v.csv.gz'),on=['pdb','chain'],how='left')
C=pd.read_csv(f'{SP}/chains.csv.gz',usecols=['pdb','chain','irm','fit','vhh','sars2','last_author','title','chain_desc','role','species','lq','lq_z'])
D=D.merge(C,on=['pdb','chain'],how='left')
D=D[(D.method=='EM')&((D.err_first)|(~D.err_any))].copy()
D['y']=D.err_first.astype(int); D['cluster']=D.paper.fillna(D.pdb); D['year']=pd.to_datetime(D.deposit).dt.year
D['ybin']=ybin(D.year.values)
modes=D.modes.fillna('')
D['mode_A']=modes.str.contains('A_FR1')
D['mode_D_inv']=modes.str.contains('D|E_FR4')            # investigator's regex (matches any 'CDR' mode too)
D['mode_D_true']=modes.str.contains('Dplus|Dminus|E_FR4') # intended: CDR3/FR4 modes only
pro=D.v_pro_pos.fillna('').str.split(';'); cisp=D.v_cis_pos.fillna('').str.split(';')
D['pro8']=[('8' in a) for a in pro]; D['cis8']=[('8PRO' in a) for a in cisp]
D['pro115']=[('115' in a) for a in pro]; D['cis115']=[('115PRO' in a) for a in cisp]
D['const_ok']=(D.n_const_modeled>60)&(D.const_pro_n>0)
D['const_cis_any']=(D.const_cis_n>0).astype(float).where(D.const_ok)
D.to_csv(f'{OUT}/frame.csv.gz',index=False)
for ct in ['K','H']:
    s=D[(D.ctype==ct)&D.const_ok]
    print(ct,'const modelled: err',s.y.sum(),'ctrl',(s.y==0).sum(),'| cis err',s[s.y==1].const_cis_any.mean().round(3),'ctrl',s[s.y==0].const_cis_any.mean().round(3))
    r=fit_feature(s,'const_cis_any','bin',['C(ybin)','res']); print('  investigator model:',{k:r.get(k) for k in ['OR','p','err_val','ctrl_val']})
    e=s[s.y==1]; print('  error chains: papers',e.cluster.nunique(),'entries',e.pdb.nunique(),'labs',e.last_author.nunique())
    print(e.groupby('cluster').agg(n=('pdb','size'),entries=('pdb','nunique'),cis=('const_cis_any','sum'),lab=('last_author','first'),yr=('year','min'),res=('res','median'),modes=('modes',lambda x:'|'.join(sorted(set(x))))).sort_values('n',ascending=False).to_string())
    print('  fraction of all EM error chains of this type with constant domain modelled:',
          round(D[(D.ctype==ct)&(D.y==1)].const_ok.mean(),3),'vs ctrl',round(D[(D.ctype==ct)&(D.y==0)].const_ok.mean(),3))
# Pro115
K=D[(D.ctype=='K')&D.pro115]
for lab,mask in [('investigator (excl. any CDR-containing mode)',~(K.mode_D_inv&(K.y==1))),('intended (excl. CDR3/FR4 modes only)',~(K.mode_D_true&(K.y==1)))]:
    s=K[mask]; r=fit_feature(s,'cis115','bin',['C(ybin)','res'])
    print('Pro115',lab,{k:r.get(k) for k in ['OR','p','err_val','ctrl_val']})
    e=s[s.y==1]; print('   modes of error chains:',e.modes.value_counts().to_dict()); print('   papers',e.cluster.nunique(),'entries',e.pdb.nunique())
    print(e.groupby('cluster').agg(n=('pdb','size'),cis=('cis115','sum'),lab=('last_author','first'),yr=('year','min'),res=('res','median')).to_string())
