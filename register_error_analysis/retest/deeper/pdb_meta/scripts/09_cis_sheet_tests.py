"""Tests on canonical cis-Pro and sheet assignment, EM error chains vs clean chains (adjusted: year bin, resolution; cluster by paper)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature, bh, ybin
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
D=pd.read_csv(f'{OUT}/chain_cis_sheet.csv.gz')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
D=D.merge(pd.read_csv(f'{OUT}/chain_novel_v.csv.gz'),on=['pdb','chain'],how='left')
D=D.merge(pd.read_csv(f'{OUT}/chain_features_rcsb.csv.gz',usecols=['pdb','chain','chain_q']),on=['pdb','chain'],how='left')
D=D.merge(pd.read_csv(f'{SP}/chains.csv.gz',usecols=['pdb','chain','irm','fit','vhh']),on=['pdb','chain'],how='left')
D=D[(D.method=='EM')&((D.err_first)|(~D.err_any))].copy()
D['y']=D.err_first.astype(int); D['cluster']=D.paper.fillna(D.pdb); D['ybin']=ybin(pd.to_datetime(D.deposit).dt.year.values)
modes=D.modes.fillna('')
D['mode_A']=modes.str.contains('A_FR1'); D['mode_D']=modes.str.contains('D|E_FR4')
pro=D.v_pro_pos.fillna('').str.split(';'); cisp=D.v_cis_pos.fillna('').str.split(';')
def has(lst,p): return p in lst
D['pro8']=[has(a,'8') for a in pro]; D['cis8']=[has(a,'8PRO') for a in cisp]
D['pro115']=[has(a,'115') for a in pro]; D['cis115']=[has(a,'115PRO') for a in cisp]
D['const_cis_any']=(D.const_cis_n>0).astype(float).where((D.n_const_modeled>60)&(D.const_pro_n>0))
res=[]
def run(sub,feat,kind,label,cov=['C(ybin)','res']):
    x=sub.copy(); 
    if kind=='bin': x[feat]=x[feat].astype(float)
    r=fit_feature(x,feat,kind,cov); r['test']=label; res.append(r)
K=D[D.ctype=='K']
run(K[K.pro8],'cis8','bin','kappa Pro8 (IMGT 8) modelled cis | all error chains')
run(K[K.pro8&~(K.mode_A&(K.y==1))],'cis8','bin','kappa Pro8 cis | error chains WITHOUT an FR1 (mode A) error')
run(K[K.pro115],'cis115','bin','kappa Pro115 (CDR-L3, Kabat 95) cis | all error chains')
run(K[K.pro115&~(K.mode_D&(K.y==1))],'cis115','bin','kappa Pro115 cis | error chains WITHOUT CDR3/FR4 (D/E) error')
for ct in ['H','K','L']:
    run(D[D.ctype==ct],'const_cis_any','bin',f'{ct}: any cis peptide in the constant domain (constant domain modelled, Pro present)')
    run(D[D.ctype==ct],'v_sheet_away','cont',f'{ct}: sheet fraction in V domain >5 residues from any error run')
    run(D[D.ctype==ct],'const_sheet','cont',f'{ct}: sheet fraction in constant domain')
run(D,'const_cis_any','bin','all: any cis in constant domain',['C(ybin)','res','C(ctype)'])
run(D,'v_sheet_away','cont','all: V-domain sheet fraction away from error run',['C(ybin)','res','C(ctype)'])
run(D,'const_sheet','cont','all: constant-domain sheet fraction',['C(ybin)','res','C(ctype)'])
N=D[D.novel_v==True]; KN=N[N.ctype=='K']
run(KN[KN.pro8],'cis8','bin','NOVEL V only: kappa Pro8 cis')
run(KN[KN.pro8&~(KN.mode_A&(KN.y==1))],'cis8','bin','NOVEL V only: kappa Pro8 cis, error chains without FR1 error')
run(KN[KN.pro115&~(KN.mode_D&(KN.y==1))],'cis115','bin','NOVEL V only: kappa Pro115 cis, error chains without CDR3/FR4 error')
run(K[K.pro8&~(K.mode_A&(K.y==1))],'cis8','bin','kappa Pro8 cis (no FR1 error), adjusted also for chain Q-score and starting-model type',['C(ybin)','res','chain_q','C(irm)'])
run(N,'v_sheet_away','cont','NOVEL V only: V-domain sheet away from error run',['C(ybin)','res','C(ctype)'])
run(D,'v_sheet_away','cont','all: V sheet away from error, adjusted also for chain Q-score and starting-model type',['C(ybin)','res','C(ctype)','chain_q','C(irm)'])
run(N,'const_cis_any','bin','NOVEL V only: any cis in constant domain',['C(ybin)','res','C(ctype)'])
run(D,'novel_v','bin','error chains are novel V regions (no identical V earlier in PDB)',['C(ybin)','res','C(ctype)'])
R=bh(pd.DataFrame(res))
R=R[['test','n_err','n_ctrl','err_val','ctrl_val','unit','OR','lo','hi','p','q']]
R.to_csv(f'{OUT}/cis_sheet_tests_em.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',80)
print(R.to_string(index=False,float_format=lambda x:f'{x:.3g}'))
