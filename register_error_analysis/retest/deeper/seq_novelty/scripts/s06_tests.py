"""Error vs clean units (cryo-EM), per mode, for sequence/germline novelty features.
Adjusted logistic: case ~ feature + C(year bin) + resolution + C(chain type); BH across all tests."""
import numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
U=pd.read_csv(f'{OUT}/units.csv.gz',low_memory=False)
U['yb']=pd.cut(U.dep_year,[0,2018,2019,2020,2021,2022,2023,2100],labels=['<=2018','2019','2020','2021','2022','2023','>=2024']).astype(str)
lg=lambda x: np.log10(1+x)
F=pd.DataFrame(index=U.index)
# ---- features
F['nn_any_V']=U.nn_any_V; F['nn_diff_V']=U.nn_diff_V; F['nn_diff_FW']=U.nn_diff_FW; F['nn_any_FW']=U.nn_any_FW
F['exact_prior']=U.exact_prior.astype(float); F['prior_same_ab']=(U.n_prior_same_ab>0).astype(float)
F['has_relative_ge90']=(U.n_prior_ge90>0).astype(float); F['log_n_relatives_ge80']=lg(U.n_prior_ge80)
for k in ['V','J','VJ','L123','VL12','VJL3','JL3','L3']:
    F[f'first_{k}']=(U[f'gprior_{k}']==0).astype(float); F[f'log_prior_{k}']=lg(U[f'gprior_{k}'])
F['v_id_hm']=U.q_v_id_hm; F['j_id_hm']=U.q_j_id_hm
mut=[c for c in U.columns if c.startswith('mut_')]
F['mut_total']=U[mut].sum(1)
for c in mut: F[c]=U[c]
for c in ['rare_FW','rare_V','rare_vernier','rare_canonical','rare_interface']: F[c]=U[c]; F[c+'_any']=(U[c]>0).astype(float)
for w in ['A','F','B','C','D','E','FR4','CDR3']:
    F[f'rare_win_{w}_any']=(U[f'rare_win_{w}']>0).astype(float); F[f'rareins_win_{w}_any']=(U[f'rareins_win_{w}']>0).astype(float)
for r in ['NT','FR1','CDR1','FR2','CDR2','FR3','CDR3','FR4']:
    F[f'nn_any_{r}']=U[f'nn_any_{r}']; F[f'nn_diff_{r}']=U[f'nn_diff_{r}']
F['nn_diff_CDR3_samelen']=U.nn_diff_CDR3_samelen; F['log_prior_samelen_cdr3']=lg(U.n_prior_samelen_cdr3)
F['cdr3_len']=U.q_len_CDR3; F['cdr1_len_ne_gl']=(U.cdr1_len_vs_gl.fillna(0)!=0).astype(float); F['cdr2_len_ne_gl']=(U.cdr2_len_vs_gl.fillna(0)!=0).astype(float)
F['humanised_sig']=U.humanised_sig.astype(float); F['humanised_kw']=U.humanised_kw.astype(float)
F['nonhuman_mouse_species']=(~U.q_species_all.isin(['human','mouse'])).astype(float)
F['vhh_hallmarks_ge2']=(U.vhh_hallmarks.fillna(0)>=2).astype(float)
F['scfv_or_multidomain']=(U.scfv_linker|U.engineered_multidomain.fillna(False).astype(bool)).astype(float)
F['seq_nt_trunc']=U.seq_nt_trunc.astype(float); F['seq_fr4_trunc']=U.seq_fr4_trunc.astype(float)
BIN={c for c in F if set(F[c].dropna().unique())<={0.0,1.0}}
GLOBAL=['nn_any_V','nn_diff_V','nn_diff_FW','exact_prior','prior_same_ab','has_relative_ge90','log_n_relatives_ge80','first_V','first_J','first_VJ','first_L123',
        'log_prior_V','log_prior_VJ','v_id_hm','mut_total','rare_FW_any','rare_vernier_any','rare_canonical_any','rare_interface_any','humanised_sig','nonhuman_mouse_species','scfv_or_multidomain']
LOC={'A_FR1Nterm':['seq_nt_trunc','nn_any_NT','nn_diff_NT','nn_diff_FR1','rare_win_A_any','rareins_win_A_any','mut_FR1'],
     'F_CDR1':['log_prior_VL12','log_prior_L123','nn_any_CDR1','nn_diff_CDR1','rare_win_F_any','rareins_win_F_any','cdr1_len_ne_gl','first_VL12','mut_CDR1'],
     'B_CDR2FR3':['log_prior_VL12','log_prior_L123','nn_any_CDR2','nn_diff_CDR2','rare_win_B_any','rareins_win_B_any','cdr2_len_ne_gl','first_VL12','mut_CDR2','mut_FR3'],
     'C_FR3':['nn_any_FR3','nn_diff_FR3','rare_win_C_any','rareins_win_C_any','mut_FR3'],
     'Dplus_CDR3short':['seq_fr4_trunc','log_prior_JL3','cdr3_len','nn_any_CDR3','nn_diff_CDR3_samelen','log_prior_samelen_cdr3','first_L3','first_JL3','first_VJL3','nn_diff_FR4','rare_win_D_any','rare_win_FR4_any','rare_win_CDR3_any','mut_FR4','j_id_hm'],
     'Dminus_CDR3long':None,'D_all':None,
     'E_FR4bulge':['seq_fr4_trunc','cdr3_len','nn_any_CDR3','nn_diff_CDR3_samelen','first_JL3','first_J','nn_diff_FR4','nn_any_FR4','rare_win_E_any','rare_win_FR4_any','rareins_win_E_any','mut_FR4','j_id_hm']}
LOC['Dminus_CDR3long']=LOC['Dplus_CDR3short']; LOC['D_all']=LOC['Dplus_CDR3short']
EM=U.method_u=='EM'
def case_mask(m):
    if m=='ANY': return U.err_first_u
    if m=='D_all': return U.err_first_u&U.modes_u.str.contains('Dplus|Dminus|D0')
    return U.err_first_u&U.modes_u.str.contains(m)
rows=[]
U['nonhm']=(~U.q_species_all.isin(['human','mouse'])).astype(int)
U['vhh_i']=U.vhh_u.fillna(0).astype(int); U['sars_i']=U.sars2_u.astype(int)
def fit_or(d,feat,cov):
    try:
        fit=smf.logit('y ~ f'+cov,data=d).fit(disp=0,maxiter=200)
        OR=np.exp(fit.params['f']); lo,hi=np.exp(fit.conf_int().loc['f']); p=fit.pvalues['f']
        if not np.isfinite(p) or hi/lo>1e4: raise ValueError
        return OR,lo,hi,p
    except Exception:
        return np.nan,np.nan,np.nan,np.nan
def test(m,feat,cases,ctrl):
    d=pd.concat([F[[feat]],U[['yb','res_med','ctype','sars_i','vhh_i','nonhm','n_prior_same_ab']]],axis=1)[cases|ctrl].copy()
    d['y']=cases[cases|ctrl].astype(int); d=d.dropna(subset=[feat,'res_med'])
    x1=d.loc[d.y==1,feat]; x0=d.loc[d.y==0,feat]
    if len(x1)<5: return None
    isb=feat in BIN
    if isb:
        if d[feat].sum()<3 or (d[feat]==0).sum()<3: return None
        t=pd.crosstab(d.y,d[feat]).reindex(index=[0,1],columns=[0.0,1.0],fill_value=0)
        p_raw=stats.fisher_exact(t.values)[1]; e1,e0=x1.mean(),x0.mean()
    else:
        p_raw=stats.mannwhitneyu(x1,x0).pvalue; e1,e0=x1.median(),x0.median()
    sd=d[feat].std() if not isb else 1.0
    d['f']=(d[feat]-(0 if isb else d[feat].mean()))/(sd if sd>0 else 1)
    ctc=' + C(ctype)' if d.ctype.nunique()>1 else ''
    c1=' + C(yb) + res_med'+ctc
    c2=c1+(' + vhh_i' if d.vhh_i.nunique()>1 and feat!='vhh_hallmarks_ge2' else '')+(' + nonhm' if d.nonhm.nunique()>1 and feat!='nonhuman_mouse_species' else '')+' + sars_i'
    r=dict(mode=m,feature=feat,binary=isb,n_err=len(x1),n_ctrl=len(x0),err_value=e1,ctrl_value=e0,err_mean=x1.mean(),ctrl_mean=x0.mean(),p_raw=p_raw,
           per='present vs absent' if isb else f'per SD ({sd:.3g})')
    r['OR_adj'],r['lo'],r['hi'],r['p_adj']=fit_or(d,feat,c1)
    r['OR_adj2'],r['lo2'],r['hi2'],r['p_adj2']=fit_or(d,feat,c2)
    # M3: first structure of this antibody, human/mouse, non-nanobody
    d3=d[(d.n_prior_same_ab==0)&(d.nonhm==0)&(d.vhh_i==0)]
    if d3.y.sum()>=5 and d3.f.nunique()>1:
        r['n_err3']=int(d3.y.sum()); r['n_ctrl3']=int((d3.y==0).sum())
        r['OR_adj3'],r['lo3'],r['hi3'],r['p_adj3']=fit_or(d3,feat,' + C(yb) + res_med'+(' + C(ctype)' if d3.ctype.nunique()>1 else '')+' + sars_i')
    r['adj']='M1: year bin + resolution'+ctc+'; M2: + nanobody + non-human/mouse species + SARS-CoV-2; M3: M2 restricted to first structure of the antibody, human/mouse non-nanobody'
    return r
for m in ['ANY','A_FR1Nterm','F_CDR1','B_CDR2FR3','C_FR3','Dplus_CDR3short','Dminus_CDR3long','D_all','E_FR4bulge']:
    cases=EM&case_mask(m); ctrl=EM&U.clean
    if m!='ANY':
        cts=U[cases].ctype.unique(); ctrl=ctrl&U.ctype.isin(cts)
    feats=GLOBAL+(LOC.get(m) or [])
    if m=='ANY': feats=GLOBAL+sorted({f for v in LOC.values() if v for f in v})+['vhh_hallmarks_ge2','humanised_kw']
    for f in dict.fromkeys(feats):
        r=test(m,f,cases,ctrl)
        if r: rows.append(r)
T=pd.DataFrame(rows)
for k in ['','2','3']:
    if 'p_adj'+k not in T: continue
    ok=T['p_adj'+k].notna()
    T.loc[ok,'q_adj'+k]=multipletests(T.loc[ok,'p_adj'+k],method='fdr_bh')[1]
T['q_raw']=multipletests(T.p_raw.fillna(1),method='fdr_bh')[1]
T.to_csv(f'{OUT}/tests_em_units.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
print(T.sort_values('p_adj')[['mode','feature','n_err','n_ctrl','err_mean','ctrl_mean','OR_adj','p_adj','q_adj','OR_adj2','q_adj2','n_err3','OR_adj3','q_adj3']].head(70).round(4).to_string())
