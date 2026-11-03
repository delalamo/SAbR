"""ANARCI-HMM-free check of the truncation flag: semi-global alignment of each unit's V-region SEQRES N-terminus to its assigned
germline V (IMGT-gapped germline from ANARCI's germline table); the IMGT position of the germline residue aligned to the first
SEQRES residue gives an independent 'sequence starts at IMGT k'. Then re-run the M2 test with this flag."""
import pandas as pd, numpy as np, warnings, statsmodels.formula.api as smf
from Bio import Align
from anarci.germlines import all_germlines as G
from scipy import stats
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
T='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
U=pd.read_csv(f'{T}/units.csv.gz',low_memory=False).copy()
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False,usecols=['pdb','chain','seq'])
U=U.merge(ch,on=['pdb','chain'],how='left')
EM=U.method_u=='EM'
A=EM&U.err_first_u&U.modes_u.fillna('').str.contains('A_FR1Nterm')
Cm=EM&U.clean&U.ctype.isin(U[A].ctype.unique())
d=U[A|Cm].copy(); d['y']=A[A|Cm].astype(int)
al=Align.PairwiseAligner(); al.mode='global'; al.open_gap_score=-6; al.extend_gap_score=-1
al.target_end_gap_score=0.0; al.query_end_gap_score=0.0
from Bio.Align import substitution_matrices
al.substitution_matrix=substitution_matrices.load('BLOSUM62')
def germ(ct,gene,species):
    for sp in ([species] if isinstance(species,str) else [])+['human','mouse']:
        g=G['V'].get(ct,{}).get(sp,{})
        if gene in g: return g[gene]
    return None
out=[]
for i,r in d.iterrows():
    gs=germ(r.ctype,r.q_v_gene_hm,r.q_species_hm)
    q=r.vseq if isinstance(r.vseq,str) else None
    if gs is None or q is None: out.append(np.nan); continue
    imgt=[k+1 for k,a in enumerate(gs) if a!='-']; gseq=gs.replace('-','')
    # align first 40 V residues of SEQRES V-region to the first 60 germline residues
    a=al.align(gseq[:60],q[:40])[0]
    tgt,qry=a.aligned  # blocks
    first_q_block=qry[0]; first_t_block=tgt[0]
    # germline residue aligned to the first aligned query residue; if query residue 0 is unaligned (insertion) treat as 1
    if first_q_block[0]>0: out.append(1); continue
    out.append(imgt[first_t_block[0]])
d['gl_first_imgt']=out
d['gl_trunc']=(d.gl_first_imgt>1).astype(float); d.loc[d.gl_first_imgt.isna(),'gl_trunc']=np.nan
print('agreement ANARCI-flag vs germline-alignment flag:'); print(pd.crosstab(d.seq_nt_trunc,d.gl_trunc,dropna=False))
print('cases:',d[d.y==1].gl_trunc.mean(),d[d.y==1].gl_trunc.sum(),'ctrl:',d[d.y==0].gl_trunc.mean())
d['yb']=pd.cut(d.dep_year,[0,2018,2019,2020,2021,2022,2023,2100]).astype(str)
d['nonhm']=(~d.q_species_all.isin(['human','mouse'])).astype(int); d['vhh_i']=d.vhh_u.fillna(0).astype(int); d['sars_i']=d.sars2_u.astype(int)
dd=d.dropna(subset=['gl_trunc'])
f=smf.logit('y ~ gl_trunc + C(yb) + res_med + C(ctype) + vhh_i + nonhm + sars_i',data=dd).fit(disp=0)
print('M2 germline-alignment truncation: OR %.2f (%.2f-%.2f) p %.2g n_err %d n_ctrl %d'%(np.exp(f.params['gl_trunc']),*np.exp(f.conf_int().loc['gl_trunc']),f.pvalues['gl_trunc'],dd.y.sum(),(dd.y==0).sum()))
# restricted definition used in the claim: starts at IMGT 2-8
d['trunc28']=d.seq_first_imgt.between(2,8)
t=pd.crosstab(d.y,d.trunc28); print('IMGT 2-8 definition:',t.values.tolist(),stats.fisher_exact(t.values))
disc=d[(d.seq_nt_trunc.astype(float)!=d.gl_trunc)&d.gl_trunc.notna()]
print('discordant examples'); print(disc[['pdb','chain','ctype','y','seq_first_imgt','gl_first_imgt','q_v_gene_hm']].assign(nt=disc.vseq.str[:12]).head(25).to_string())
d[['unit','pdb','chain','ctype','y','seq_first_imgt','gl_first_imgt']].to_csv('v06_gl_trunc.csv.gz',index=False)
