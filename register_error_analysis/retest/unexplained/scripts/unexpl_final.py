import json, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
warnings.filterwarnings('ignore'); pd.set_option('display.width',250)
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
R=pd.read_pickle(f'{SP}/unexpl/runs.pkl'); U=pd.read_pickle(f'{SP}/unexpl/unexplained_final.pkl'); M=pd.read_pickle(f'{SP}/unexpl/master4.pkl')
bnab=set(M[M.chain_desc.str.upper().str.contains(r'VRC0?3\b|VRC03|3BNC|8ANC195|VRC40|N49P|VRC-?PG|VRC33|NIH45')].seq.str[:105])
pat=dict(zip(zip(U.pdb,U.chain,U.imgt0),U.pattern)); atyp2=dict(zip(zip(M.pdb,M.chain),M.L2_atyp))
def cat(r):
    if r['cat']=='template': return '1a copied from an earlier high-res template (HIV FR3-insertion bnAbs; ambiguous)' if r.seqkey in bnab else '1b copied from an earlier high-res template'
    if r['cat']=='earlier 3-5A other paper': return '2 copied from an earlier 3–5 Å structure (other paper)'
    if r['cat']=='N-term/gap': return '3 chain end or next to an unbuilt loop'
    if r['cat']=='FR3 convention': return '4 FR3 insertion-convention region (possible artifact)'
    p=pat.get((r.pdb,r.chain,r.imgt0),'other')
    if p.startswith('CDR3→FR4') or p.startswith('CDR3 internal'): return '5 CDR3–FR4 junction: CDR3 built one residue short/long, FR4 anchor displaced'
    if p.startswith('FR4: SAbR skips'): return '6 FR4 bulge after the W/F-G motif (IMGT 120)'
    if p.startswith('CDR2') and atyp2.get((r.pdb,r.chain))==1: return '7 CDR2 boundary, atypical CDR2 length'
    if p.startswith('CDR2') or p.startswith('CDR1'): return '8 CDR1/CDR2 boundary, typical length'
    return '9 FR3 / other (still unexplained)'
R['origin']=R.apply(cat,axis=1)
ch=R.groupby(['pdb','chain','era']).origin.min().reset_index()
t=pd.crosstab(ch.origin,ch.era); p=(100*pd.crosstab(ch.origin,ch.era,normalize='columns')).round(0)
print(pd.concat([t,p.add_suffix(' %')],axis=1).to_string())
print('\nrepeated in another entry of the same paper (any category): %.0f%% of runs'%(100*R.same_paper_other_entry.mean()))
# AlphaFold starting model vs CDR3-FR4 junction errors, cryo-EM 2022-26
E=M[(M.method=='EM')&(M.year>=2022)].dropna(subset=['lq_z']).copy(); E['af']=(E.irm=='AlphaFold model').astype(int); E['exp']=(E.irm=='experimental (PDB) model').astype(int)
for y,lab in [('cdr3fr4','CDR3–FR4 junction'),('flag','any disagreement')]:
    m=smf.logit(f"{y} ~ af + exp + C(rb) + C(ctype) + lq_z",data=E).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(E.pdb)[0]})
    print(f'cryo-EM 2022–26, {lab}: AlphaFold starting model OR %.2f [%.2f-%.2f] p=%.2g; experimental template OR %.2f [%.2f-%.2f] (vs none recorded; n AF=%d)'%(np.exp(m.params.af),*np.exp(m.conf_int().loc['af']),m.pvalues.af,np.exp(m.params.exp),*np.exp(m.conf_int().loc['exp']),E.af.sum()))
ch.to_csv(f'{SP}/unexpl/flag_origin_v2.csv',index=False); R.to_csv(f'{SP}/unexpl/runs_origin_v2.csv',index=False)
