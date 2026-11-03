"""Context numbers for the literature comparison:
 1. canonical-pull fraction by starting-model type and period;
 2. ImmuneBuilder/ABodyBuilder2 training-filter pass (X-ray or EM <=3.5 A, no internal missing residues, released before
    the 31 Jul 2021 extraction date) for error vs clean chains;
 3. PDB-wide composition shift: new unique V domains per year, share known only from cryo-EM / without any X-ray <=3.0 A
    structure (i.e. invisible to SAbDab-style curated sets), share nanobody-like (H without pairing not tracked; use VHH flag in study set);
 4. germline coverage: unique V domains per V gene (human/mouse) in the curated reference."""
import pickle, numpy as np, pandas as pd
from scipy import stats
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/literature_dean/tables'
log=open(f'{OUT}/context.log','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); log.write(s+'\n')
A=pd.read_csv(f'{OUT}/apparent_length_error_units.csv')
c=pd.read_pickle(f'{SP}/deeper/cache/literature_dean/chains_aug.pkl')
u=c.groupby('unit').agg(irm=('irm','first'),fit=('fit','first')).reset_index()
A=A.merge(u,on='unit',how='left')
A['period']=pd.cut(A.year,[0,2019,2021,2100],labels=['<=2019','2020-21','>=2022']).astype(str)
P('1. Canonical pull (apparent length more common than true) by starting model / period / fit:')
for col in ['irm','period','fit']:
    t=A.groupby(col).agg(n=('more_common','size'),obs=('more_common','sum'),exp=('p_exp','sum'))
    P(t.round(1).to_string())
# 2. ABB2 filter
c['rel']=pd.to_datetime(c.release)
c['abb2_pass']=(c.res<=3.5)&(c.int_gap==0)
c['abb2_window']=c.rel<=pd.Timestamp('2021-07-31')
for m in ['EM','X-ray']:
    d=c[(c.method==m)&c.abb2_window]
    e=d[d.err_first]; k=d[~d.err_any]
    tab=[[e.abb2_pass.sum(),(~e.abb2_pass).sum()],[k.abb2_pass.sum(),(~k.abb2_pass).sum()]]
    orr,p=stats.fisher_exact(tab)
    P(f'2. {m}, released <=2021-07-31: chains passing ABB2-style filter (<=3.5 A, no internal V-domain gap): error {e.abb2_pass.mean():.3f} ({int(e.abb2_pass.sum())}/{len(e)}) vs clean {k.abb2_pass.mean():.3f} ({int(k.abb2_pass.sum())}/{len(k)}); OR {orr:.2f} p={p:.2g}')
    P(f'   -> error UNITS with >=1 passing error chain: {e[e.abb2_pass].unit.nunique()} of {e.unit.nunique()}; entries: {e[e.abb2_pass].pdb.nunique()}')
# internal gap alone (by method), all dates
for m in ['EM','X-ray']:
    d=c[c.method==m]; e=d[d.err_first]; k=d[~d.err_any]
    P(f'   internal V-domain gap present: {m} error {(e.int_gap>0).mean():.3f} (n={len(e)}) vs clean {(k.int_gap>0).mean():.3f} (n={len(k)})')
    P(f'   N-terminus not modelled (first ANARCI IMGT pos >2): {m} error {(e.first_imgt>2).mean():.3f} vs clean {(k.first_imgt>2).mean():.3f}')
# 3. PDB-wide composition
R=pd.read_csv(f'{OUT}/ref_vdomains.csv.gz')
R=R[R.first_year.between(2010,2025)]
comp=R.groupby('first_year').agg(new_V=('ctype','size'),frac_no_xray30=('hq30',lambda s:1-s.mean()),frac_any_em=('any_em','mean'),
     frac_lambda=('ctype',lambda s:(s=='L').mean()),frac_heavy=('ctype',lambda s:(s=='H').mean()))
P('\n3. PDB-wide: new unique antibody V domains per first-release year; share with no X-ray <=3.0 A structure ever; share seen in any EM entry')
P(comp.round(3).to_string()); comp.to_csv(f'{OUT}/pdbwide_composition_by_year.csv')
# study set by deposition year
c['dyear']=pd.to_datetime(c.deposit).dt.year
s=c[c.dyear.between(2012,2025)].groupby('dyear').agg(chains=('pdb','size'),frac_em=('method',lambda x:(x=='EM').mean()),
   frac_vhh=('vhh','mean'),frac_sars2=('sars2','mean'),mean_vid=('v_id','mean'),err_rate=('err_first','mean'))
P('\n   study set (3-5 A antibody V chains) by deposition year'); P(s.round(3).to_string()); s.to_csv(f'{OUT}/study_composition_by_year.csv')
# 4. germline coverage (curated reference: unique V domains with X-ray<=3.0 A; human+mouse V genes)
pk=pickle.load(open(f'{SP}/deeper/cache/literature_dean/ref.pkl','rb'))
for lab,sub in [('X-ray<=3.0 A, all years',pk[pk.hq30]),('all PDB V domains',pk)]:
    g=sub.dropna(subset=['v_gene_hm']).assign(gene=lambda d:d.v_gene_hm.str.split('*').str[0]).groupby(['species_hm','gene']).size()
    for sp in ['human','mouse']:
        x=g.loc[sp]
        P(f'4. {lab} {sp}: V genes covered {len(x)}; unique V domains per gene median {x.median():.0f}, IQR {x.quantile(.25):.0f}-{x.quantile(.75):.0f}, top: {x.sort_values(ascending=False).head(5).to_dict()}')
log.close()
