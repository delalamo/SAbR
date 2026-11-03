"""Synthesis: consolidate the detector-artifact / copy flags found by the lenses and their verifiers into one
run-level ledger, and count how many first-occurrence error units remain once they are removed.
Sources (all read-only):
  pdb_meta/tables/amyloid_fibril_chains.csv                    AL amyloid fibrils (non-Ig fold)
  case_files_ABCF/tables/artifact_runs.csv                      FR3/FR1 indel gap-placement artifacts (confirmed)
  verify/case_files_ABCF_FR3_INDEL_C_*                          extra C artifacts: 8c8t, 7m3l, 9yfu (probable), 7y0o
  verify/case_files_ABCF_FR3START_INDEL_B_confounding           B gap-placement artifacts: 8bw0, 8r4d/8r4b, 9pyh, 8snc; 9om5 NCS variant
  verify/seq_novelty_SN3_rare_column_in_window_confounding      insertion artifacts: 7cws, 8fsj, 7nk2 (+8c8t, 8bw0, 8r4b)
  case_files_DE/tables/independent_check_DE_units.csv           D/E runs not confirmed by framework superposition
  case_files_ABCF/tables/near_copy_matches.csv (+8xlk<-7te9)    near-identical copies missed by the exact-copy rule
"""
import pandas as pd, numpy as np
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
D='/home/user/SAbR/register_error_analysis/retest/deeper'
runs=pd.read_csv(f'{SP}/runs_first_occurrence.csv')
ch=pd.read_csv(f'{SP}/chains.csv.gz',usecols=['pdb','chain','unit','method','err_first'])
runs=runs.merge(ch[['pdb','chain','unit']],on=['pdb','chain'],how='left')
runs['flag']=''
def setflag(mask,label):
    m=mask&(runs.flag=='')
    runs.loc[m,'flag']=label
amy=set(pd.read_csv(f'{D}/pdb_meta/tables/amyloid_fibril_chains.csv').pdb.str.lower())
setflag(runs.pdb.str.lower().isin(amy),'amyloid_fibril')
art=pd.read_csv(f'{D}/case_files_ABCF/tables/artifact_runs.csv')
k=set(zip(art.pdb,art.chain,art.imgt0))
setflag(pd.Series([ (p,c,i) in k for p,c,i in zip(runs.pdb,runs.chain,runs.imgt0)]),'indel_gap_artifact_confirmed')
extra={ # pdb: (modes, label)
 '8c8t':({'C_FR3'},'indel_gap_artifact_confirmed'),'7m3l':({'C_FR3'},'indel_gap_artifact_confirmed'),
 '7y0o':({'C_FR3'},'indel_gap_artifact_confirmed'),'9yfu':({'C_FR3'},'indel_gap_artifact_probable'),
 '8bw0':({'B_CDR2FR3'},'indel_gap_artifact_confirmed'),'8r4d':({'B_CDR2FR3'},'indel_gap_artifact_confirmed'),
 '8r4b':({'B_CDR2FR3'},'indel_gap_artifact_confirmed'),'9pyh':({'B_CDR2FR3'},'indel_gap_artifact_confirmed'),
 '8snc':({'B_CDR2FR3'},'indel_gap_artifact_probable'),'9om5':({'B_CDR2FR3'},'not_a_register_shift'),
 '7cws':({'A_FR1Nterm'},'indel_gap_artifact_confirmed'),'8fsj':({'A_FR1Nterm'},'indel_gap_artifact_confirmed'),
 '7nk2':({'B_CDR2FR3'},'indel_gap_artifact_confirmed'),
}
for p,(modes,lab) in extra.items():
    setflag((runs.pdb==p)&(runs['mode'].isin(modes)),lab)
de=pd.read_csv(f'{D}/case_files_DE/tables/independent_check_DE_units.csv')
nc=de[de.confirm!='confirmed']
for r in nc.itertuples():
    lab='DE_chain_end_artifact' if (r.last_built_imgt<=120 and r.pdb!='30us') else 'DE_not_confirmed_other'
    setflag((runs.pdb==r.pdb)&(runs.chain==r.chain)&(runs['mode']==r.mode),lab)
cp=pd.read_csv(f'{D}/case_files_ABCF/tables/near_copy_matches.csv')
k=set(zip(cp.pdb,cp.chain,cp.imgt0))
setflag(pd.Series([(p,c,i) in k for p,c,i in zip(runs.pdb,runs.chain,runs.imgt0)]),'near_copy')
setflag((runs.pdb=='8xlk')&(runs['mode']=='C_FR3'),'near_copy')
runs['flag']=runs.flag.replace('', 'none')
runs.to_csv(f'{D}/synthesis/artifact_ledger_runs.csv',index=False)
out=[]
def P(s): print(s); out.append(s)
P('Run-level flags (all methods):'); P(runs.flag.value_counts().to_string())
# unit level: a unit is removed if every one of its first-occurrence runs is flagged
strict={'amyloid_fibril','indel_gap_artifact_confirmed','DE_chain_end_artifact','near_copy','not_a_register_shift'}
loose=strict|{'indel_gap_artifact_probable','DE_not_confirmed_other'}
u=runs.groupby('unit').agg(method=('method','first'),fl=('flag',lambda s:set(s)))
for lab,S in [('strict (confirmed artifacts + copies)',strict),('loose (+probable / unconfirmed)',loose)]:
    u['removed']=u.fl.apply(lambda f: f<=S)
    for m in ['all','EM']:
        uu=u if m=='all' else u[u.method=='EM']
        P(f'{lab} | {m}: units {len(uu)}, removed {int(uu.removed.sum())} ({uu.removed.mean():.1%}), remaining {int((~uu.removed).sum())}')
# breakdown of removed units by primary flag (strict)
u['removed']=u.fl.apply(lambda f: f<=strict)
u['primary']=u.fl.apply(lambda f: sorted(f)[0])
P('Strict removed units by flag (all methods):'); P(u[u.removed].primary.value_counts().to_string())
u['removed_loose']=u.fl.apply(lambda f: f<=loose)
P('Loose removed units by flag (all methods):'); P(u[u.removed_loose].fl.apply(lambda f:'+'.join(sorted(f))).value_counts().to_string())
open(f'{D}/synthesis/artifact_ledger_summary.txt','w').write('\n'.join(out)+'\n')
