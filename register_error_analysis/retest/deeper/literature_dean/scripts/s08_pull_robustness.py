"""Robustness of the canonical-pull test to the choice of reference distribution, plus a per-transition table mapped to
North/PyIgClassify CDR-length nomenclature (L1: North = IMGT+5; H1: IMGT+5; H2: IMGT+2; L3/H3 kept as IMGT)."""
import pickle, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/literature_dean/tables'
A=pd.read_csv(f'{OUT}/apparent_length_error_units.csv')
R=pickle.load(open(f'{SP}/deeper/cache/literature_dean/ref.pkl','rb'))
L=pd.read_csv(f'{OUT}/apparent_cdr_lengths_all_chains.csv.gz')
refs={'X-ray<=3.0 A unique V, <=2019 (main)':R[(R.hq30)&(R.first_year<=2019)].rename(columns={'len_CDR1':'l1','len_CDR2':'l2','len_CDR3':'l3'}),
      'all PDB unique V domains, all years':R.rename(columns={'len_CDR1':'l1','len_CDR2':'l2','len_CDR3':'l3'}),
      'study set clean chains (3-5 A, no run)':L[~L.err_any].drop_duplicates('unit').rename(columns={'rL1':'l1','rL2':'l2','rL3':'l3'})}
rng=np.random.default_rng(7); rows=[]
for name,df in refs.items():
    F={(ct,k):(df[df.ctype==ct]['l'+k].value_counts(normalize=True)).to_dict() for ct in 'HKL' for k in '123'}
    N={(ct,k):(df.ctype==ct).sum() for ct in 'HKL' for k in '123'}
    fr=lambda ct,k,l: F[(ct,str(k))].get(l,0.5/N[(ct,str(k))])
    ft=np.array([fr(r.ctype,r.cdr,r.true_len) for r in A.itertuples()]); fa=np.array([fr(r.ctype,r.cdr,r.app_len) for r in A.itertuples()])
    fo=np.array([fr(r.ctype,r.cdr,r.true_len-r.diff) for r in A.itertuples()])
    o=int((fa>ft).sum()); pe=0.5*((fa>ft).astype(int)+(fo>ft).astype(int)); e=pe.sum()
    sims=rng.binomial(1,np.tile(pe,(50000,1))).sum(1); p=max(2*min((sims>=o).mean(),(sims<=o).mean()),2/50000)
    rows.append(dict(reference=name,n=len(A),observed_more_common=o,expected=round(e,1),p=p))
T=pd.DataFrame(rows); T.to_csv(f'{OUT}/canonical_pull_robustness.csv',index=False); print(T.to_string())
off={('H','1'):5,('K','1'):5,('L','1'):5,('H','2'):2}
A['north_true']=[f"{'H' if r.ctype=='H' else 'L'}{r.cdr}-{r.true_len+off.get((r.ctype,str(r.cdr)),0)}" for r in A.itertuples()]
A['north_app']=[f"{'H' if r.ctype=='H' else 'L'}{r.cdr}-{r.app_len+off.get((r.ctype,str(r.cdr)),0)}" for r in A.itertuples()]
t=A.groupby(['ctype','cdr','north_true','north_app']).agg(n=('unit','size'),f_true=('f_true','first'),f_app=('f_app','first'),
     em=('method',lambda s:(s=='EM').sum()),modes=('modes',lambda s:';'.join(sorted(set(';'.join(s).split(';')))))).reset_index().sort_values('n',ascending=False)
t.to_csv(f'{OUT}/canonical_pull_transitions_north.csv',index=False); print(t[t.n>=2].round(3).to_string())
