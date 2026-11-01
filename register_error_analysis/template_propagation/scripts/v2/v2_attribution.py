import json, gzip, pandas as pd, numpy as np
A='/tmp/analysis'; T='/tmp/tmpl'; O='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/tmpl2/out'
t=pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx').drop(columns=['Unnamed: 0'],errors='ignore')
t[['pdb','chain']]=t.PDB_CHAIN_ID.str.split(':',expand=True); t['pdb']=t.pdb.str.lower()
N=len(t); NE=t.pdb.nunique()
ch=pd.read_pickle(f'{A}/chains.pkl')[['pdb','chain','ab']]; t=t.merge(ch,on=['pdb','chain'],how='left')
org={}
for l in open('/tmp/supp/meta2/meta_35.jsonl'):
    e=json.loads(l); p=e['rcsb_id'].lower()
    for pe in e['polymer_entities'] or []:
        tax={o['ncbi_taxonomy_id'] for o in (pe['rcsb_entity_source_organism'] or [])}
        for c in pe['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'] or []: org[(p,c)]=2697049 in tax
t['chain_sars2']=[org.get(k,False) for k in zip(t.pdb,t.chain)]
ent=pd.read_pickle(f'{A}/entries35.pkl')[['pdb','sars2','year','role']]; t=t.merge(ent,on='pdb',how='left')
t['entry_sars2']=t.sars2.fillna(False).astype(bool)
em=pd.read_pickle(f'{T}/errors_seqmapped.pkl')[['PDB_CHAIN_ID','RESIDUE_RANGE','family']]; t=t.merge(em,on=['PDB_CHAIN_ID','RESIDUE_RANGE'],how='left')
Fa=pd.read_pickle(f'{O}/family_summary_arb.pkl').set_index('family'); F=pd.read_pickle(f'{T}/families.pkl').set_index('family')
real_set=set(Fa.index[Fa.arb.isin(['contested: shifted fits worse','uncontested: flagged segment fits poorly (error likely real)'])])
strict_set=set(Fa.index[Fa.arb=='contested: shifted fits worse'])
t['papers_S']=t.family.map(Fa.papers_S).fillna(0); t['labs_S']=t.family.map(Fa.labs_S).fillna(0); t['arb']=t.family.map(Fa.arb)
t['prop_papers']=t.papers_S>=2; t['prop_labs']=t.labs_S>=2; t['likely_real']=t.family.isin(real_set); t['strict_real']=t.family.isin(strict_set)
t['is_ab']=t.ab=='antibody'
def summ(mask,label):
    s=t[mask]; return dict(category=label,rows=len(s),pct_rows=round(100*len(s)/N,2),entries=s.pdb.nunique(),pct_entries=round(100*s.pdb.nunique()/NE,2),families=s.family.nunique())
rows=[summ(t.entry_sars2,'SARS-CoV-2 entries: any flagged chain'),summ(t.entry_sars2&(t.year==2020),'SARS-CoV-2 entries deposited 2020'),
 summ(t.is_ab,'All antibody-chain errors'),summ(t.is_ab&t.prop_papers,'Antibody errors recurring in ≥2 papers'),summ(t.is_ab&t.prop_labs,'Antibody errors recurring in ≥2 labs'),
 summ(t.is_ab&t.prop_papers&t.likely_real,'   ...and map supports real error'),summ(t.entry_sars2|(t.is_ab&t.prop_papers),'UNION SARS-CoV-2 or propagated antibody'),
 summ(t.prop_papers,'ANY protein: recurs in ≥2 papers'),summ(t.prop_labs,'ANY protein: recurs in ≥2 labs'),summ(t.prop_papers&t.likely_real,'ANY protein: ≥2 papers & map supports real'),
 summ(t.prop_labs&t.strict_real,'ANY protein: ≥2 labs & strict')]
R=pd.DataFrame(rows); pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
print(f'== ALL ERRORS: {N} rows, {NE} entries'); print(R.to_string(index=False)); R.to_csv(f'{O}/attribution_all.csv',index=False)
a=t[t.is_ab].copy(); a['seq_papers']=a.family.map(F.papers)
def grp(r):
    if r.sars2 and r.year==2020: return '1 SARS-CoV-2 complex, deposited 2020'
    if r.sars2: return '2 SARS-CoV-2 complex, deposited 2021–22'
    if pd.isna(r.family): return '7 could not be mapped'
    if r.labs_S>=2: return '3 error copied across labs (template)'
    if r.papers_S>=2: return '4 error repeated across papers, same lab'
    if r.seq_papers>=2: return '5 one-off: antibody re-used elsewhere, error not copied'
    return '6 one-off: antibody appears in only one paper'
a['grp']=a.apply(grp,axis=1)
g=a.groupby('grp').agg(chains=('pdb','size'),entries=('pdb','nunique'),families=('family','nunique')); g['%chains']=(100*g.chains/len(a)).round(0); g['%entries']=(100*g.entries/a.pdb.nunique()).round(0)
print(f'\n== ANTIBODY ERRORS: {len(a)} chains, {a.pdb.nunique()} entries'); print(g.to_string()); g.to_csv(f'{O}/attribution_antibody.csv')
names={}
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz','rt') as fh:
    for line in fh:
        if line.startswith('>'):
            h=line[1:].split(None,3); names[h[0]]=h[3].strip() if len(h)>3 else ''
cp=a[a.papers_S>=2]
L=cp.groupby('family').agg(name=('PDB_CHAIN_ID',lambda x: names.get(x.iloc[0].replace(':','_'),'')),flagged_chains=('pdb','size'),labs_with_error=('labs_S','first'),papers_with_error=('papers_S','first'),sars2=('sars2','any'),verdict=('arb','first'),example=('PDB_CHAIN_ID','first'))
print('\nAntibody error families recurring in ≥2 papers:'); print(L.sort_values('labs_with_error',ascending=False).to_string())
s=a[a.sars2==True]; print('\nSARS-CoV-2 antibody errors recurring in ≥2 papers:',int((s.papers_S>=2).sum()),'chains; ≥2 labs:',int((s.labs_S>=2).sum()))
