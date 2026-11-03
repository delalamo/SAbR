# Identify every PDB entry containing an antibody V domain (ANARCI H/K/L on SEQRES) and write the list
import json, pickle, collections, re
C='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/antigen_competition'
d=pickle.load(open(f'{C}/seqres.pkl','rb')); chains=d['chains']
an=json.load(open(f'{C}/anarci_pdbwide.json'))
def ctype(s):
    r=an.get(s)
    if not r: return None
    t={x['chain_type'] for x in r}
    if t & {'H','K','L'}: return 'Ab'
    return 'TCR'
lab={}
for k,s in chains.items():
    t=ctype(s)
    if t: lab[k]=t
ent=collections.defaultdict(dict)
for k,t in lab.items():
    p,c=k.split('_',1); ent[p][c]=t
abent=sorted(p for p,v in ent.items() if 'Ab' in v.values())
print('chains Ab',sum(v=='Ab' for v in lab.values()),'TCR',sum(v=='TCR' for v in lab.values()),'entries with Ab',len(abent))
json.dump(abent,open(f'{C}/ab_entries_pdbwide.json','w'))
pickle.dump(lab,open(f'{C}/chain_iglabel.pkl','wb'))
