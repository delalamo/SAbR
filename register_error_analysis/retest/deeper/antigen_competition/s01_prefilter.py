# Parse PDB-wide SEQRES, prefilter Ig-V-like protein chains, write unique candidate sequences
import gzip, re, json, collections, pickle
C='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/antigen_competition'
# conserved V-domain features: Cys23 ... Trp41 (C x{5,20} W [VIFYLHW]? [RQKH]) and FR4 [WF]G.G after it
pat=re.compile(r'C.{6,22}W[VIFYLHWAS][RQKHEL][QKRHEL].{20,110}[WF][GA].G')
chains={}; desc={}
name=None
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz','rt') as f:
    for line in f:
        if line[0]=='>':
            h=line[1:].split()
            name=h[0] if 'mol:protein' in line else None
            if name: desc[name]=line.split('  ',1)[1].strip() if '  ' in line else ''
        elif name:
            chains[name]=line.strip()
seqs=collections.defaultdict(list)
for k,s in chains.items(): seqs[s].append(k)
cand=[s for s in seqs if 60<=len(s)<=1500 and pat.search(s)]
print('protein chains',len(chains),'unique',len(seqs),'candidates',len(cand), 'cand chains',sum(len(seqs[s]) for s in cand))
pickle.dump(dict(chains=chains,desc=desc),open(f'{C}/seqres.pkl','wb'))
json.dump(cand,open(f'{C}/cand_seqs.json','w'))
