"""Collect unique protein sequences from pdb_seqres and prefilter for Ig V-like chains (regex), for ANARCI."""
import gzip, re, json, collections, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
C=f'{SP}/deeper/cache/seq_novelty'
P=[re.compile(r'C.{50,105}C.{1,50}[WF][GA]..'),
   re.compile(r'Y[YFHLICW]C.{2,45}[WFLY]G.G'),
   re.compile(r'W[YFVILHSNC][QRLKH][QKRE].{25,70}[YFH][YFHLICW]C')]
def isv(s): return any(p.search(s) for p in P)
seq2ch=collections.defaultdict(list)
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz','rt') as f:
    hdr=None
    for line in f:
        if line[0]=='>': hdr=line; continue
        if 'mol:protein' not in hdr: continue
        s=line.strip()
        if len(s)<70: continue
        seq2ch[s].append(hdr[1:].split()[0])
print('unique protein seqs >=70', len(seq2ch))
cand={s:v for s,v in seq2ch.items() if isv(s)}
print('candidates', len(cand), 'chains', sum(len(v) for v in cand.values()))
json.dump(cand, open(f'{C}/cand_seqs.json','w'))
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',usecols=['seq','vseq'])
print('sensitivity on study chains', ch.seq.map(isv).mean(), ch.vseq.map(isv).mean(), ch.seq.isin(cand.keys()).mean())
