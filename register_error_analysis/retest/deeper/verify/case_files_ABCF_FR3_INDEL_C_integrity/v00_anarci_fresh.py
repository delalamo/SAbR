"""Independent ANARCI IMGT numbering of every unique V-region sequence (chains.csv vseq), human+mouse HMMs
(the scan/detector default) - used to recompute FR3 length and gap positions without the seq_novelty cache.
Output: cache/verify_fr3_integrity/anarci_vseq_hm.pkl  {vseq: (chain_type, [(pos, ins, aa), ...]) or None}"""
import pickle, sys, os, pandas as pd
from anarci import run_anarci
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUTF = f'{SP}/deeper/cache/verify_fr3_integrity/anarci_vseq_hm.pkl'
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False, usecols=['vseq', 'method'])
seqs = sorted(set(C.vseq.dropna()))
res = pickle.load(open(OUTF, 'rb')) if os.path.exists(OUTF) else {}
todo = [s for s in seqs if s not in res]
print(len(seqs), 'todo', len(todo), flush=True)
for k in range(0, len(todo), 400):
    ch = todo[k:k + 400]
    _, num, ali, _ = run_anarci([(str(i), s) for i, s in enumerate(ch)], scheme='imgt', ncpu=2, allowed_species=['human', 'mouse'])
    for s, n, a in zip(ch, num, ali):
        if not n: res[s] = None; continue
        res[s] = (a[0]['chain_type'], [(p[0], p[1].strip(), aa) for p, aa in n[0][0] if aa != '-'])
    pickle.dump(res, open(OUTF, 'wb'))
    print(k + len(ch), flush=True)
