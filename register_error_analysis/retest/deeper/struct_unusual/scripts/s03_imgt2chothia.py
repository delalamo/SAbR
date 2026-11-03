"""Derive the IMGT -> Chothia position map for the ABangle core-set positions by numbering a random sample of
V sequences with ANARCI in both schemes. Core-set positions avoid insertion sites, so the map should be unique."""
import json, collections, numpy as np, pandas as pd
from anarci import anarci
from common import SP, C, TAB

d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['ctype', 'vseq', 'vhh'])
core = json.load(open(f'{C}/abangle_src/ABangle/data/coresets.json'))
rng = np.random.default_rng(1)
res = {}
for ct, chain in [('H', 'H'), ('K', 'L'), ('L', 'L')]:
    seqs = d[d.ctype == ct].vseq.dropna().drop_duplicates().sample(150, random_state=1).tolist()
    cnt = collections.defaultdict(collections.Counter)
    for scheme_pair in [None]:
        num_i = anarci([(str(i), s) for i, s in enumerate(seqs)], scheme='imgt', output=False)[0]
        num_c = anarci([(str(i), s) for i, s in enumerate(seqs)], scheme='chothia', output=False)[0]
        for a, b in zip(num_i, num_c):
            if a is None or b is None:
                continue
            ai = {}  # seq index -> imgt
            k = a[0][1]
            for (pos, ins), aa in a[0][0]:
                if aa != '-':
                    ai[k] = (pos, ins.strip()); k += 1
            k = b[0][1]
            for (pos, ins), aa in b[0][0]:
                if aa != '-':
                    if k in ai and ins.strip() == '' and pos in core[chain]:
                        cnt[pos][ai[k]] += 1
                    k += 1
    m = {}
    for pos in core[chain]:
        (best, n), tot = cnt[pos].most_common(1)[0], sum(cnt[pos].values())
        m[pos] = best[0]
        if n / tot < 0.98 or best[1] != '':
            print('ambiguous', ct, pos, cnt[pos].most_common(3))
    res[ct] = {str(k): v for k, v in m.items()}
    print(ct, res[ct])
json.dump(res, open(f'{TAB}/chothia_core_to_imgt.json', 'w'), indent=1)
