"""Is the deposited sequence itself inconsistent? For every D+/D-/E error unit, find all PDB chains (pdb_seqres) that share
its V region up to the start of CDR3 exactly (IMGT 1-104 prefix, >=80 aa) and compare the CDR3+FR4 segment.
A different CDR3/FR4 length for the same antibody elsewhere suggests a sequence-record discrepancy."""
import sys, gzip, re
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/scripts')
from common import *
C = chains(); R = runs(); E = R[R['mode'].isin(MODES)]
ek = set(zip(E.pdb, E.chain))
L = pd.read_pickle(f'{CACHE}/chains_local.pkl')
X = C.merge(L, on=['pdb', 'chain'])
X['err'] = [k in ek for k in zip(X.pdb, X.chain)]
X['mode'] = X.pdb + X.chain
md = E.groupby(['pdb', 'chain'])['mode'].first().to_dict()
X['mode'] = [md.get(k, '') for k in zip(X.pdb, X.chain)]
# prefix = V sequence up to Cys104 (cdr3 starts right after); use vseq minus cdr3+fr4
def prefix(r):
    v = r.vseq if isinstance(r.vseq, str) else ''
    c3 = r.cdr3_seq if isinstance(r.cdr3_seq, str) else ''
    i = v.find(c3) if c3 else -1
    return v[:i] if i > 60 else None
X['pre'] = X.apply(prefix, axis=1)
X['post'] = [ (s[s.find(p) + len(p): s.find(p) + len(p) + 35] if isinstance(s, str) and isinstance(p, str) and p in s else None) for s, p in zip(X.seq, X.pre)]
units = X[X.err].drop_duplicates('unit')
want = {p[-80:]: p for p in units.pre.dropna() if len(p) >= 80}
print("units with >=80-aa prefix", len(want), "of", len(units))
hits = {}
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz', 'rt') as f:
    hdr = None
    for line in f:
        if line.startswith('>'): hdr = line[1:].split()[0]; continue
        s = line.strip()
        if 'mol:protein' not in '' and len(s) < 90: continue
        for key in want:
            i = s.find(key)
            if i >= 0:
                full = want[key]
                if s.find(full[-min(len(full), 88):]) >= 0:  # whole pre-CDR3 V region (up to 88 aa) identical
                    hits.setdefault(key, []).append((hdr, s[i + len(key): i + len(key) + 35]))
rows = []
for r in units.itertuples():
    if not isinstance(r.pre, str) or len(r.pre) < 80: continue
    key = r.pre[-80:]
    own = r.post or ''
    others = hits.get(key, [])
    # CDR3+FR4 segment up to 'WG.G' or 'FG.G' motif
    def seg(t):
        m = re.search(r'[WF]G.G', t or '')
        return t[:m.start()] if m else None
    o = seg(own)
    variants = {}
    for h, t in others:
        sg = seg(t)
        if sg is None: continue
        variants.setdefault(sg, []).append(h)
    rows.append(dict(pdb=r.pdb, chain=r.chain, mode=r.mode, own_cdr3=o, n_chains_same_prefix=len(others),
                     n_variants=len(variants), variants='; '.join(f'{k}({len(v)}:{",".join(sorted(set(x.split("_")[0] for x in v))[:6])})' for k, v in variants.items())))
Q = pd.DataFrame(rows)
Q['other_len'] = [any(len(k.split('(')[0]) != len(o or '') for k in v.split('; ') if k) for o, v in zip(Q.own_cdr3, Q.variants)]
Q.to_csv(f'{OUT}/tables/seq_consistency_DE.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 140)
print(Q[Q.n_variants > 1].to_string())
print('units', len(Q), 'with >1 variant', (Q.n_variants > 1).sum(), 'with a different-length variant', Q.other_len.sum())
