"""Within-paper sequence inconsistency: the same antibody (identical V region up to the end of FR3, >=80 aa) deposited in
the same paper with CDR3+FR4 segments of different length. Does it mark D/E errors?"""
import sys
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/scripts')
from common import *
from scipy.stats import fisher_exact
CH = chains(); R = runs(); L = pd.read_pickle(f'{CACHE}/chains_local.pkl')
X = CH.merge(L, on=['pdb', 'chain'])
def pre(r):
    v = r.vseq if isinstance(r.vseq, str) else ''; c3 = r.cdr3_seq if isinstance(r.cdr3_seq, str) else ''
    i = v.find(c3) if c3 else -1
    return v[:i] if i >= 80 else None
X['pre'] = X.apply(pre, axis=1)
X['post_len'] = X.cdr3_n.fillna(0) + X.fr4_seq.fillna('').str.len()
X['c3f4'] = X.cdr3_seq.fillna('') + '|' + X.fr4_seq.fillna('')
ER = pd.read_pickle(f'{CACHE}/err_struct.pkl'); bad = set(zip(ER[ER.confirm == 'not confirmed'].pdb, ER[ER.confirm == 'not confirmed'].chain))
md = {}
for r in R.itertuples():
    if r.mode in MODES and (r.pdb, r.chain) not in bad: md.setdefault((r.pdb, r.chain), set()).add(r.mode)
X['DE'] = [bool(md.get(k, set()) & set(MODES)) for k in zip(X.pdb, X.chain)]
g = X.dropna(subset=['pre']).groupby(['paper', 'pre'])
rows = []
for (p, pr), x in g:
    lens = x.cdr3_n.nunique()
    rows.append(dict(paper=p, n_chains=len(x), n_entries=x.pdb.nunique(), cdr3_len_variants=lens, cdr3_seq_variants=x.cdr3_seq.nunique(),
                     fr4_variants=x.fr4_seq.str[:7].nunique(), any_DE=x.DE.any(), method=x.method.iloc[0],
                     variants='; '.join(sorted(set(x.c3f4)))[:200], entries=','.join(sorted(x.pdb.unique()))[:120]))
G = pd.DataFrame(rows)
G = G[G.n_entries > 1]
G['inconsistent_cdr3'] = G.cdr3_len_variants > 1
tab = pd.crosstab(G.inconsistent_cdr3, G.any_DE)
print(tab); print(fisher_exact(tab.values) if tab.shape == (2, 2) else '')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
print(G[G.inconsistent_cdr3][['paper', 'n_entries', 'any_DE', 'variants', 'entries']].to_string())
G.to_csv(f'{OUT}/tables/within_paper_sequence_variants.csv', index=False)
