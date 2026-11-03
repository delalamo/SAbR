"""Find PDB chains (any resolution/method) carrying the identical V region of each EM mode-C FR3-indel error chain
(and of artifact reference cases), from pdb_seqres. Output tables/v05a_same_v.csv"""
import gzip, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3_INDEL_C_integrity'
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
Q = {('7te9', 'H'), ('8xlk', 'K'), ('7y0o', 'H'), ('9dm7', 'H'), ('9ml5', 'H'), ('9yfu', 'N'), ('7m3l', 'H2'), ('8c8t', 'R'),
     ('9bop', 'M'), ('9e89', 'H'), ('2x7l', 'A'), ('30ad', 'H')}
q = {(p, c): v for p, c, v in zip(C.pdb, C.chain, C.vseq) if (p, c) in Q}
core = {k: v[5:95] for k, v in q.items()}  # FR1(6)..FR3 core, robust to tags / truncation
idx = {}
for line in open('/dev/null'): pass
hits = []
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz', 'rt') as f:
    hdr = None
    for line in f:
        if line.startswith('>'): hdr = line[1:].split()[0]; continue
        s = line.strip()
        for k, cs in core.items():
            if cs in s:
                p, c = hdr.split('_', 1)
                hits.append(dict(query=f'{k[0]}_{k[1]}', pdb=p, chain=c, full_v_identical=q[k] in s))
H = pd.DataFrame(hits)
ent = pd.read_csv('/tmp/supp/pdbwide/entries.idx', sep='\t', header=None, skiprows=2, dtype=str, on_bad_lines='skip')
print(ent.head(2).T)
ent = ent.rename(columns={0: 'pdb', 2: 'dep', 6: 'res', 7: 'method'})
ent['pdb'] = ent.pdb.str.lower()
H = H.merge(ent[['pdb', 'dep', 'res', 'method']], on='pdb', how='left')
H = H.drop_duplicates(['query', 'pdb'])
H.to_csv(f'{OUT}/tables/v05a_same_v.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
for qq, x in H.groupby("query"): print(qq, " ".join(f"{a}_{b}({r},{str(m)[:5]})" for a, b, r, m in zip(x.pdb, x.chain, x.res, x.method)))
