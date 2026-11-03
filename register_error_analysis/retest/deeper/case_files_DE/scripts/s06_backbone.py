"""Backbone continuity and geometry around the slip: CA(i)-CA(i+1) distances (ANARCI numbering) for 114-128 in error
chains vs clean controls; a distance > 4.2 A is a chain break (two fragments joined only by numbering)."""
import sys
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/scripts')
from common import *
import s02_struct as S2
from Bio.PDB import MMCIFParser
from concurrent.futures import ProcessPoolExecutor
def job(args):
    p, reqs = args
    out = []
    try:
        parser = MMCIFParser(QUIET=True); s = parser.get_structure(p, S2.open_cif(S2.cif_path(p))); m = next(iter(s))
        for c, res in reqs:
            if c not in m: continue
            ch = m[c]; v = [x for x in res if x[4] is not None]
            rm = {(r.id[1], r.id[2].strip()): r for r in ch if r.id[0] == ' '}
            seq = [(x[4], x[5], rm.get((x[0], x[1]))) for x in v]
            row = dict(pdb=p, chain=c)
            dmax, where, breaks = 0, None, []
            for (n0, i0, r0), (n1, i1, r1) in zip(seq, seq[1:]):
                if r0 is None or r1 is None or 'CA' not in r0 or 'CA' not in r1: continue
                if not (100 <= n0 <= 128): continue
                if r1.id[1] - r0.id[1] > 1 and not r1.id[2].strip(): continue  # numbering gap = unbuilt residues
                d = float(np.linalg.norm(r0['CA'].coord - r1['CA'].coord))
                if d > 4.2: breaks.append(f'{n0}{i0}-{n1}{i1}:{d:.1f}')
                if d > dmax: dmax, where = d, f'{n0}{i0}'
            row.update(max_caca_100_128=dmax, max_at=where, breaks=';'.join(breaks), n_breaks=len(breaks))
            out.append(row)
    except Exception as e:
        out.append(dict(pdb=p, chain='*', err=str(e)))
    return out
if __name__ == '__main__':
    S = pd.read_csv(f'{CACHE}/struct_sample.csv')
    rec = sabr_records(set(zip(S.pdb, S.chain)))
    jobs = [(p, [(c, rec[(p, c)]) for c in sorted(set(g.chain)) if (p, c) in rec]) for p, g in S.groupby('pdb')]
    res = []
    with ProcessPoolExecutor(3) as ex:
        for o in ex.map(job, jobs, chunksize=4): res += o
    B = pd.DataFrame(res).merge(S[['pdb', 'chain', 'group']], on=['pdb', 'chain'])
    Rr = runs(); md = Rr[Rr['mode'].isin(MODES)].groupby(['pdb', 'chain'])['mode'].first().to_dict()
    B['mode'] = [md.get(k, '') for k in zip(B.pdb, B.chain)]
    B.to_csv(f'{OUT}/tables/backbone_breaks.csv', index=False)
    B['brk'] = B.n_breaks > 0
    print(B.groupby(['group', 'mode']).agg(n=('pdb', 'size'), with_break=('brk', 'mean'), med_max=('max_caca_100_128', 'median')))
    print(B[(B.group == 'error') & B.brk][['pdb', 'chain', 'mode', 'breaks']].to_string())
