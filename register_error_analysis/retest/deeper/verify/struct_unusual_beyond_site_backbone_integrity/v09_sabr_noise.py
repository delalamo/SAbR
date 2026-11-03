"""Circularity check: is SAbR (structure-based numbering, ProteinMPNN-encoder embeddings of backbone coordinates)
prone to emit spurious >=3-residue disagreement runs when backbone geometry is non-ideal? If so, chains with poor
geometry would be flagged more often without a real register error, and the geometry association would partly be
built into the error definition.
Test: 40 control chains (clean, SAbR == ANARCI), Gaussian coordinate noise of sigma on every atom, SAbR rerun;
count residues/runs whose SAbR number differs from the unperturbed SAbR number. Non-favoured Ramachandran rate of
the perturbed backbone is estimated with a grid classifier trained on the 150k wwPDB-labelled residues of the
sample (per category: Gly, Pro, pre-Pro, Ile/Val, general)."""
import gzip, shutil, os, sys, json, numpy as np, pandas as pd, warnings
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OC = f'{SP}/deeper/cache/verify_beyond'
TAB = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_beyond_site_backbone_integrity'
SIGMAS = [0.0, 0.15, 0.3, 0.5, 0.8]
REPS = 2


def cat(rn, nxt):
    if rn == 'GLY': return 'gly'
    if rn == 'PRO': return 'pro'
    if nxt == 'PRO': return 'prepro'
    if rn in ('ILE', 'VAL'): return 'ileval'
    return 'gen'


def build_grid():
    R = pd.read_pickle(f'{SP}/deeper/cache/struct_unusual/residues.pkl')
    R = R.sort_values(['pdb', 'chain', 'idx'])
    R['nxt'] = R.groupby(['pdb', 'chain']).rn.shift(-1)
    R = R.dropna(subset=['phi', 'psi', 'rama'])
    R['cat'] = [cat(a, b) for a, b in zip(R.rn, R.nxt)]
    grid = {}
    for c, g in R.groupby('cat'):
        i = ((g.phi.values + 180) // 10).astype(int).clip(0, 35); j = ((g.psi.values + 180) // 10).astype(int).clip(0, 35)
        fav = np.zeros((36, 36)); tot = np.zeros((36, 36))
        np.add.at(fav, (i, j), (g.rama.values == 'Favored')); np.add.at(tot, (i, j), 1)
        grid[c] = ((fav + 0.1) / (tot + 1.0))   # empty cells -> non-favoured
    # accuracy on the training residues
    acc = np.mean([grid[c][min(int((p + 180) // 10), 35), min(int((q + 180) // 10), 35)] > 0.5 for c, p, q in zip(R.cat, R.phi, R.psi)] ==
                  (R.rama.values == 'Favored'))
    return grid, acc


def dih(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n; w = b2 - np.dot(b2, b1n) * b1n
    return np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


def nonfav(chain_res, grid):
    n = k = 0
    for i in range(1, len(chain_res) - 1):
        a, pr, nx = chain_res[i], chain_res[i - 1], chain_res[i + 1]
        if not all(x in a for x in ('N', 'CA', 'C')) or 'C' not in pr or 'N' not in nx:
            continue
        if nx.id[1] - a.id[1] > 1 or a.id[1] - pr.id[1] > 1:
            continue
        phi = dih(pr['C'].coord, a['N'].coord, a['CA'].coord, a['C'].coord)
        psi = dih(a['N'].coord, a['CA'].coord, a['C'].coord, nx['N'].coord)
        g = grid[cat(a.resname, nx.resname)]
        n += 1; k += g[min(int((phi + 180) // 10), 35), min(int((psi + 180) // 10), 35)] <= 0.5
    return 100 * k / max(n, 1)


def runs_ge3(a, b):
    """a,b: lists of numbering keys per residue (None = unnumbered). Count residues differing and runs >=3."""
    diff = [(x is not None and y is not None and x != y) for x, y in zip(a, b)]
    nres = sum(diff); runs = 0; cur = 0
    for dflag in diff + [False]:
        if dflag: cur += 1
        else:
            runs += cur >= 3; cur = 0
    return nres, runs


def job(args):
    p, ch, grid = args
    from Bio.PDB import MMCIFParser
    from sabr import renumber_structure
    fn = f'{SP}/sabr/dl/{p}.cif'
    tmp = None
    if not os.path.exists(fn):
        tmp = f'{OC}/tmp_{p}.cif'
        with gzip.open(f'{SP}/deeper/cache/struct_unusual/cif/{p}.cif.gz', 'rb') as a, open(tmp, 'wb') as b:
            shutil.copyfileobj(a, b)
        fn = tmp
    out = []
    try:
        st = MMCIFParser(QUIET=True).get_structure(p, fn)
        model = st[0]
        for c in [c.id for c in model]:
            if c != ch:
                model.detach_child(c)
        for m in [m.id for m in st][1:]:
            st.detach_child(m)
        orig = {a.get_full_id(): a.coord.copy() for a in st.get_atoms()}
        rng = np.random.default_rng(abs(hash((p, ch))) % 2**32)
        base = None
        for sg in SIGMAS:
            for rep in range(REPS if sg > 0 else 1):
                for a in st.get_atoms():
                    a.coord = orig[a.get_full_id()] + (rng.normal(0, sg, 3) if sg > 0 else 0)
                res = list(st[0][ch])
                nf = nonfav(res, grid)
                try:
                    o = renumber_structure(st, ch)
                    num = [(r.id[1], r.id[2]) for r in o[0][ch]]
                except Exception as e:
                    out.append(dict(pdb=p, chain=ch, sigma=sg, rep=rep, nonfav=nf, err=str(e)[:60])); continue
                if base is None:
                    base = num
                nres, nr = runs_ge3(base, num) if len(num) == len(base) else (np.nan, np.nan)
                out.append(dict(pdb=p, chain=ch, sigma=sg, rep=rep, nonfav=nf, n_diff=nres, runs3=nr, nres=len(num)))
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    return out


if __name__ == '__main__':
    grid, acc = build_grid()
    print('grid classifier accuracy on labelled residues', round(acc, 3))
    s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
    c = s[s.role_cc == 'control'].drop_duplicates('pdb')
    c = c.groupby('cclass', group_keys=False).apply(lambda g: g.sample(min(len(g), 10), random_state=3))
    jobs = [(p, ch, grid) for p, ch in zip(c.pdb, c.chain)]
    rows = []
    with ProcessPoolExecutor(2) as ex:
        for o in ex.map(job, jobs):
            rows.extend(o)
    T = pd.DataFrame(rows); T.to_csv(f'{OUT}/v_sabr_noise.csv', index=False)
    S = T.groupby('sigma').agg(chains=('pdb', 'nunique'), runs=('n_diff', 'size'), nonfav_mean=('nonfav', 'mean'),
                               frac_any_diff=('n_diff', lambda v: (v > 0).mean()), frac_run3=('runs3', lambda v: (v > 0).mean()),
                               mean_diff_res=('n_diff', 'mean'))
    print(S.round(3).to_string())
    S.to_csv(f'{OUT}/v_sabr_noise_summary.csv')
