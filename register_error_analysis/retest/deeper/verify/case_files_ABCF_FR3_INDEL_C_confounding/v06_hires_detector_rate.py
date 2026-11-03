"""Detector false-positive rate at FR3 indels in genuinely high-resolution structures.

NB: the investigator's s08 labelled sabr/out/validation.jsonl as a "<3 A high-resolution set"; it is not (all entries
are 3.0-5.0 A by the PDB index), so the 'n_hires'/'hires_dis' columns of their gap_patterns.csv are 3-5 A data.
Here: PDB-wide SEQRES chains with a cached ANARCI numbering, X-ray at <= 2.5 A. For each FR3-atypical pattern (SEQRES
numbering, chain-type norm gaps removed) up to 4 distinct V sequences, plus a random sample of FR3-typical V sequences,
are renumbered with SAbR; >=3 FR3 (66-104) residues where SAbR != ANARCI on the modelled residues = "would be flagged".
Only chains whose modelled FR3 carries the same ANARCI pattern as SEQRES (fully built FR3) are scored.
Output: tables/hires_detector_rate.csv
"""
import gzip, os, pickle, random, sys, urllib.request, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CA = f'{SP}/deeper/cache/verify_fr3_indel_confounding'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3_INDEL_C_confounding'
os.makedirs(f'{CA}/cif', exist_ok=True)
AN = pickle.load(open(f'{SP}/deeper/cache/seq_novelty/anarci_num.pkl', 'rb'))
RES = {}
with open('/tmp/supp/pdbwide/entries.idx', errors='replace') as f:
    for line in f:
        t = line.rstrip('\n').split('\t')
        if len(t) >= 8 and len(t[0]) == 4:
            try: RES[t[0].lower()] = (float(t[6]), t[7])
            except ValueError: pass
NORMGAP = {'H': {73}, 'K': {73, 81, 82}, 'L': {73}}
NORMLEN = {'H': {38}, 'K': {36}, 'L': {36, 38}}


def fr3_pattern(nm, ct):
    pos = {p for p, i, a in nm if i == ''}
    ins = sorted({f'{p}{i}' for p, i, a in nm if i != '' and 66 <= p <= 104})
    gaps = tuple(k for k in range(66, 105) if k not in pos and k not in NORMGAP[ct])
    return str(gaps) + ("+" + ",".join(ins) if ins else ""), sum(66 <= p <= 104 for p, i, a in nm), (min(pos) if pos else 999), (max(pos) if pos else -1)


cand = []
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz', 'rt') as f:
    h = None
    for line in f:
        if line[0] == '>': h = line[1:].split()[0]; continue
        s = line.strip()
        d = AN.get(s)
        if not d: continue
        p, c = h.split('_', 1)
        r, m = RES.get(p, (np.nan, ''))
        if not (r <= 2.5 and 'X-RAY' in m): continue
        for dom in d[:1]:
            ct = dom[0] if dom[0] in 'HKL' else None
            if ct is None: continue
            nm = [(int(pp), str(ii).strip(), a) for pp, ii, a in dom[4]]
            pat, L, lo, hi = fr3_pattern(nm, ct)
            if lo > 60 or hi < 110: continue
            vkey = ''.join(a for pp, ii, a in nm)
            cand.append(dict(pdb=p, chain=c, res=r, ct=ct, pattern=pat, fr3len=L, atyp=L not in NORMLEN[ct], vkey=vkey))
Cd = pd.DataFrame(cand).sort_values('res').drop_duplicates('vkey')
print('high-res X-ray V chains (distinct V):', len(Cd), 'atypical FR3:', Cd.atyp.sum(), flush=True)
random.seed(1)
pick = []
for (ct, pat), g in Cd[Cd.atyp].groupby(['ct', 'pattern']):
    pick += g.head(4).to_dict('records')
typ = Cd[~Cd.atyp]
for ct, n in [('H', 30), ('K', 10), ('L', 10)]:
    g = typ[typ.ct == ct]
    pick += g.sample(min(n, len(g)), random_state=1).to_dict('records')
print('to renumber:', len(pick), flush=True)
A3 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G', HIS='H', ILE='I', LEU='L', LYS='K', MET='M', PHE='F',
          PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V')
from Bio.PDB import MMCIFParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from anarci import run_anarci
from sabr import renumber_structure
rows = []
done = set()
if os.path.exists(f'{OUT}/tables/hires_detector_rate.csv'):
    old = pd.read_csv(f'{OUT}/tables/hires_detector_rate.csv'); rows = old.to_dict('records'); done = set(zip(old.pdb, old.chain))
for k, x in enumerate(pick):
    p, c, ct = x['pdb'], x['chain'], x['ct']
    if (p, c) in done: continue
    rec = dict(x); rec.pop('vkey')
    try:
        fn = f'{CA}/cif/{p}.cif'
        if not os.path.exists(fn):
            open(fn, 'wb').write(urllib.request.urlopen(f'https://files.rcsb.org/download/{p.upper()}.cif', timeout=120).read())
        st = MMCIFParser(QUIET=True).get_structure(p, fn)
        model = next(iter(st)); ch = model[c]
        res = [r for r in ch if r.id[0] == ' ' and r.get_resname() in A3][:135]
        tst = Structure(p); tst.header = dict(getattr(st, 'header', {}) or {}); tm = Model(0); tst.add(tm); tc = ch.copy()
        keep = {r.id for r in res}
        for r in list(tc):
            if r.id not in keep: tc.detach_child(r.id)
        tm.add(tc)
        new = renumber_structure(tst, chain=c, chain_type='H' if ct == 'H' else 'K,L')
        sab = [(r.id[1], r.id[2].strip()) for r in next(iter(new))[c]]
        seq = ''.join(A3[r.get_resname()] for r in res)
        an = run_anarci([('q', seq)], scheme='imgt')
        num = [(pp, ii.strip()) for (pp, ii), aa in an[1][0][0][0] if aa != '-']
        st0 = an[1][0][0][1]
        nm_mod = [(pp, ii, '') for (pp, ii) in num]
        pat_mod = fr3_pattern(nm_mod, ct)[0]
        pairs = list(zip(num, sab[st0:st0 + len(num)]))
        dis = [(a, b) for a, b in pairs if 66 <= a[0] <= 104 and a != b]
        rec.update(pattern_model=pat_mod, built_same=pat_mod == x['pattern'], n_dis=len(dis),
                   dis_pos=','.join(f'{a[0]}{a[1]}>{b[0]}{b[1]}' for a, b in dis)[:200])
    except Exception as ex:
        rec.update(error=f'{type(ex).__name__}: {str(ex)[:100]}')
    rows.append(rec)
    if k % 10 == 0:
        pd.DataFrame(rows).to_csv(f'{OUT}/tables/hires_detector_rate.csv', index=False); print(k, rec, flush=True)
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/tables/hires_detector_rate.csv', index=False)
ok = R[R.built_same == True]
ok = ok.assign(dis3=ok.n_dis >= 3)
print(ok.groupby('atyp').agg(n=('dis3', 'size'), flagged=('dis3', 'sum'), rate=('dis3', 'mean')))
print(ok[ok.atyp].groupby(['ct', 'pattern']).agg(n=('dis3', 'size'), flagged=('dis3', 'sum')).to_string())
