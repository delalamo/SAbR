"""For each FR3-atypical mode-C error unit (EM), find other PDB chains with the IDENTICAL atypical FR3 segment
(IMGT 66-104 from SEQRES; i.e. the same indel in the same sequence context), from other entries, and ask whether SAbR
and ANARCI disagree there too. Chains already renumbered by SAbR (3-5 A scan / validation outputs) are read from
tables/fr3_pattern_chains.csv.gz; for up to 3 extra higher-resolution (<3.0 A) chains per unit SAbR is run here.
Same disagreement in independent (especially high-resolution) structures -> numbering convention, not a model error.
Output: tables/same_fr3_check.csv
"""
import gzip, json, os, re, urllib.request, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CA = f'{SP}/deeper/cache/verify_fr3_indel_confounding'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3_INDEL_C_confounding'
os.makedirs(f'{CA}/cif', exist_ok=True)
U = pd.read_csv(f'{OUT}/tables/units_em_flags.csv.gz')
CF = pd.read_csv(f'{OUT}/tables/chain_features.csv.gz', low_memory=False)
DT = pd.read_csv(f'{OUT}/tables/c_units_detail.csv')
P = pd.read_csv(f'{OUT}/tables/fr3_pattern_chains.csv.gz')
err = U[(U.first_modes.fillna('').str.contains('C_FR3')) & (U.fr3_atyp == 1)]
targets = {}
for r in err.itertuples():
    targets[r.unit] = dict(fr3=r.fr3_seq, pdb=r.pdb, paper=r.paper, pattern=r.fr3_pattern)
# resolution / method index
RES = {}
with open('/tmp/supp/pdbwide/entries.idx', errors='replace') as f:
    for line in f:
        t = line.rstrip('\n').split('\t')
        if len(t) >= 8 and len(t[0]) == 4:
            try: RES[t[0].lower()] = (float(t[6]), t[7])
            except ValueError: RES[t[0].lower()] = (np.nan, t[7])
hits = {u: [] for u in targets}
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz', 'rt') as f:
    head = None
    for line in f:
        if line.startswith('>'):
            head = line[1:].split()[0]; continue
        if 'mol:protein' not in '' and head:
            s = line.strip()
            for u, t in targets.items():
                if t['fr3'] and t['fr3'] in s:
                    p, c = head.split('_', 1)
                    hits[u].append((p, c))
A3 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G', HIS='H', ILE='I', LEU='L', LYS='K', MET='M', PHE='F',
          PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V')


def run_sabr(p, c):
    from Bio.PDB import MMCIFParser
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from anarci import run_anarci
    from sabr import renumber_structure
    fn = f'{CA}/cif/{p}.cif'
    if not os.path.exists(fn):
        data = urllib.request.urlopen(f'https://files.rcsb.org/download/{p.upper()}.cif', timeout=120).read()
        open(fn, 'wb').write(data)
    st = MMCIFParser(QUIET=True).get_structure(p, fn)
    model = next(iter(st)); ch = model[c]
    res = [x for x in ch if x.id[0] == ' ' and x.get_resname() in A3][:130]
    tst = Structure(p); tst.header = dict(getattr(st, "header", {}) or {}); tm = Model(0); tst.add(tm); tc = ch.copy()
    keep = {x.id for x in res}
    for x in list(tc):
        if x.id not in keep: tc.detach_child(x.id)
    tm.add(tc)
    new = renumber_structure(tst, chain=c, chain_type='H')
    sab = [(x.id[1], x.id[2].strip()) for x in next(iter(new))[c]]
    seq = ''.join(A3[x.get_resname()] for x in res)
    an = run_anarci([('q', seq)], scheme='imgt')
    num = [(pp, ii.strip()) for (pp, ii), aa in an[1][0][0][0] if aa != '-']
    st0 = an[1][0][0][1]
    pairs = list(zip(num, sab[st0:st0 + len(num)]))
    dis = [(a, b) for a, b in pairs if 66 <= a[0] <= 104 and a != b]
    return len(dis), ','.join(f'{a[0]}{a[1]}>{b[0]}{b[1]}' for a, b in dis)[:200]


rows = []
Pk = P.set_index(['pdb', 'chain'])
for u, t in targets.items():
    hl = sorted(set(hits[u]), key=lambda pc: (RES.get(pc[0], (99,))[0] if RES.get(pc[0], (99,))[0] == RES.get(pc[0], (99,))[0] else 99))
    own_pdbs = set(CF[CF.unit == u].pdb)
    own_paper = t['paper']
    ran = 0
    for p, c in hl:
        res, meth = RES.get(p, (np.nan, ''))
        paper = CF[(CF.pdb == p)].paper.iloc[0] if (CF.pdb == p).any() else p
        rel = 'same_unit' if p in own_pdbs else ('same_paper' if paper == own_paper else 'other')
        src = None; ndis = np.nan; dpos = ''
        if (p, c) in Pk.index:
            x = Pk.loc[(p, c)]; x = x.iloc[0] if isinstance(x, pd.DataFrame) else x
            src = 'scan_output'; ndis = x.n_dis; dpos = x.dis_pos
        elif rel == 'other' and res == res and res < 3.0 and ran < 3:
            try:
                ndis, dpos = run_sabr(p, c); src = 'sabr_run_here'; ran += 1
            except Exception as ex:
                src = f'fail {type(ex).__name__}: {str(ex)[:60]}'
        rows.append(dict(err_unit_pdb=t['pdb'], pattern=t['pattern'], pdb=p, chain=c, res=res, method=meth, relation=rel, source=src,
                         n_dis_FR3=ndis, dis_pos=dpos))
        if src: print(rows[-1], flush=True)
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/tables/same_fr3_check.csv', index=False)
s = R[R.source.notna() & ~R.source.astype(str).str.startswith('fail')]
s = s.assign(dis3=s.n_dis_FR3 >= 3)
print(s.groupby(['err_unit_pdb', 'pattern', 'relation']).agg(n=('dis3', 'size'), dis=('dis3', 'sum'), min_res=('res', 'min')).to_string())
