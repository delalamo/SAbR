"""V02: independent SAbR-vs-ANARCI rerun on (a) X-ray structures carrying the claimed gap patterns, extended beyond the
investigator's picks (PDB-wide search in v01), and (b) ImmuneBuilder / IgFold predicted models of the SAME antibodies
as the flagged error chains (a model that cannot contain a deposition register error), plus genuine-error controls.

Pipeline mirrors the scan (sabr_scan_light.py, fast mode): chain_type fixed to H, chain trimmed to the ANARCI heavy
V domain +-3 residues, unobserved-residue metadata via MMCIF2Dict; ANARCI is run on the SEQRES entity sequence and
mapped to modelled residues through label_seq_id (structures) or directly (predicted models).
Output: tables/rerun_residues.csv (per chain: disagreement list and >=3-residue runs)."""
import os, sys, json, gzip, urllib.request, warnings, logging, pandas as pd
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
from multiprocessing import Pool
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CA = f'{SP}/deeper/cache/verify_artifact_indel'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_ARTIFACT_INDEL_integrity'
A3 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G', HIS='H', ILE='I', LEU='L', LYS='K', MET='M',
          PHE='F', PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V', MSE='M')

# (group, pattern, kind, id, chain)
JOBS = [
    # FR3 gap 84 (investigator: 9f1i, 6dbd, 4o4y, 4ma3) + extensions 7kbk (VHH 2.09), 9f18, 4o51, 4rzc (2.72)
    ('xray', 'FR3:84', 'pdb', '9f1i', 'H'), ('xray', 'FR3:84', 'pdb', '6dbd', 'A'), ('xray', 'FR3:84', 'pdb', '4o4y', 'H'),
    ('xray', 'FR3:84', 'pdb', '4ma3', 'B'), ('xray', 'FR3:84', 'pdb', '7kbk', 'C'), ('xray', 'FR3:84', 'pdb', '9f18', 'H'),
    ('xray', 'FR3:84', 'pdb', '4o51', 'B'), ('xray', 'FR3:84', 'pdb', '4rzc', 'H'),
    ('xray', 'FR3:80+84', 'pdb', '5dhv', 'A'), ('xray', 'FR3:80+84', 'pdb', '5dhv', 'H'),
    ('xray', 'FR3:84+85', 'pdb', '4z9k', 'B'), ('xray', 'FR3:84+85', 'pdb', '7mlh', 'C'), ('xray', 'FR3:84+85', 'pdb', '8zca', 'C'),
    ('xray', 'FR3:84+85', 'pdb', '30bh', 'H'), ('xray', 'FR3:84+85', 'pdb', '5trp', 'H'),
    ('xray', 'FR1:26', 'pdb', '5aam', 'A'), ('xray', 'FR1:26', 'pdb', '1ghf', 'H'),
    ('xray', 'FR3:96', 'pdb', '7te4', 'H'), ('xray', 'FR3:96', 'pdb', '7te4', 'B'),
    ('xray', 'FR3:81', 'pdb', '9p4c', 'H'), ('xray', 'FR3:81', 'pdb', '4ztp', 'H'), ('xray', 'FR3:81', 'pdb', '6cjk', 'B'),
    ('xray', 'FR3:83', 'pdb', '5v6m', 'H'), ('xray', 'FR3:83', 'pdb', '9dmf', 'H'),
]
# predicted models of the flagged antibodies (fv ids from pred/full/units.json) and of genuine-error controls
PRED = {'fv0056': ('2x7l', 'FR3:80+84'), 'fv0158': ('9bop', 'FR3:84'), 'fv0164': ('9e89', 'FR3:84+85'), 'fv0189': ('30bg', 'FR3:84+85'),
        'fv0190': ('30ad', 'FR3:84'), 'fv0204': ('9pf9', 'FR3:84'), 'fv0125': ('8gs*', 'FR1:26'), 'fv0025': ('7eaj/7eak', 'FR1:26'),
        'fv0174': ('9ml5', 'FR3:84 (genuine 85-95 run?)'), 'fv0185': ('9yfu', 'FR3:85')}
for fv, (lab, pat) in PRED.items():
    for tool in ['immunebuilder', 'igfold']:
        JOBS.append((f'pred_{tool}', pat, 'pred', f'{SP}/pred/full/{tool}/{fv}.pdb|{lab}', 'H'))


def fetch_cif(p):
    f = f'{CA}/cif/{p}.cif'
    if not os.path.exists(f):
        data = urllib.request.urlopen(f'https://files.rcsb.org/download/{p.upper()}.cif', timeout=120).read()
        open(f, 'wb').write(data)
    return f


def anarci_heavy(seq, species=('human', 'mouse')):
    # scan setting = ANARCI default allowed_species ['human','mouse']; species=None -> all HMMs (rabbit, alpaca, ...)
    from anarci import run_anarci
    an = run_anarci([('q', seq)], scheme='imgt', allowed_species=list(species) if species else None)
    for dom, det in zip(an[1][0] or [], an[2][0] or []):
        if det['chain_type'] == 'H':
            num = [((p, i.strip()), a) for (p, i), a in dom[0] if a != '-']
            return dom[1], dom[2], [n for n, a in num]
    return None


def runs_of(flags):
    out, cur = [], []
    for k, f in enumerate(flags):
        if f: cur.append(k)
        else:
            if len(cur) >= 3: out.append((cur[0], cur[-1]))
            cur = []
    if len(cur) >= 3: out.append((cur[0], cur[-1]))
    return out


def process(job):
    group, pat, kind, ident, ch = job
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from sabr import renumber_structure
    rec = dict(group=group, pattern=pat, id=ident.split('|')[-1] if kind == 'pred' else ident, chain=ch)
    try:
        if kind == 'pdb':
            f = fetch_cif(ident)
            md = MMCIF2Dict(f)
            st = MMCIFParser(QUIET=True).get_structure(ident, f)
            # entity sequence and auth->label_seq mapping for this chain
            asym = md['_atom_site.auth_asym_id']; lsi = md['_atom_site.label_seq_id']; asi = md['_atom_site.auth_seq_id']
            ins = md['_atom_site.pdbx_PDB_ins_code']; ent = md['_atom_site.label_entity_id']; mdl = md['_atom_site.pdbx_PDB_model_num']
            m0 = mdl[0]
            amap, eid = {}, None
            for a, l, s_, i_, e, mm in zip(asym, lsi, asi, ins, ent, mdl):
                if a == ch and mm == m0 and l not in ('.', '?'):
                    amap[(int(s_), '' if i_ in ('?', '.') else i_)] = int(l) - 1; eid = e
            eids = md['_entity_poly.entity_id']; seqs = md['_entity_poly.pdbx_seq_one_letter_code_can']
            eids = [eids] if isinstance(eids, str) else eids; seqs = [seqs] if isinstance(seqs, str) else seqs
            seq = dict(zip(eids, seqs))[eid].replace('\n', '')
        else:
            path = ident.split('|')[0]
            st = PDBParser(QUIET=True).get_structure('pred', path); md = None
            res0 = [r for r in next(iter(st))[ch] if r.id[0] == ' ']
            seq = ''.join(A3.get(r.get_resname(), 'X') for r in res0)
            amap = {(r.id[1], r.id[2].strip()): k for k, r in enumerate(res0)}
        h = anarci_heavy(seq)
        if h is None:
            rec['note'] = 'no heavy domain'; return rec
        d0, d1, num = h
        model = next(iter(st)); chobj = model[ch]
        keep = [r for r in chobj if r.id[0] == ' ' and (r.id[1], r.id[2].strip()) in amap and d0 - 3 <= amap[(r.id[1], r.id[2].strip())] <= d1 + 3]
        keepids = {r.id for r in keep}
        tst = Structure(rec['id']); tst.header = dict(getattr(st, 'header', {}) or {}); tm = Model(model.id, model.serial_num); tst.add(tm)
        tc = chobj.copy()
        for r in list(tc):
            if r.id not in keepids: tc.detach_child(r.id)
        tm.add(tc)
        kw = dict(chain=ch, chain_type='H')
        if md is not None: kw['mmcif_dict'] = md
        new = renumber_structure(tst, **kw)
        sab = [(r.id[1], r.id[2].strip()) for r in next(iter(new))[ch]]
        olds = [r for r in tc]
        assert len(sab) == len(olds)
        for tag, hh in [('', h), ('_allsp', anarci_heavy(seq, None))]:
            if hh is None: continue
            d0, d1, num = hh
            rows = []
            for o, s in zip(olds, sab):
                if o.id[0] != ' ': continue
                si = amap.get((o.id[1], o.id[2].strip()))
                if si is None or not (d0 <= si <= d1): continue
                a = num[si - d0]
                rows.append((A3.get(o.get_resname(), 'X'), a, s))
            flags = [a != s for _, a, s in rows]
            rr = runs_of(flags)
            rec['n_res' + tag] = len(rows)
            rec['disagree' + tag] = ' '.join(f'{aa}{a[0]}{a[1]}/{s[0]}{s[1]}' for aa, a, s in rows if a != s)
            rec['runs' + tag] = ';'.join(f'{rows[i][1][0]}{rows[i][1][1]}-{rows[j][1][0]}{rows[j][1][1]}' for i, j in rr)
            # ANARCI-absent FR3/FR1 positions (pattern) under this ANARCI setting
            pos = {p for (p, i) in num if i == ''}
            miss = sorted(set(range(min(pos), 105)) - pos)
            rec['pattern' + tag + '_obs'] = '+'.join(str(k) for k in miss if (11 <= k <= 26) or (74 <= k <= 100))
        rec['seqres_V'] = seq[h[0]:h[1] + 1]
    except Exception as ex:
        rec['note'] = f'{type(ex).__name__}: {str(ex)[:200]}'
    return rec


if __name__ == '__main__':
    os.makedirs(f'{CA}/cif', exist_ok=True)
    for j in JOBS:
        if j[2] == 'pdb': fetch_cif(j[3])
    with Pool(2) as p:
        R = p.map(process, JOBS, chunksize=1)
    D = pd.DataFrame(R)
    D.to_csv(f'{OUT}/tables/rerun_residues.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
    print(D[['group', 'pattern', 'id', 'chain', 'pattern_obs', 'runs', 'disagree', 'pattern_allsp_obs', 'runs_allsp'] + (['note'] if 'note' in D else [])].to_string())
