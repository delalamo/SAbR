"""Shared helpers: minimal mmCIF atom_site reader, geometry, SAbR record lookup."""
import gzip, json, os, numpy as np

SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
C = f'{SP}/deeper/cache/struct_unusual'
OUTD = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual'
TAB = f'{OUTD}/tables'

HEAVY = {'GLY': 4, 'ALA': 5, 'SER': 6, 'CYS': 6, 'VAL': 7, 'THR': 7, 'PRO': 7, 'ILE': 8, 'LEU': 8, 'ASP': 8,
         'ASN': 8, 'GLU': 9, 'GLN': 9, 'LYS': 9, 'MET': 8, 'HIS': 10, 'PHE': 11, 'ARG': 11, 'TYR': 12, 'TRP': 14}
THREE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
         'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
         'TYR': 'Y', 'VAL': 'V', 'MSE': 'M'}


def cif_path(pdb):
    p = f'{SP}/sabr/dl/{pdb}.cif'
    if os.path.exists(p):
        return p
    return f'{C}/cif/{pdb}.cif.gz'


def read_atoms(path, chains=None):
    """Return {auth_chain: [(auth_seq:int, icode:str, resname, {atom: xyz})...]} for model 1, first altloc."""
    op = gzip.open if path.endswith('.gz') else open
    fields, out, inloop, seen = [], {}, False, {}
    with op(path, 'rt') as f:
        for line in f:
            if line.startswith('_atom_site.'):
                fields.append(line.strip().split('.', 1)[1]); inloop = True; continue
            if inloop and fields and (line.startswith('ATOM') or line.startswith('HETATM')):
                if not isinstance(fields, dict):
                    fields = {k: i for i, k in enumerate(fields)}
                t = line.split()
                if t[fields['pdbx_PDB_model_num']] not in ('1',):
                    continue
                ch = t[fields['auth_asym_id']]
                if chains is not None and ch not in chains:
                    continue
                alt = t[fields['label_alt_id']]
                rn = t[fields['auth_comp_id']]
                seq = int(t[fields['auth_seq_id']]); ic = t[fields['pdbx_PDB_ins_code']]
                ic = '' if ic in ('?', '.') else ic
                an = t[fields['auth_atom_id']].strip('"')
                key = (ch, seq, ic)
                lst = out.setdefault(ch, [])
                if key not in seen:
                    seen[key] = len(lst); lst.append([seq, ic, rn, {}])
                res = lst[seen[key]]
                if an in res[3]:
                    continue  # keep first altloc
                if t[fields['type_symbol']] == 'H':
                    continue
                res[3][an] = np.array([float(t[fields['Cartn_x']]), float(t[fields['Cartn_y']]),
                                       float(t[fields['Cartn_z']])])
            elif inloop and isinstance(fields, dict) and (line.startswith('#') or line.startswith('loop_')):
                break
    return out


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w); y = np.dot(np.cross(b1n, v), w)
    return np.degrees(np.arctan2(y, x))


def angle(a, b, c):
    u, v = a - b, c - b
    return np.degrees(np.arccos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))


def kabsch(P, Q):
    """Return R, t minimising |(P @ R + t) - Q| (rows are points)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    dd = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, dd])
    R = U @ D @ Vt
    return R, qc - pc @ R


def rmsd_after(P, Q):
    R, t = kabsch(P, Q)
    return float(np.sqrt(((P @ R + t - Q) ** 2).sum(1).mean())), R, t


def load_sabr(keys):
    """keys: set of (pdb, chain). Returns {(pdb,chain): record}. Later files override earlier ones."""
    out = {}
    for fn in ['validation', 'snapshot_rest', 'new']:
        with open(f'{SP}/sabr/out/{fn}.jsonl') as f:
            for line in f:
                i = line.find('"chain": "')
                p = line[9:13]
                ch = line[i + 10:line.index('"', i + 10)]
                if (p, ch) in keys:
                    out[(p, ch)] = json.loads(line)
    return out


# ---------------- ABangle (Dunbar et al. 2013), published reference frame -----------------
_AB = f'{C}/abangle_src/ABangle/data'
_CORE = json.load(open(f'{_AB}/coresets.json'))
_PC = {'H': np.loadtxt(f'{_AB}/principal_components_heavy.csv'), 'L': np.loadtxt(f'{_AB}/principal_components_light.csv')}


def _consensus(chain):
    out = {}
    for line in open(f'{_AB}/consensus_{chain}.pdb'):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            out[int(line[22:26])] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return out


_CONS = {'H': _consensus('H'), 'L': _consensus('L')}
_C2I = None


def core_imgt(ct):
    """Return {chothia_core_pos: imgt_pos} for chain type H/K/L (from s03_imgt2chothia)."""
    global _C2I
    if _C2I is None:
        _C2I = json.load(open(f'{TAB}/chothia_core_to_imgt.json'))
    return {int(k): v for k, v in _C2I[ct].items()}


def _frame(ca_by_imgt, ct, chain):
    """Superimpose the consensus core onto the structure core (CA by IMGT position, no insertion) and map the
    plane points. Returns (C, V1, V2, n_core_used, core_rmsd)."""
    m = core_imgt(ct)
    P, Q = [], []
    for cpos in _CORE[chain]:
        ip = m[cpos]
        if ip in ca_by_imgt and cpos in _CONS[chain]:
            P.append(_CONS[chain][cpos]); Q.append(ca_by_imgt[ip])
    if len(P) < 25:
        return None
    P, Q = np.array(P), np.array(Q)
    rms, R, t = rmsd_after(P, Q)
    pcs = _PC[chain]
    coefs = np.array([-5, 0.5, 1]) if chain == 'H' else np.array([3, -1, 1])
    Cp = coefs @ pcs
    pts = [v @ R + t for v in (Cp, Cp + pcs[0], Cp + pcs[1])]
    return pts + [len(P), rms]


def abangle(caH, caL, ctL):
    """caH/caL: {imgt_pos: CA xyz}. ctL: 'K' or 'L'. Returns dict of HL,HC1,LC1,HC2,LC2,dc (+core fit info)."""
    fh = _frame(caH, 'H', 'H'); fl = _frame(caL, ctL, 'L')
    if fh is None or fl is None:
        return None
    u = lambda v: v / np.linalg.norm(v)
    Cv = u(fh[0] - fl[0]); L1 = u(fl[1] - fl[0]); L2 = u(fl[2] - fl[0]); H1 = u(fh[1] - fh[0]); H2 = u(fh[2] - fh[0])
    dc = np.linalg.norm(fh[0] - fl[0])
    nx = np.cross(L1, Cv); ny = np.cross(Cv, nx)
    tL = u(np.array([0, L1 @ nx, L1 @ ny])); tH = u(np.array([0, H1 @ nx, H1 @ ny]))
    HL = np.degrees(np.arccos(np.clip(tL @ tH, -1, 1)))
    if np.cross(tL, tH) @ np.array([1, 0, 0]) < 0:
        HL = -HL
    ang = lambda a, b: np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
    return dict(HL=HL, HC1=ang(H1, -Cv), LC1=ang(L1, Cv), HC2=ang(H2, -Cv), LC2=ang(L2, Cv), dc=dc,
                nH=fh[3], nL=fl[3], rmsH=fh[4], rmsL=fl[4])
