"""Score every (chain, error window) unit against the ImmuneBuilder and IgFold predictions of its Fv.

For each unit:
  * dep_run: does the deposited V domain itself show a SAbR-vs-ANARCI run overlapping the window
    (recomputed here; for template-family chains this says whether that copy carries the error);
  * for each method: pred_run (detector run on the prediction overlapping the window, and its shift),
    ladder offsets of predicted vs deposited residues in the window, and a call:
      copies error        >=50% of window residues sit on the same residue as in the deposit (offset 0)
      canonical           >=50% at offset = delta (the register SAbR assigns)
      shifted, not copied >=50% at a nonzero offset of the same sign as delta
      poor superposition  framework RMSD > 2.5 A
      no ladder           fewer than 15 framework CA pairs outside the window
      unclear             otherwise
usage: h19_readout_full.py units.json preddir out.csv [shard k n]
"""
import sys, os, json, warnings, logging
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h19_readout import anarci_labels, kabsch, pred_map, detector_runs, SP
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1
CIF = f'{SP}/sabr/dl'
CDR = lambda p: 27 <= p <= 38 or 56 <= p <= 65 or 105 <= p <= 117
_cache = {}
def entry(pdb):
    if pdb not in _cache:
        _cache.clear()
        fn = f'{CIF}/{pdb}.cif'
        _cache[pdb] = (MMCIF2Dict(fn), MMCIFParser(QUIET=True).get_structure(pdb, fn)[0])
    return _cache[pdb]
def dep_domain(pdb, chain, vseq):
    """-> (Bio model with chain 'X' = deposited V-domain residues, {V index: CA})"""
    md, st = entry(pdb)
    strand = md['_pdbx_poly_seq_scheme.pdb_strand_id']; mon = md['_pdbx_poly_seq_scheme.mon_id']
    num = md['_pdbx_poly_seq_scheme.pdb_seq_num']; ins = md['_pdbx_poly_seq_scheme.pdb_ins_code']; seqid = md['_pdbx_poly_seq_scheme.seq_id']
    rows = [(int(i) - 1, (int(n), '' if c in '.?' else c), m) for s, i, n, c, m in zip(strand, seqid, num, ins, mon) if s == chain]
    full = ''.join(seq1(m) for _, _, m in rows); off = full.find(vseq)
    if off < 0: raise ValueError('V sequence not found in deposited chain')
    k2v = {k: i - off for i, k, _ in rows if off <= i < off + len(vseq)}
    S = Structure('d'); M = Model(0); S.add(M); X = Chain('X'); M.add(X); ca = {}
    for r in st[chain]:
        v = k2v.get((r.id[1], r.id[2].strip()))
        if v is None or r.id[0] != ' ': continue
        r2 = r.copy(); r2.detach_parent(); X.add(r2)
        if 'CA' in r: ca[v] = r['CA'].coord
    return M, ca
def runs_in(runs, lo, hi):
    return [r for r in (runs or []) if r[0] <= hi and r[1] >= lo]
def ladder(dep, pred, lab, lo, hi):
    dep = {k: v for k, v in dep.items() if k < len(lab)}
    use = [k for k in dep if k in pred and not CDR(lab[k][0]) and not (lo - 2 <= lab[k][0] <= hi + 2)]
    if len(use) < 15: return None, None
    P = np.array([pred[k] for k in use]); Q = np.array([dep[k] for k in use])
    R, pc, qc = kabsch(P, Q); pt = {k: (v - pc) @ R.T + qc for k, v in pred.items()}
    rms = float(np.sqrt(((np.array([pt[k] for k in use]) - Q) ** 2).sum(1).mean()))
    keys = list(pt); arr = np.array([pt[k] for k in keys]); sh = []
    for k in sorted(dep):
        if not (lo <= lab[k][0] <= hi): continue
        d = np.linalg.norm(arr - dep[k], axis=1); j = d.argmin()
        sh.append(keys[j] - k if d[j] <= 3.0 else None)
    return sh, rms
def call(sh, rms, delta):
    if sh is None: return 'no ladder'
    if rms > 2.5: return 'poor superposition'
    ok = [s for s in sh if s is not None]; n = len(sh)
    same = sum(s == 0 for s in ok)
    canon = sum(s == delta for s in ok) if delta else 0
    moved = sum(s != 0 and (delta is None or np.sign(s) == np.sign(delta)) for s in ok)
    if same >= max(2, 0.5 * n): return 'copies error'
    if canon >= max(2, 0.5 * n): return 'canonical'
    if moved >= max(2, 0.5 * n): return 'shifted, not copied'
    return 'unclear'
if __name__ == '__main__':
    units = json.load(open(sys.argv[1])); pdir = sys.argv[2]; outf = sys.argv[3]
    k, n = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
    units = [u for u in units if u.get('status') == 'ok']
    pdbs = sorted(set(u['pdb'] for u in units)); mine = set(p for i, p in enumerate(pdbs) if i % n == k)
    rows = []
    for u in sorted([u for u in units if u['pdb'] in mine], key=lambda u: u['pdb']):
        base = {kk: u.get(kk) for kk in ('source', 'family', 'call', 'pdb', 'chain', 'chain_type', 'mode', 'fv_id', 'release', 'method', 'res',
                                          'era', 'loc', 'kind', 'run_len', 'delta', 'mapfit_pct', 'fr_insertion', 'family_flagged')}
        if base['call'] is not None: base['template_call'] = base.pop('call')
        try:
            vseq = u['vseq']; ct = u['chain_type']; lab, _ = anarci_labels(vseq)
            lab = lab + [(999, '')] * (len(vseq) - len(lab))
            if u['source'] == 'template_family':
                a, b = u['seq_window']; lo, hi = lab[a][0], lab[min(b, len(lab) - 1)][0]
            else:
                lo, hi = u['imgt_start'], u['imgt_end']
            base.update(imgt_lo=lo, imgt_hi=hi)
            dm, dca = dep_domain(u['pdb'], u['chain'], vseq)
            dr = runs_in(detector_runs(dm, 'X', ct, vseq), lo, hi)
            base.update(dep_run=bool(dr), dep_run_shift=';'.join(f'{x[2]:+d}' for x in dr))
        except Exception as e:
            base.update(error=f'deposit: {e}'[:200]); rows.append(base); continue
        for meth in ('immunebuilder', 'igfold'):
            r = dict(base, predictor=meth)
            fn = f"{pdir}/{meth}/{u['fv_id']}.pdb"
            if not os.path.exists(fn): r['error'] = 'no prediction'; rows.append(r); continue
            try:
                st = PDBParser(QUIET=True).get_structure('p', fn)[0]
                pc = 'H' if ct == 'H' else 'L'
                pr = runs_in(detector_runs(st, pc, ct, vseq), lo, hi)
                pm = pred_map(st, pc, vseq)
                pca = {pm[i]: x['CA'].coord for i, x in enumerate([x for x in st[pc] if x.id[0] == ' ']) if 'CA' in x and pm[i] is not None}
                sh, rms = ladder(dca, pca, lab, lo, hi)
                r.update(pred_run=bool(pr), pred_run_shift=';'.join(f'{x[2]:+d}' for x in pr),
                         ladder=None if sh is None else ' '.join('.' if s is None else str(s) for s in sh),
                         fw_rmsd=None if rms is None else round(rms, 2), ladder_call=call(sh, rms, u.get('delta')))
            except Exception as e:
                r['error'] = f'{meth}: {e}'[:200]
            rows.append(r)
        if len(rows) % 50 < 3: print(k, len(rows), flush=True)
    pd.DataFrame(rows).to_csv(outf, index=False)
    print('done', k, len(rows))
