"""Per-domain atom coordinates for the 3D viewer, all superposed onto one reference (VH) with TM-align."""
import sys, json
sys.path[:0] = ['/tmp/pan', '/tmp/sabr_unified/tools']
import numpy as np, tmtools, ulib
from Bio.PDB import MMCIFParser
D = json.load(open('/tmp/pan/viz/data.json'))
doms, cache, _ = ulib.load_dataset(); byid = {d['id']: d for d in doms}
parser = MMCIFParser(QUIET=True)

def residues(d):
    c = cache[(d['pdb'], d['chain'])]; a, b = d['rows']
    model = next(iter(parser.get_structure(d['pdb'], f"/tmp/sabr_unified/cif/{d['pdb']}.cif")))
    ch = {f"{r.id[1]}{r.id[2].strip()}": r for r in model[d['chain']]}
    return [ch[str(x)] for x in c['ids'][a:b]]

ref = None; out = {}
order = sorted(D['domains'], key=lambda x: x['cls'] != 'VH')   # VH first: it is the reference
for x in order:
    R = residues(byid[x['rep']])
    assert len(R) == len(x['residues']), x['cls']
    ca = np.array([r['CA'].coord for r in R], float)
    seq = "".join(r['aa'] for r in x['residues'])
    if ref is None:
        ref = (ca, seq); u, t, tm = np.eye(3), np.zeros(3), 1.0
    else:
        res = tmtools.tm_align(ca, ref[0], seq, ref[1]); u, t, tm = res.u, res.t, res.tm_norm_chain2
    lines = []
    k = 0
    for i, r in enumerate(R):
        for a in r:
            if a.element == 'H': continue
            X = u @ a.coord + t; k += 1
            name = a.get_id(); nm = f" {name:<3}" if len(name) < 4 else name
            lines.append(f"ATOM  {k:5d} {nm} {r.get_resname():>3} A{i + 1:4d}    {X[0]:8.3f}{X[1]:8.3f}{X[2]:8.3f}  1.00  0.00          {a.element:>2}")
    out[x['cls']] = dict(pdb="\n".join(lines) + "\nEND\n", tm=round(float(tm), 3))
    print(f"{x['cls']:18s} TM to VH {tm:.2f}  atoms {k}")
json.dump(out, open('/tmp/pan/viz/structures.json', 'w'), separators=(",", ":"))
