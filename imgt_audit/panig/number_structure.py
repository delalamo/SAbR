"""Pan-Ig numbering of a protein chain from its structure (algorithm M7, see ALGORITHMS.md).

Usage:
    python number_structure.py STRUCTURE.(pdb|cif) CHAIN [--min-anchors 30]

Every Ig-fold domain in the chain is found and numbered on the pan-Ig frame (PANIG.md).
Output is a TSV: residue id, amino acid, domain index, pan-Ig label, and whether the
label is a structural anchor claim ("A") or a deterministic positional fill ("-").

Requires numpy, biopython, tmtools and SAbR (for chain extraction).
"""
import argparse
import os
import sys

import numpy as np
import tmtools

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools"))
from number2 import Scheme  # noqa: E402

CORE = {str(i) for i in list(range(18, 26)) + list(range(39, 45)) + list(range(86, 92)) + list(range(99, 105))}


class Frame:
    def __init__(self, names, xyz, cbu):
        self.names = [str(n) for n in names]
        self.xyz = np.asarray(xyz, float)
        self.cbu = np.asarray(cbu, float)


def load_frames(path=os.path.join(HERE, "panig_reference.npz")):
    z = np.load(path)
    frames = {"pooled": Frame(z["frame_names"], z["frame_xyz"], z["frame_cbu"])}
    for k in z.files:
        if k.startswith("class_") and k.endswith("_names"):
            c = k[len("class_"):-len("_names")]
            frames[c] = Frame(z[k], z[f"class_{c}_xyz"], z[f"class_{c}_cbu"])
    return frames, sorted(int(a) for a in z["anchors"])


def kabsch(P, Q, w=None):
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    pc, qc = (w[:, None] * P).sum(0), (w[:, None] * Q).sum(0)
    U, _, Vt = np.linalg.svd(((P - pc) * w[:, None]).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def dp_assign(S, thr=0.05):
    """Order-preserving residue -> site assignment maximising the summed score."""
    n, m = S.shape
    F = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        cand = np.maximum(F[i - 1, 1:], F[i - 1, :-1] + np.where(S[i - 1] > thr, S[i - 1], -1))
        F[i, 1:] = np.maximum.accumulate(cand)
    i, j, out = n, m, []
    while i > 0 and j > 0:
        if F[i, j] == F[i - 1, j]:
            i -= 1
        elif F[i, j] == F[i, j - 1]:
            j -= 1
        else:
            out.append((i - 1, j - 1))
            i -= 1
            j -= 1
    return out[::-1]


def match_frame(ca, cbu, f, cbw, sigma=1.5, dmax=3.5, rounds=3):
    r = tmtools.tm_align(ca, f.xyz, "A" * len(ca), "A" * len(f.xyz))
    R, t = r.u, r.t
    pairs, Ss = [], None
    for _ in range(rounds):
        X, U = ca @ R.T + t, cbu @ R.T
        D = np.linalg.norm(X[:, None] - f.xyz[None], axis=2)
        Ss = np.exp(-D ** 2 / (2 * sigma ** 2)) * ((1 - cbw) + cbw * np.clip((1 + U @ f.cbu.T) / 2, 0, 1))
        Ss[D > dmax] = 0
        pairs = dp_assign(Ss)
        use = [(i, j) for i, j in pairs if f.names[j] in CORE]
        if len(use) >= 12:
            w = np.exp(-np.array([D[i, j] for i, j in use]) ** 2 / 8.0) + 1e-3
            R, t = kabsch(ca[[i for i, _ in use]], f.xyz[[j for _, j in use]], w)
    return sum(Ss[i, j] for i, j in pairs), {i: int(f.names[j]) for i, j in pairs}


def m7(ca, cbu, frames, anchors):
    """Best class frame by structural score; C-beta term halved for irregular backbones."""
    dd = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    dd = dd[dd < 4.5]
    cbw = 1.0 if (len(dd) < 5 or np.std(dd) < 0.3) else 0.5
    best = max((match_frame(ca, cbu, f, cbw) + (c,) for c, f in frames.items()), key=lambda r: r[0])
    aset = set(anchors)
    return {i: a for i, a in best[1].items() if a in aset}, best[2]


def number_chain(ca, cbu, frames, anchors, min_anchors=30, max_domains=8):
    """Find domains greedily (best-matching segment first), then number each with the pan-Ig scheme."""
    n = len(ca)
    free = np.ones(n, bool)
    domains = []
    for _ in range(max_domains):
        idx = np.nonzero(free)[0]
        if len(idx) < 40:
            break
        site, cls = m7(ca[idx], cbu[idx], frames, anchors)
        if len(site) < min_anchors:
            break
        rows = sorted(int(idx[i]) for i in site)
        lo, hi = rows[0], rows[-1]
        domains.append((lo, hi, {int(idx[i]): a for i, a in site.items()}, cls))
        free[lo:hi + 1] = False
    domains.sort()
    scheme = Scheme(anchors)
    labels = [None] * n
    dom_of = [None] * n
    for k, (lo, hi, site, cls) in enumerate(domains):
        a = (domains[k - 1][1] + lo) // 2 + 1 if k else 0           # split linkers at their midpoint
        b = (hi + domains[k + 1][0]) // 2 + 1 if k + 1 < len(domains) else n
        lab = scheme.number(b - a, {i - a: s for i, s in site.items() if a <= i < b})
        for i in range(a, b):
            labels[i] = lab[i - a]
            dom_of[i] = k
    claims = {i for _, _, s, _ in domains for i in s}
    return labels, dom_of, claims, [(lo, hi, cls) for lo, hi, _, cls in domains]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure")
    ap.add_argument("chain")
    ap.add_argument("--min-anchors", type=int, default=30)
    args = ap.parse_args()
    from Bio.PDB import MMCIFParser, PDBParser
    from sabr.structure import extract_chain
    parser = MMCIFParser(QUIET=True) if args.structure.endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)
    structure = parser.get_structure("q", args.structure)
    if len(structure) > 1:
        for model in list(structure)[1:]:
            structure.detach_child(model.id)
    data = extract_chain(structure, args.chain, None)
    X = np.asarray(data.coords, float)
    ca = X[:, 1]
    u = X[:, 3] - X[:, 1]
    cbu = u / np.linalg.norm(u, axis=1)[:, None]
    frames, anchors = load_frames()
    labels, dom_of, claims, doms = number_chain(ca, cbu, frames, anchors, args.min_anchors)
    for k, (lo, hi, cls) in enumerate(doms):
        print(f"# domain {k}: rows {lo}-{hi}, best-matching class frame {cls}")
    print("residue\taa\tdomain\tlabel\tclaim")
    for i, (rid, aa) in enumerate(zip(data.residue_ids, data.sequence)):
        print(f"{rid[0]}{rid[1].strip()}\t{aa}\t{'' if dom_of[i] is None else dom_of[i]}\t{labels[i] or ''}\t{'A' if i in claims else '-'}")


if __name__ == "__main__":
    main()
