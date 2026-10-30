"""Derive a structure-consistent numbering from IMGT-threaded domains.

Pipeline per fold family:
  1. superpose every domain on a consensus model (TM-align first, then Kabsch on matched sites),
  2. assign residues to consensus sites with an order-preserving DP scored on CA distance and
     virtual-CB direction,
  3. update the consensus (mean site positions), demote diffuse/rare sites to loop numbers and
     create new sites where many domains place residues consistently,
  4. repeat until assignments are stable.
Remaining residues are numbered by the loop rule in `number_domain`.
"""
import glob, json, os, re, sys
from collections import Counter, defaultdict
import numpy as np
import tmtools

ROOT = os.path.join(os.path.dirname(__file__), "..")
BIG = os.path.join(ROOT, "big")


# ---------------------------------------------------------------- labels
def lab_key(l):
    """Sort key: integer, then letter suffix (structural sub-sites: A < B < ... < Z < AA), then decimal
    (loop insertions). 8 < 8.1 < 8A < 8A.1 < 8B < 9."""
    m = re.match(r"(-?\d+)([A-Z]*)(?:\.(\d+))?$", str(l))
    return (int(m.group(1)), (len(m.group(2)), m.group(2)), int(m.group(3) or 0))


def is_int(l):
    return re.fullmatch(r"-?\d+", str(l)) is not None


def imgt_order_key(doms):
    """Order key for IMGT labels: decimals n.k that precede n in the chain (1.x, 85.x, 112.x ...)
    sort before n, in descending k; all others sort after n in ascending k."""
    votes = defaultdict(Counter)
    for d in doms:
        pos = {l: i for i, l in enumerate(d["lab"])}
        for l, i in pos.items():
            m = re.fullmatch(r"(-?\d+)\.(\d+)", l)
            if m and m.group(1) in pos:
                votes[int(m.group(1))]["before" if i < pos[m.group(1)] else "after"] += 1
    before = {n for n, c in votes.items() if c["before"] > c["after"]}

    def key(l):
        n, a, k = lab_key(l)
        if k and n in before:
            return (n - 1, (99, "~"), 10 ** 6 - k)
        return (n, a, k)
    key.before = before
    return key


def canonicalize(cons):
    """Rename sites into the new convention: integer sites keep their number; all other sites are
    named by the loop rule against the integer sites (N-attached decimals, negatives at the N-terminus)."""
    ints = {i: n for i, n in enumerate(cons.names) if is_int(n)}
    lab = number_domain(len(cons.names), ints, list(ints.values()))
    new = [lab[i] for i in range(len(cons.names))]
    c = Consensus(new, cons.xyz, cons.cbu)
    c.stats = {lab[i]: cons.stats.get(n, {}) | {"imgt_name": n} for i, n in enumerate(cons.names)}
    return c, dict(zip(cons.names, new))


# ---------------------------------------------------------------- data
def subclass(d):
    lab, g = d["label"], d["gene"].split("*")[0]
    if lab == "F-DOMAIN":
        return "F:FN/TNC" if g in ("FN1", "TNC") else "F:other"
    if lab == "C-LIKE":
        if g == "B2M": return "C-LIKE:B2M"
        if g.startswith(("HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "HLA-G", "MH1", "H2-", "CD1", "MR1", "FCGRT", "HFE", "MIC", "AZGP1", "ULBP", "RAET")):
            return "C-LIKE:MHC-I-a3"
        if g.startswith(("HLA-D", "MH2", "H2-A", "H2-E")): return "C-LIKE:MHC-II"
        return "C-LIKE:IgSF"
    if lab.startswith("C-BETA"): return "C-BETA"
    if lab.startswith("C-GAMMA"): return "C-GAMMA"
    return lab


def virtual_cb(a):
    if not all(k in a for k in ("N", "CA", "C")):
        return None
    n, ca, c = (np.array(a[k]) for k in ("N", "CA", "C"))
    b, cc = ca - n, c - ca
    x = np.cross(b, cc)
    return -0.58273431 * x + 0.56802827 * b - 0.54067466 * cc + ca


def load_big(min_frac=0.8, max_res=3.0):
    doms = []
    raw = []
    for f in sorted(glob.glob(os.path.join(BIG, "dom", "*.json"))):
        raw += json.load(open(f))["domains"]
    small = os.path.join(ROOT, "domains.json")
    if os.path.exists(small):
        for d in json.load(open(small)):
            d.setdefault("start", 0); d.setdefault("src", ["audit1"]); raw.append(d)
    if True:
        for d in raw:
            if len(d["residues"]) < min_frac * d["n_seq"] or len(d["residues"]) < 50:
                continue
            if d["resolution"] and d["resolution"] > max_res:
                continue
            d["cls"] = subclass(d)
            d["seq"] = "".join(r["aa"] if r["aa"] != "X" else "G" for r in d["residues"])
            d["ca"] = np.array([r["atoms"]["CA"] for r in d["residues"]], float)
            cb = [virtual_cb(r["atoms"]) for r in d["residues"]]
            u = np.full((len(cb), 3), np.nan)
            for i, v in enumerate(cb):
                if v is not None:
                    w = v - d["ca"][i]; u[i] = w / np.linalg.norm(w)
            d["cbu"] = u
            d["lab"] = [r["imgt"] for r in d["residues"]]
            doms.append(d)
    # non-redundant: identical domain sequence + label kept once (best resolution)
    best = {}
    for d in sorted(doms, key=lambda d: d["resolution"] or 9):
        best.setdefault((d["label"], d["seq"]), d)
    out = list(best.values())
    ids = Counter()
    for d in out:
        base = f'{d["pdb"]}_{d["chain"]}_{d["label"]}_{d["start"]}'
        ids[base] += 1
        d["id"] = base if ids[base] == 1 else f"{base}#{ids[base]}"
    return out


# ---------------------------------------------------------------- geometry
def kabsch(P, Q, w=None):
    """R, t minimising |R P + t - Q|."""
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    pc, qc = (w[:, None] * P).sum(0), (w[:, None] * Q).sum(0)
    H = ((P - pc) * w[:, None]).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def dp_assign(S, thr=0.05):
    """Order-preserving max-score matching of rows (residues) to columns (sites); no gap cost."""
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
            out.append((i - 1, j - 1)); i -= 1; j -= 1
    return out[::-1]


class Consensus:
    def __init__(self, names, xyz, cbu, kind=None, key=None):
        key = key or lab_key
        self.key = key
        order = sorted(range(len(names)), key=lambda k: key(names[k]))
        self.names = [names[k] for k in order]
        self.xyz = np.asarray(xyz)[order]
        self.cbu = np.asarray(cbu)[order]
        self.kind = kind or {}
        self.stats = {}

    def to_json(self):
        return dict(names=self.names, xyz=self.xyz.round(3).tolist(), cbu=np.nan_to_num(self.cbu).round(3).tolist(),
                    kind=self.kind, stats=self.stats)

    @classmethod
    def from_json(cls, j):
        c = cls(j["names"], j["xyz"], j["cbu"], j.get("kind")); c.stats = j.get("stats", {}); return c


def assign(d, cons, sigma=1.5, dmax=3.0, gamma=1.0, init=None, rounds=3):
    """Superpose domain d on consensus and assign residues to sites. Returns (matches, R, t)."""
    X, U = d["ca"], d["cbu"]
    if init is None:
        r = tmtools.tm_align(X, cons.xyz, d["seq"], "A" * len(cons.xyz))
        R, t = r.u, r.t
    else:
        R, t = init
    matches = []
    for _ in range(rounds):
        Xs = X @ R.T + t
        Us = U @ R.T
        D = np.linalg.norm(Xs[:, None, :] - cons.xyz[None, :, :], axis=2)
        cosang = np.nan_to_num(Us @ cons.cbu.T, nan=0.5)
        S = np.exp(-D ** 2 / (2 * sigma ** 2)) * np.clip((1 + cosang) / 2, 0, 1) ** gamma
        S[D > dmax] = 0
        pairs = dp_assign(S)
        matches = [(i, j, float(D[i, j])) for i, j in pairs]
        if len(matches) < 12:
            break
        P = X[[i for i, _, _ in matches]]; Q = cons.xyz[[j for _, j, _ in matches]]
        w = np.array([np.exp(-dd ** 2 / (2 * 2.0 ** 2)) for _, _, dd in matches]) + 1e-3
        R, t = kabsch(P, Q, w)
    return matches, R, t


def build_consensus(doms, seed, iters=6, occ_min=0.10, min_count=6, spread_max=2.0, birth_spread=1.6,
                    sigma=1.5, dmax=3.0, log=print, key=None):
    """Iterative consensus; site names seeded from IMGT labels (ordered by `key`)."""
    cons = Consensus(seed["lab"], seed["ca"], np.nan_to_num(seed["cbu"]), key=key)
    N = len(doms)
    min_count = min(min_count, max(3, N // 2))
    xf = {}
    for it in range(iters):
        acc_xyz = defaultdict(list); acc_cb = defaultdict(list); unmatched = defaultdict(list)
        for d in doms:
            m, R, t = assign(d, cons, sigma, dmax, init=xf.get(d["id"]))
            xf[d["id"]] = (R, t)
            Xs = d["ca"] @ R.T + t; Us = d["cbu"] @ R.T
            got = set()
            for i, j, dd in m:
                acc_xyz[cons.names[j]].append(Xs[i]); got.add(i)
                if not np.isnan(Us[i]).any(): acc_cb[cons.names[j]].append(Us[i])
            for i, l in enumerate(d["lab"]):
                if i not in got:
                    unmatched[l].append(Xs[i])
        names, xyz, cbu, stats = [], [], [], {}
        for nm in cons.names:
            pts = np.array(acc_xyz.get(nm, np.zeros((0, 3))))
            if len(pts) < max(min_count, occ_min * N):
                continue
            mu = pts.mean(0); spread = float(np.sqrt(((pts - mu) ** 2).sum(1).mean()))
            if spread > spread_max:
                continue
            cb = np.array(acc_cb.get(nm, [[0, 0, 0]])).mean(0)
            names.append(nm); xyz.append(mu); cbu.append(cb / (np.linalg.norm(cb) + 1e-9))
            stats[nm] = dict(occ=len(pts) / N, spread=round(spread, 2), cb_order=round(float(np.linalg.norm(cb)), 2))
        demoted = [n for n in cons.names if n not in stats]
        born = []
        for l, pts in unmatched.items():
            if l in stats or len(pts) < max(min_count, occ_min * N):
                continue
            pts = np.array(pts); med = np.median(pts, 0)
            core = pts[np.linalg.norm(pts - med, axis=1) < 2.0]
            if len(core) < max(min_count, occ_min * N):
                continue
            mu = core.mean(0); spread = float(np.sqrt(((core - mu) ** 2).sum(1).mean()))
            if spread > birth_spread or (xyz and np.min(np.linalg.norm(np.array(xyz) - mu, axis=1)) < 2.8):
                continue
            names.append(l); xyz.append(mu); cbu.append(np.zeros(3)); born.append(l)
            stats[l] = dict(occ=len(core) / N, spread=round(spread, 2), cb_order=0.0, born=it)
        log(f"  iter {it}: sites {len(names)} demoted {sorted(demoted, key=lab_key)} born {sorted(born, key=lab_key)}")
        cons = Consensus(names, xyz, cbu, key=key); cons.stats = stats
        if not demoted and not born and it >= 2:
            break
    return cons, xf


# ---------------------------------------------------------------- numbering rule
def _decimals(base, taken, k):
    """k fresh insertion labels base.1, base.2, ... (base may carry a letter suffix)."""
    m = re.match(r"(-?\d+[A-Z]*)(?:\.(\d+))?$", str(base))
    root, start = m.group(1), int(m.group(2) or 0)
    used = [int(t.split(".")[1]) for t in taken if "." in str(t) and str(t).split(".")[0] == root]
    start = max([start] + used)
    return [f"{root}.{start + q + 1}" for q in range(k)]


def number_domain(n_res, site_of, site_names, frame=None, exclude=()):
    """Full numbering of one domain from its structural anchor assignment.

    frame: ordered list of all names of the frame (anchors and loop names). Defaults to the
    anchors plus every integer between them. Rules:
      * matched residues take their anchor name;
      * a loop between matched anchors a < b may use the non-anchor frame names lying after a up
        to the next anchor, and before b back to the previous anchor (identical sets when a and b
        are consecutive anchors), i.e. a loop never borrows names that belong to a skipped strand;
      * loop residues fill those names alternately from both ends (gaps fall at the loop centre;
        the N side takes the extra residue when the count is odd);
      * surplus residues are insertions numbered n.1, n.2, ... on the last name used on the N side
        (the loop apex); insertions are always N-attached;
      * N-terminal residues count down through free integers (0 and negatives allowed),
        C-terminal residues count up beyond the last anchor.
    """
    own = set(site_names)
    anchors = own | set(exclude)
    if frame is None:
        ints = [lab_key(x)[0] for x in site_names]
        frame = sorted(set(site_names) | {str(k) for k in range(min(ints), max(ints) + 1)}, key=lab_key)
    pos = {nm: k for k, nm in enumerate(frame)}
    idx = sorted(site_of)
    lab = {i: site_of[i] for i in idx}
    if not idx:
        return lab
    used = set(lab.values())
    first, last = idx[0], idx[-1]
    n0 = lab_key(site_of[first])[0]
    free = [str(k) for k in range(n0 - 1, n0 - 400, -1) if str(k) not in anchors]
    for off, i in enumerate(range(first - 1, -1, -1)):
        lab[i] = free[off]
    n1 = lab_key(site_of[last])[0]
    free = [str(k) for k in range(n1 + 1, n1 + 400) if str(k) not in anchors]
    for off, i in enumerate(range(last + 1, n_res)):
        lab[i] = free[off]
    anchor_pos = sorted(pos[a] for a in own if a in pos)
    for a, b in zip(idx, idx[1:]):
        res = list(range(a + 1, b))
        if not res:
            continue
        pa, pb = pos[site_of[a]], pos[site_of[b]]
        nxt = next((q for q in anchor_pos if q > pa), pb)
        prv = next((q for q in reversed(anchor_pos) if q < pb), pa)
        if nxt >= pb:
            slots = [frame[q] for q in range(pa + 1, pb) if frame[q] not in anchors]
            split = None
        else:
            left = [frame[q] for q in range(pa + 1, nxt) if frame[q] not in anchors]
            right = [frame[q] for q in range(prv + 1, pb) if frame[q] not in anchors and q >= nxt]
            slots = left + right
            split = len(left)
        k, m = len(res), len(slots)
        n_fill = min(k, m)
        n_left = (n_fill + 1) // 2 if split is None else (min(split, (n_fill + 1) // 2) if n_fill < m else split)
        n_right = n_fill - n_left
        for q in range(n_left):
            lab[res[q]] = slots[q]
        for q in range(n_right):
            lab[res[k - 1 - q]] = slots[m - 1 - q]
        extra = res[n_left:k - n_right]
        if extra:
            base = slots[n_left - 1] if n_left else site_of[a]
            taken = set(lab.values()) | set(frame)
            for i, nm in zip(extra, _decimals(base, taken, len(extra))):
                lab[i] = nm
    return lab


# ---------------------------------------------------------------- DSSP-lite
def dssp_lite(d, partners=False):
    """E/H/- per residue from backbone H-bonds (Kabsch-Sander energy); optionally bridge partners."""
    A = [r["atoms"] for r in d["residues"]]
    n = len(A)
    N = np.array([a.get("N", [np.nan] * 3) for a in A], float)
    C = np.array([a.get("C", [np.nan] * 3) for a in A], float)
    O = np.array([a.get("O", [np.nan] * 3) for a in A], float)
    H = N.copy()
    co = C[:-1] - O[:-1]
    with np.errstate(invalid="ignore"):
        H[1:] = N[1:] + co / np.linalg.norm(co, axis=1)[:, None]
    def dist(X, Y):
        return np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        E = 0.084 * 332 * (1 / dist(O, N) + 1 / dist(C, H) - 1 / dist(O, H) - 1 / dist(C, N))
    hb = np.nan_to_num(E, nan=0) < -0.5        # hb[i, j]: C=O(i) ... H-N(j)
    for k in (0, 1, 2):
        idx = np.arange(n - k); hb[idx, idx + k] = False
    ss = np.array(["-"] * n)
    hel = np.zeros(n, bool)
    hel[1:n - 4] = hb[np.arange(0, n - 5), np.arange(4, n - 1)] & hb[np.arange(1, n - 4), np.arange(5, n)]
    for i in np.where(hel)[0]:
        ss[i:i + 4] = "H"
    Z = np.zeros((n, n), bool)
    def sh(M, di, dj):
        out = np.zeros_like(M)
        rs = slice(max(0, -di), n - max(0, di)); rd = slice(max(0, di), n - max(0, -di))
        cs = slice(max(0, -dj), n - max(0, dj)); cd = slice(max(0, dj), n - max(0, -dj))
        out[rd, cd] = M[rs, cs]
        return out
    hbT = hb.T
    # parallel: hb(i-1, j) & hb(j, i+1)  or  hb(j-1, i) & hb(i, j+1)
    par = (sh(hb, 1, 0) & sh(hbT, -1, 0)) | (sh(hbT, 0, 1) & sh(hb, 0, -1))
    # antiparallel: hb(i, j) & hb(j, i)  or  hb(i-1, j+1) & hb(j-1, i+1)
    anti = (hb & hbT) | (sh(hb, 1, -1) & sh(hbT, -1, 1))
    br = (par | anti)
    ii, jj = np.indices((n, n))
    br &= np.abs(ii - jj) >= 3
    isE = br.any(1) & (ss != "H")
    ss[isE] = "E"
    if not partners:
        return "".join(ss)
    part = np.where(br.any(1), br.argmax(1), -1)
    return "".join(ss), part
