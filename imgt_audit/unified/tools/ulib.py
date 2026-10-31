"""Unified Ig-fold numbering experiments: dataset, reference profile, alignment, loop rule, metrics."""
import glob, gzip, hashlib, json, os, re, sys
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, "/home/user/SAbR/src")
from sabr.alignment import _align_reference

W = "/tmp/sabr_unified"
RENUM = "/home/user/SAbR/imgt_audit/results_large/renumbered.json.gz"
FRAMES = "/home/user/SAbR/imgt_audit/results_large/frames.json"


# ------------------------------------------------------------------ labels
def lab_key(l):
    m = re.match(r"(-?\d+)([A-Z]*)(?:\.(\d+))?$", str(l))
    return (int(m.group(1)), (len(m.group(2)), m.group(2)), int(m.group(3) or 0))


def is_int(l):
    return re.fullmatch(r"-?\d+", str(l)) is not None


def fill_loops(n, site_of):
    """Class-agnostic loop rule.

    site_of: {residue index: anchor label}. Residues between two matched anchors take the
    integers strictly between them, filled alternately from both ends (gaps at the centre);
    surplus residues are N-attached insertions on the last N-side number (n.1, n.2, ...).
    Letter sites are never loop slots. Termini count outward through integers.
    """
    idx = sorted(site_of)
    lab = dict(site_of)
    if not idx:
        return lab
    used = set(site_of.values())
    first, last = idx[0], idx[-1]
    k0 = lab_key(site_of[first])[0]
    for off, i in enumerate(range(first - 1, -1, -1)):
        lab[i] = str(k0 - 1 - off)
    k1 = lab_key(site_of[last])[0]
    for off, i in enumerate(range(last + 1, n)):
        lab[i] = str(k1 + 1 + off)
    for a, b in zip(idx, idx[1:]):
        res = list(range(a + 1, b))
        if not res:
            continue
        ka, kb = lab_key(site_of[a]), lab_key(site_of[b])
        lo = ka[0] + 1
        slots = [str(k) for k in range(lo, kb[0]) if str(k) not in used]
        k, m = len(res), len(slots)
        nf = min(k, m)
        nl = (nf + 1) // 2
        nr = nf - nl
        for q in range(nl):
            lab[res[q]] = slots[q]
        for q in range(nr):
            lab[res[k - 1 - q]] = slots[m - 1 - q]
        extra = res[nl:k - nr]
        base = slots[nl - 1] if nl else site_of[a]
        root = re.match(r"(-?\d+[A-Z]*)", base).group(1)
        start = int(base.split(".")[1]) if "." in base else 0
        for q, i in enumerate(extra):
            lab[i] = f"{root}.{start + q + 1}"
    return lab


# ------------------------------------------------------------------ data
def split_of(pdb, frac_test=0.2):
    h = int(hashlib.sha1(pdb.encode()).hexdigest(), 16) % 1000
    return "test" if h < frac_test * 1000 else "train"


def load_dataset():
    recs = [r for r in json.load(gzip.open(RENUM, "rt")) if r["domtype"] in "VCF"]
    cache = {}
    out, miss = [], Counter()
    for r in recs:
        key = (r["pdb"], r["chain"])
        if key not in cache:
            p = f"{W}/emb/{r['pdb']}_{r['chain']}.npz"
            cache[key] = dict(np.load(p)) if os.path.exists(p) else None
        c = cache[key]
        if c is None:
            miss["nochain"] += 1; continue
        pos = {x: i for i, x in enumerate(c["ids"])}
        rows = [pos.get(a) for a in r["auth"]]
        keep = [k for k, x in enumerate(rows) if x is not None]
        if len(keep) < 0.9 * len(rows):
            miss["unmapped"] += 1; continue
        rows = [rows[k] for k in keep]
        # domain residues must be contiguous in the chain arrays
        if rows != list(range(rows[0], rows[0] + len(rows))):
            miss["noncontig"] += 1; continue
        site_of = {j: r["new"][k] for j, k in enumerate(keep) if r["site"][k]}
        d = dict(id=r["id"], pdb=r["pdb"], chain=r["chain"], cls=r["cls"], domtype=r["domtype"],
                 rows=(rows[0], rows[-1] + 1), imgt=[r["imgt"][k] for k in keep],
                 gt_sites=site_of, split=split_of(r["pdb"]))
        d["target"] = [fill_loops(len(keep), site_of)[j] for j in range(len(keep))]
        out.append(d)
    return out, cache, miss


def dom_emb(d, cache):
    c = cache[(d["pdb"], d["chain"])]
    a, b = d["rows"]
    return c["emb"][a:b]


def dom_coords(d, cache):
    c = cache[(d["pdb"], d["chain"])]
    a, b = d["rows"]
    return c["coords"][a:b]


# ------------------------------------------------------------------ reference
def anchor_list():
    fr = json.load(open(FRAMES))
    return sorted(fr["anchors_Ig"], key=lab_key)


def build_profile(doms, cache, anchors, balance=True, min_count=5, rescale=True):
    """Per-anchor mean embedding over training domains (optionally class-balanced)."""
    idx = {a: k for k, a in enumerate(anchors)}
    sums = defaultdict(lambda: defaultdict(lambda: np.zeros(64)))
    cnts = defaultdict(Counter)
    norms = defaultdict(list)
    for d in doms:
        E = dom_emb(d, cache)
        for j, a in d["gt_sites"].items():
            if a in idx:
                sums[a][d["cls"]] += E[j]; cnts[a][d["cls"]] += 1
                norms[a].append(np.linalg.norm(E[j]))
    keep, vecs = [], []
    for a in anchors:
        n = sum(cnts[a].values())
        if n < min_count:
            continue
        if balance:
            v = np.mean([sums[a][c] / cnts[a][c] for c in cnts[a]], axis=0)
        else:
            v = sum(sums[a].values()) / n
        if rescale:
            v = v / np.linalg.norm(v) * np.median(norms[a])
        keep.append(a); vecs.append(v)
    return keep, np.array(vecs, np.float32)


# ------------------------------------------------------------------ numbering
def assign(E, names, prof, gaps=frozenset()):
    soft, sim, score = _align_reference(E, prof, query_gap_indices=frozenset(gaps))
    hard = soft > 0.5
    site_of = {}
    for i, j in zip(*np.nonzero(hard)):
        site_of[int(i)] = names[int(j)]
    return site_of, score


def number(E, names, prof):
    site_of, score = assign(E, names, prof)
    lab = fill_loops(len(E), site_of)
    return [lab[i] for i in range(len(E))], site_of, score


# ------------------------------------------------------------------ metrics
def compare(d, pred_sites, pred_labels):
    gt = d["gt_sites"]
    tp = sum(1 for i, a in pred_sites.items() if gt.get(i) == a)
    return dict(n=len(d["target"]), gt_anchors=len(gt), pred_anchors=len(pred_sites), anchor_tp=tp,
                label_eq=sum(p == t for p, t in zip(pred_labels, d["target"])))


# ------------------------------------------------------------------ discriminative reference
def training_matrix(doms, cache, names, balance=True):
    idx = {a: k for k, a in enumerate(names)}
    X, y, w = [], [], []
    ncls = Counter()
    for d in doms:
        ncls[d["cls"]] += len(d["target"])
    for d in doms:
        E = dom_emb(d, cache)
        wt = (1.0 / ncls[d["cls"]]) if balance else 1.0
        for j in range(len(E)):
            a = d["gt_sites"].get(j)
            X.append(E[j]); y.append(idx.get(a, len(names)) if a is not None else len(names)); w.append(wt)
    X = np.array(X, np.float64); y = np.array(y); w = np.array(w)
    return X, y, w / w.mean()


def load_negatives(pattern="/tmp/sabr_unified/neg/*.npz"):
    return [np.load(p)["emb"] for p in sorted(glob.glob(pattern))]


def train_profile(doms, cache, anchors, l2=1e-3, min_count=5, balance=True, maxiter=400, negatives=None, neg_weight=4.0):
    """Softmax regression over anchor sites + a loop class; reference = w_site - w_loop (log-odds).

    negatives: list of [n, 64] embeddings of non-Ig residues, labelled with the loop/background class;
    together they carry neg_weight times the weight of one domain class."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    cnt = Counter(a for d in doms for a in d["gt_sites"].values())
    names = [a for a in anchors if cnt[a] >= min_count]
    X, y, w = training_matrix(doms, cache, names, balance)
    if negatives:
        N = np.concatenate(negatives).astype(np.float64)
        n_cls = len({d["cls"] for d in doms})
        w_pos_total = w.sum()
        wn = np.full(len(N), neg_weight * w_pos_total / n_cls / len(N))
        X = np.concatenate([X, N]); y = np.concatenate([y, np.full(len(N), len(names))]); w = np.concatenate([w, wn])
        w = w / w.mean()
    K, D = len(names) + 1, X.shape[1]
    Y = np.zeros((len(y), K)); Y[np.arange(len(y)), y] = 1

    def f(theta):
        Wt = theta.reshape(K, D)
        Z = X @ Wt.T
        lse = logsumexp(Z, axis=1)
        P = np.exp(Z - lse[:, None])
        loss = np.sum(w * (lse - np.sum(Z * Y, 1))) / len(y) + l2 * np.sum(Wt ** 2)
        G = ((P - Y) * w[:, None]).T @ X / len(y) + 2 * l2 * Wt
        return loss, G.ravel()

    res = minimize(f, np.zeros(K * D), jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    Wt = res.x.reshape(K, D)
    ref = (Wt[:-1] - Wt[-1]).astype(np.float32)
    acc = np.mean(np.argmax(X @ Wt.T, 1) == y)
    return names, ref, dict(loss=float(res.fun), train_residue_acc=float(acc), iters=int(res.nit))
