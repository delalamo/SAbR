"""Build the structure-consistent numbering for all families, renumber every domain, evaluate."""
import json, os, random, re, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import tmtools
sys.path.insert(0, os.path.dirname(__file__))
from scheme import (_decimals, is_int, load_big, build_consensus, imgt_order_key, canonicalize, assign, number_domain, Consensus, kabsch, dp_assign,
                    lab_key, is_int, dssp_lite)

OUT = os.path.join(os.path.dirname(__file__), "..", "big", "scheme")
os.makedirs(OUT, exist_ok=True)
SPREAD = 1.6
CANON_V = {"VH", "V-KAPPA", "V-LAMBDA", "V-ALPHA", "V-BETA", "V-GAMMA", "V-DELTA"}
IG_CORE = [str(i) for i in list(range(18, 26)) + list(range(39, 45)) + list(range(86, 92)) + list(range(99, 105))]


def letters_to_decimal(doms):
    for d in doms:
        new = []
        for l in d["lab"]:
            m = re.fullmatch(r"(-?\d+)([A-Z])", l)
            new.append(f"{m.group(1)}.{ord(m.group(2)) - 64}" if m else l)
        d["lab"] = new


HYPERVARIABLE = {"V": [(27, 38), (56, 65), (105, 117)]}


def anchor_filter(cons, ds, log, tight=1.6, loose=1.6, ss_min=0.5, occ_loop=0.65, hv=()):
    """Keep sites that are rigid (spread <= tight) or regular secondary structure (spread <= loose
    and E/H in >= ss_min of matched residues). Everything else is numbered as loop."""
    tally = defaultdict(Counter)
    for d in ds:
        m, R, t = assign(d, cons)
        ss = d.get("_ss") or dssp_lite(d)
        d["_ss"] = ss
        for i, j, _ in m:
            tally[cons.names[j]][ss[i]] += 1
    keep, dropped = [], []
    for k, n in enumerate(cons.names):
        st = cons.stats.get(n, {})
        tot = sum(tally[n].values()) or 1
        frac = (tally[n]["E"] + tally[n]["H"]) / tot
        st["ss_frac"] = round(frac, 2)
        st["ss"] = tally[n].most_common(1)[0][0] if tally[n] else "-"
        sp = st.get("spread", 0)
        occ = st.get("occ", 0)
        in_hv = any(lo <= lab_key(n)[0] <= hi for lo, hi in hv)
        rigid_loop = sp <= tight and occ >= occ_loop and not in_hv
        (keep if rigid_loop or (sp <= loose and frac >= ss_min) else dropped).append(k)
    log(f"  anchor filter: kept {len(keep)} dropped {[cons.names[k] for k in dropped]}")
    c = Consensus([cons.names[k] for k in keep], cons.xyz[keep], cons.cbu[keep], key=cons.key)
    c.stats = {cons.names[k]: cons.stats.get(cons.names[k], {}) for k in keep}
    return c


def pick_seed(ds, cls):
    pool = [d for d in ds if d["cls"] == cls] or ds
    return max(pool, key=lambda d: (len(d["residues"]), -(d["resolution"] or 9)))


def merge_frame(uni, uni_anchors, other, other_anchors, core, log, tag):
    """Add consensus `other` to the union frame `uni` (positions in uni's coordinates).
    Positions of `other` within 3 A of a union position (order-preserving DP) take that name;
    each run of unmatched positions gets fresh N-attached decimals on the preceding union name."""
    shared = [n for n in core if n in uni.names and n in other.names]
    R, t = kabsch(np.array([other.xyz[other.names.index(n)] for n in shared]),
                  np.array([uni.xyz[uni.names.index(n)] for n in shared]))
    X = other.xyz @ R.T + t; U = other.cbu @ R.T
    D = np.linalg.norm(X[:, None] - uni.xyz[None], axis=2)
    S = np.exp(-D ** 2 / (2 * 1.5 ** 2)) * np.clip((1 + U @ uni.cbu.T) / 2, 0, 1)
    S[D > 3.0] = 0
    # names are shared only between anchors (strand/helix or rigid) of both types
    uni_a = np.array([n in uni_anchors for n in uni.names])
    oth_a = np.array([n in other_anchors for n in other.names])
    S[~oth_a, :] = 0
    S[:, ~uni_a] = 0
    pairs = dict(dp_assign(S))
    names = list(uni.names); xyz = list(uni.xyz); cbu = list(uni.cbu)
    taken = set(names)
    newname = {}
    prev = None
    run = []
    def flush(run, prev):
        if not run:
            return
        if prev is None:
            first_int = lab_key(newname.get(run[-1] + 1, uni.names[0]))[0] if run[-1] + 1 in newname else lab_key(uni.names[0])[0]
            nm = [str(k) for k in range(first_int - 1, first_int - 400, -1) if str(k) not in taken][:len(run)][::-1]
        else:
            nm = _decimals(prev, taken, len(run))
        for i, x in zip(run, nm):
            newname[i] = x; taken.add(x)
            names.append(x); xyz.append(X[i]); cbu.append(U[i])
    for i in range(len(other.names)):
        if i in pairs:
            flush(run, prev); run = []
            newname[i] = uni.names[pairs[i]]; prev = newname[i]
        else:
            run.append(i)
    if run:
        last_int = lab_key(prev)[0] if prev else 0
        nm = [str(k) for k in range(last_int + 1, last_int + 400) if str(k) not in taken][:len(run)]
        for i, x in zip(run, nm):
            newname[i] = x; taken.add(x); names.append(x); xyz.append(X[i]); cbu.append(U[i])
    mapping = {other.names[i]: newname[i] for i in range(len(other.names))}
    order = [lab_key(newname[i]) for i in range(len(other.names))]
    viol = sum(1 for a, b in zip(order, order[1:]) if b <= a)
    anchors = {k: dict(v) for k, v in uni_anchors.items()}
    for n, ss in other_anchors.items():
        a = anchors.setdefault(mapping[n], {"ss": False, "types": set()})
        a["ss"] = a["ss"] or ss; a["types"].add(tag.split()[0])
    log(f"  merge {tag}: core rmsd {np.sqrt(np.mean(np.sum((X[[other.names.index(n) for n in shared]] - uni.xyz[[uni.names.index(n) for n in shared]]) ** 2, 1))):.2f} A over {len(shared)}; "
        f"{len(pairs)} positions shared, {len(other.names) - len(pairs)} new; order violations {viol}")
    log(f"    renamed: {[(k, v) for k, v in mapping.items() if k != v]}")
    return Consensus(names, xyz, cbu), anchors, mapping


def make_frame(names):
    ints = [lab_key(x)[0] for x in names]
    return sorted(set(names) | {str(k) for k in range(min(ints), max(ints) + 1)}, key=lab_key)


def anchors_only(cons, anchors):
    keep = [k for k, n in enumerate(cons.names) if n in anchors]
    c = Consensus([cons.names[k] for k in keep], cons.xyz[keep], cons.cbu[keep])
    c.stats = {cons.names[k]: cons.stats.get(cons.names[k], {}) for k in keep}
    return c


def scoped_view(cons, anchors, t):
    """Anchors usable by a domain of type t: all strand/helix anchors plus t's own loop anchors."""
    names = [n for n in cons.names if anchors[n]["ss"] or t in anchors[n]["types"]]
    other = [n for n in cons.names if n not in names and not is_int(n)]
    return anchors_only(cons, set(names)), other


def refine_scoped(ds, cons, anchors, log, iters=3):
    xf = {}
    for it in range(iters):
        acc = defaultdict(list); accb = defaultdict(list)
        views = {t: scoped_view(cons, anchors, t)[0] for t in {d["domtype"] for d in ds}}
        for d in ds:
            v = views[d["domtype"]]
            m, R, t = assign(d, v, init=xf.get(d["id"]))
            xf[d["id"]] = (R, t)
            Xs = d["ca"] @ R.T + t; Us = d["cbu"] @ R.T
            for i, j, _ in m:
                acc[v.names[j]].append(Xs[i])
                if not np.isnan(Us[i]).any(): accb[v.names[j]].append(Us[i])
        xyz, cbu = [], []
        for k, n in enumerate(cons.names):
            xyz.append(np.mean(acc[n], 0) if acc.get(n) else cons.xyz[k])
            b = np.mean(accb[n], 0) if accb.get(n) else cons.cbu[k]
            cbu.append(b / (np.linalg.norm(b) + 1e-9))
        stats = {n: dict(occ=len(acc.get(n, [])) / len(ds)) for n in cons.names}
        cons = Consensus(cons.names, xyz, cbu); cons.stats = stats
        log(f"  scoped refine iter {it}")
    return cons


def renumber(ds, typed, frame, all_anchors, own_names=None):
    """own_names[t]: the decimal names belonging to type t; other types' decimals are not offered."""
    out = {}
    tframes = {}
    for d in ds:
        t = d["domtype"]
        v = typed[t]
        if t not in tframes:
            tframes[t] = [n for n in frame if is_int(n) or own_names is None or n in own_names[t]]
        frame_t = tframes[t]
        other = [n for n in all_anchors if n not in v.names and not is_int(n)]
        m, R, tt = assign(d, v, dmax=3.0)
        site_of = {i: v.names[j] for i, j, _ in m}
        lab = number_domain(len(d["residues"]), site_of, v.names, frame_t, exclude=other)
        out[d["id"]] = dict(new=[lab[i] for i in range(len(d["residues"]))], site=[i in site_of for i in range(len(d["residues"]))])
    return out


def _pair_eval(args):
    a, b = args
    r = tmtools.tm_align(a["ca"], b["ca"], a["seq"], b["seq"])
    tm = max(r.tm_norm_chain1, r.tm_norm_chain2)
    Xa = a["ca"] @ r.u.T + r.t
    eq = {}
    i = j = 0
    for ca, cb in zip(r.seqxA, r.seqyA):
        if ca != "-" and cb != "-" and np.linalg.norm(Xa[i] - b["ca"][j]) <= 3.0:
            eq[i] = j
        i += ca != "-"; j += cb != "-"
    res = {}
    for key in ("lab", "new"):
        la, lb = a[key], b[key]
        pos_b = {l: k for k, l in enumerate(lb)}
        eq_same = sum(la[i] == lb[j] for i, j in eq.items())
        same_lab = [(i, pos_b[l]) for i, l in enumerate(la) if l in pos_b]
        same_close = sum(np.linalg.norm(Xa[i] - b["ca"][j]) <= 3.0 for i, j in same_lab)
        # restricted to residues the new scheme calls structural sites in both domains
        site_pairs = [(i, pos_b[l]) for i, l in enumerate(a["new"]) if a["site"][i] and l in pos_b and b["site"][pos_b[l]]] if key == "new" else []
        res[key] = (len(eq), eq_same, len(same_lab), same_close)
        if key == "new":
            res["site_prec"] = (len(site_pairs), sum(np.linalg.norm(Xa[i] - b["ca"][j]) <= 3.0 for i, j in site_pairs))
    return a["cls"], b["cls"], tm, res


def evaluate(ds, k=20, seed=0, label=""):
    rnd = random.Random(seed)
    slim = [dict(id=d["id"], cls=d["cls"], ca=d["ca"], seq=d["seq"], lab=d["lab"], new=d["new"], site=d["site"]) for d in ds]
    jobs = []
    for i, a in enumerate(slim):
        for b in rnd.sample(slim, min(k, len(slim))):
            if b["id"] != a["id"]:
                jobs.append((a, b))
    with Pool(4) as p:
        rows = p.map(_pair_eval, jobs, chunksize=20)
    agg = defaultdict(lambda: np.zeros(10))
    for ca, cb, tm, res in rows:
        if tm < 0.5:
            continue
        key = tuple(sorted((ca, cb)))
        v = np.array(list(res["lab"]) + list(res["new"]) + list(res["site_prec"]), float)
        agg[key] += v; agg["ALL"] += v
    return agg


def fmt(v):
    n_eq, imgt_same, n_lab, imgt_close, _, new_same, n_new, new_close, n_site, site_close = v
    return (f"recall IMGT {imgt_same/max(n_eq,1):.3f} new {new_same/max(n_eq,1):.3f} | "
            f"precision IMGT {imgt_close/max(n_lab,1):.3f} new {new_close/max(n_new,1):.3f} | "
            f"site-precision new {site_close/max(n_site,1):.3f}")


def main():
    t0 = time.time()
    logf = open(os.path.join(OUT, "log.txt"), "w")
    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n")
    doms = load_big()
    letters_to_decimal(doms)
    log("domains", len(doms), Counter(d["domtype"] for d in doms))
    fam = defaultdict(list)
    for d in doms:
        fam[d["domtype"]].append(d)
    for t, ds in fam.items():
        log(t, dict(Counter(d["cls"] for d in ds)))
    cons, anch = {}, {}
    # --- per-type consensus
    build = {
        "V": ([d for d in fam["V"] if d["cls"] in CANON_V], "V-KAPPA"),
        "C": ([d for d in fam["C"] if d["cls"] not in ("C-LIKE:IgSF", "C-ALPHA")], "C-KAPPA"),
        "F": (fam["F"], "F:FN/TNC"),
        "G": (fam["G"], "G-ALPHA1"),
        "S": (fam["S"], "S-DOMAIN"),
        "A": (fam["A"], "A-DOMAIN"),
    }
    for t, (ds, seedcls) in build.items():
        if len(ds) < 5:
            continue
        log(f"== consensus {t}: {len(ds)} domains")
        key = imgt_order_key(fam[t])
        log(f"  IMGT decimals attached to the following integer: {sorted(key.before)}")
        full, _ = build_consensus(ds, pick_seed(ds, seedcls), spread_max=SPREAD, log=log, key=key)
        anc = anchor_filter(full, ds, log, hv=HYPERVARIABLE.get(t, ()))
        full_c, ren = canonicalize(full)
        cons[t] = full_c
        anch[t] = {ren[n]: anc.stats.get(n, {}).get("ss_frac", 0) >= 0.5 for n in anc.names}
        for n in full_c.names:
            full_c.stats[n] = {**full_c.stats.get(n, {}), **anc.stats.get(full_c.stats.get(n, {}).get("imgt_name", n), {})}
        log(f"  canonical names: {[(a, b) for a, b in ren.items() if a != b]}")
    # --- one Ig frame: V is the reference; C and FN3 are expressed in it
    ig_anch = {n: {"ss": ss, "types": {"V"}} for n, ss in anch["V"].items()}
    ig, maps = cons["V"], {}
    for t in ("C", "F"):
        if t in cons:
            ig, ig_anch, maps[t] = merge_frame(ig, ig_anch, cons[t], anch[t], IG_CORE, log, f"{t} into Ig")
    frames = {"Ig": make_frame(ig.names)}
    ig_doms = fam["V"] + fam["C"] + fam["F"]
    n_ss = sum(v["ss"] for v in ig_anch.values())
    log(f"== Ig frame: {len(ig_anch)} anchors ({n_ss} strand/helix); frame size {len(frames['Ig'])}")
    # each type keeps its own consensus coordinates under the shared names
    typed = {}
    for t in ("V", "C", "F"):
        if t not in cons:
            continue
        m = maps.get(t, {n: n for n in cons[t].names})
        c = Consensus([m[n] for n in cons[t].names], cons[t].xyz, cons[t].cbu)
        typed[t] = anchors_only(c, {m[n] for n in anch[t]})
        typed[t].stats = {m[n]: v for n, v in cons[t].stats.items()}
    cons["Ig"] = ig
    scoped = {"Ig": ig_anch}
    for t in ("G", "S", "A"):
        if t in cons:
            frames[t] = make_frame(cons[t].names)
            scoped[t] = {n: {"ss": True, "types": {t}} for n in anch[t]}
            cons[t] = anchors_only(cons[t], set(anch[t]))
    for t, c in list(cons.items()) + [(f"Ig_{k}", v) for k, v in typed.items()]:
        j = c.to_json(); j["frame"] = frames.get(t.split("_")[0])
        if t.split("_")[0] in scoped:
            fam_t = t.split("_")[0]
            j["anchors"] = {n: {"ss": v["ss"], "types": sorted(v["types"])} for n, v in scoped[fam_t].items()}
        json.dump(j, open(os.path.join(OUT, f"consensus_{t}.json"), "w"))
    json.dump({t: m for t, m in maps.items()}, open(os.path.join(OUT, "crossmap.json"), "w"), indent=1)
    # --- renumber
    final = {}
    for t, ds in (("Ig", ig_doms), ("G", fam["G"]), ("S", fam["S"]), ("A", fam["A"])):
        if t not in cons or not ds:
            continue
        tv = typed if t == "Ig" else {t: cons[t]}
        own = {k: set(maps.get(k, {n: n for n in cons[k].names}).values()) for k in ("V", "C", "F") if k in cons} if t == "Ig" else None
        r = renumber(ds, tv, frames[t], scoped[t], own)
        for d in ds:
            d["new"], d["site"] = r[d["id"]]["new"], r[d["id"]]["site"]
            d["ss"], part = dssp_lite(d, partners=True)
            d["partner"] = part.tolist()
        final[t] = ds
    json.dump([dict(id=d["id"], pdb=d["pdb"], chain=d["chain"], domtype=d["domtype"], cls=d["cls"], gene=d["gene"],
                    species=d["species"], aa=d["seq"], auth=[r["auth"] for r in d["residues"]], imgt=d["lab"], new=d["new"],
                    site=d["site"], ss=d["ss"], partner=d["partner"]) for ds in final.values() for d in ds],
              open(os.path.join(OUT, "renumbered.json"), "w"))
    log("renumbered", sum(len(v) for v in final.values()), f"{time.time()-t0:.0f}s")
    # --- evaluation
    for t, ds in final.items():
        agg = evaluate(ds, k=20 if len(ds) > 100 else len(ds))
        log(f"== evaluation {t}: ALL {fmt(agg['ALL'])}")
        for key in sorted(k for k in agg if k != "ALL"):
            if agg[key][0] > 2000:
                log(f"   {key[0]:>16} x {key[1]:<16} {fmt(agg[key])}")
        json.dump({("|".join(k) if k != "ALL" else k): v.tolist() for k, v in agg.items()}, open(os.path.join(OUT, f"eval_{t}.json"), "w"))
    log(f"done {time.time()-t0:.0f}s")


def build_consensus_fixed(ds, cons, log, iters=3):
    """Refine site positions over a mixed set without creating or demoting sites."""
    xf = {}
    for it in range(iters):
        acc = defaultdict(list); accb = defaultdict(list)
        for d in ds:
            m, R, t = assign(d, cons, init=xf.get(d["id"]))
            xf[d["id"]] = (R, t)
            Xs = d["ca"] @ R.T + t; Us = d["cbu"] @ R.T
            for i, j, _ in m:
                acc[cons.names[j]].append(Xs[i])
                if not np.isnan(Us[i]).any(): accb[cons.names[j]].append(Us[i])
        xyz = [np.mean(acc[n], 0) if acc.get(n) else cons.xyz[k] for k, n in enumerate(cons.names)]
        cbu = []
        for k, n in enumerate(cons.names):
            v = np.mean(accb[n], 0) if accb.get(n) else cons.cbu[k]
            cbu.append(v / (np.linalg.norm(v) + 1e-9))
        stats = {n: dict(occ=len(acc.get(n, [])) / len(ds)) for n in cons.names}
        cons = Consensus(cons.names, xyz, cbu); cons.stats = stats
        log(f"  refine iter {it}")
    return cons, xf


if __name__ == "__main__":
    main()
