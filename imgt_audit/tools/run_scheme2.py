"""Hierarchical structure-consistent numbering.

  class consensus (sharp, named by the class's own IMGT labels)
    -> merged into its family frame (V, C, FN3, G, S, A) by order-preserving 3 A anchor matching
    -> V, C and FN3 family frames merged into one Ig frame on the shared B/C/E/F core
Each domain is assigned against its own class model; names come from the merged frame.
"""
import json, os, re, sys, time
from collections import Counter, defaultdict
import numpy as np
import tmtools
sys.path.insert(0, os.path.dirname(__file__))
from scheme import (_decimals, is_int, lab_key, load_big, build_consensus, imgt_order_key, canonicalize,
                    assign, number_domain, Consensus, kabsch, dp_assign, dssp_lite)
import run_scheme as rs

OUT = os.path.join(os.path.dirname(__file__), "..", "big", "scheme2")
os.makedirs(OUT, exist_ok=True)
IG_CORE = rs.IG_CORE
G_CORE = [str(i) for i in list(range(3, 14)) + list(range(21, 37))]
HV = {"V": [(27, 38), (56, 65), (105, 117)]}
MIN_CLASS = 8
REF = {"V": "V-KAPPA", "C": "C-KAPPA", "F": "F:FN/TNC", "G": "G-ALPHA1", "S": "S-DOMAIN", "A": "A-DOMAIN"}
BAD_SEED_GENES = {("IGHG1*03", "Mus musculus")}


def seed_of(ds):
    ok = [d for d in ds if (d["gene"], d["species"]) not in BAD_SEED_GENES]
    return max(ok or ds, key=lambda d: (d["ident"] >= 95, len(d["residues"]), -(d["resolution"] or 9)))


def merge(uni, uni_anc, other, other_anc, core, log, tag, ref=None):
    """Name `other`'s positions in `uni`'s frame. Anchors within 3 A (order-preserving DP) share
    a name. Unmatched positions keep their own integer name when it is free in the frame and fits
    the order; otherwise they get N-attached decimals on the preceding frame name."""
    shared = [n for n in core if n in uni.names and n in other.names]
    if ref is not None:
        # sequence-independent superposition on the family's reference class model
        r = tmtools.tm_align(other.xyz, ref.xyz, "A" * len(other.xyz), "A" * len(ref.xyz))
        R, t = r.u, r.t
    else:
        R, t = kabsch(np.array([other.xyz[other.names.index(n)] for n in shared]),
                      np.array([uni.xyz[uni.names.index(n)] for n in shared]))
    oa = np.array([n in other_anc for n in other.names]); ua = np.array([n in uni_anc for n in uni.names])
    for _ in range(3):
        X = other.xyz @ R.T + t; U = other.cbu @ R.T
        D = np.linalg.norm(X[:, None] - uni.xyz[None], axis=2)
        S = np.exp(-D ** 2 / (2 * 1.5 ** 2)) * np.clip((1 + U @ uni.cbu.T) / 2, 0, 1)
        S[D > 3.0] = 0
        S[~oa, :] = 0
        S[:, ~ua] = 0
        pairs = dict(dp_assign(S))
        if ref is None or len(pairs) < 10:
            break
        ii = sorted(pairs)
        R, t = kabsch(other.xyz[ii], uni.xyz[[pairs[i] for i in ii]])
    shared = [other.names[i] for i in pairs] if ref is not None else shared
    taken = set(uni.names)
    newname = {}
    for i in pairs:
        newname[i] = uni.names[pairs[i]]
    matched = sorted(pairs)
    # unmatched runs
    i = 0
    n = len(other.names)
    while i < n:
        if i in newname:
            i += 1; continue
        j = i
        while j < n and j not in newname:
            j += 1
        run = list(range(i, j))
        lo = newname[i - 1] if i > 0 else None
        hi = newname[j] if j < n else None
        own = [other.names[k] for k in run]
        fits = all(is_int(x) and x not in taken for x in own) and \
            (lo is None or lab_key(own[0]) > lab_key(lo)) and (hi is None or lab_key(own[-1]) < lab_key(hi)) and \
            all(lab_key(a) < lab_key(b) for a, b in zip(own, own[1:]))
        if fits:
            names = own
        elif lo is None:
            first = lab_key(hi)[0]
            names = [str(k) for k in range(first - 1, first - 500, -1) if str(k) not in taken][:len(run)][::-1]
        elif hi is None:
            last = lab_key(lo)[0]
            names = [str(k) for k in range(last + 1, last + 500) if str(k) not in taken][:len(run)]
        else:
            names = _decimals(lo, taken, len(run))
        for k, nm in zip(run, names):
            newname[k] = nm; taken.add(nm)
        i = j
    mapping = {other.names[k]: newname[k] for k in range(n)}
    names = list(uni.names); xyz = list(uni.xyz); cbu = list(uni.cbu)
    for k in range(n):
        if k not in pairs:
            names.append(newname[k]); xyz.append(X[k]); cbu.append(U[k])
    anc = set(uni_anc) | {mapping[x] for x in other_anc}
    if ref is not None:
        rms = np.sqrt(np.mean([np.sum((X[i] - uni.xyz[j]) ** 2) for i, j in pairs.items()]))
    else:
        rms = np.sqrt(np.mean(np.sum((X[[other.names.index(x) for x in shared]] - uni.xyz[[uni.names.index(x) for x in shared]]) ** 2, 1)))
    order = [lab_key(newname[k]) for k in range(n)]
    viol = sum(1 for a, b in zip(order, order[1:]) if b <= a)
    changed = [(k, v) for k, v in mapping.items() if k != v]
    log(f"  merge {tag}: core {rms:.2f} A/{len(shared)}; {len(pairs)} shared anchors; {len(changed)} renamed; order violations {viol}")
    if changed:
        log(f"    {changed}")
    return Consensus(names, xyz, cbu), anc, mapping, (R, t)


def class_model(ds, fam, log, tag):
    key = imgt_order_key(ds)
    full, _ = build_consensus(ds, seed_of(ds), spread_max=1.6, log=lambda *a: None, key=key)
    anc = rs.anchor_filter(full, ds, lambda *a: None, hv=HV.get(fam, ()))
    full_c, ren = canonicalize(full)
    anchors = {ren[x] for x in anc.names}
    for x in full_c.names:
        full_c.stats[x] = {**full_c.stats.get(x, {}), **anc.stats.get(full_c.stats.get(x, {}).get("imgt_name", x), {})}
    log(f"  class {tag}: {len(ds)} domains, {len(full_c.names)} sites, {len(anchors)} anchors")
    return full_c, anchors


def family_frame(fam, classes, log, core):
    """Merge class models into one family frame; reference class first, then by size."""
    order = sorted(classes, key=lambda c: (c != REF[fam], -classes[c][2]))
    uni, uni_anc, ref = None, set(), None
    maps = {}
    for c in order:
        model, anc, n = classes[c]
        if uni is None:
            uni, uni_anc, ref = model, set(anc), model
            maps[c] = {x: x for x in model.names}
            continue
        uni, uni_anc, maps[c], _ = merge(uni, uni_anc, model, anc, core, log, f"{c} -> {fam}", ref=ref)
    return uni, uni_anc, maps


def _letters(k):
    out = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        out = chr(65 + r) + out
    return out


def letterize(seqs, log, tag, coords=None):
    """Global order of all anchors from every class's chain order (topological sort); integer
    anchors keep their number, every other anchor becomes a letter sub-site of the last integer
    before it (8A, 8B, ...). Class-specific runs are placed straight after their preceding anchor."""
    import heapq
    succ, pred, nodes = defaultdict(set), defaultdict(set), set()
    for seq in seqs:
        nodes |= set(seq)
        for a, b in zip(seq, seq[1:]):
            if a != b and a not in succ[b]:
                succ[a].add(b); pred[b].add(a)
    # class-specific anchors are placed right after the spatially nearest shared integer anchor
    ints = [x for x in nodes if is_int(x)]
    near = {}
    if coords is not None:
        P = np.array([coords[x] for x in ints])
        for x in nodes:
            if not is_int(x) and x in coords:
                near[x] = ints[int(np.argmin(np.linalg.norm(P - coords[x], axis=1)))]
    def prio(x):
        if is_int(x):
            return (lab_key(x), 0)
        return (lab_key(near.get(x, x)), 1)
    indeg = {x: len(pred[x]) for x in nodes}
    heap = [(prio(x), x) for x in nodes if indeg[x] == 0]
    heapq.heapify(heap)
    order, done, cycles = [], set(), 0
    while len(order) < len(nodes):
        if not heap:
            x = min((y for y in nodes if y not in done), key=prio); cycles += 1
        else:
            _, x = heapq.heappop(heap)
            if x in done:
                continue
        order.append(x); done.add(x)
        for y in succ[x]:
            indeg[y] -= 1
            if indeg[y] <= 0 and y not in done:
                heapq.heappush(heap, (prio(y), y))
    ren, last, count, bad = {}, None, Counter(), 0
    first_int = next((int(x) for x in order if is_int(x)), 1)
    for x in order:
        if is_int(x):
            if last is not None and int(x) <= int(last):
                bad += 1
            last = x; ren[x] = x
        else:
            base = last if last is not None else str(first_int - 1)
            ren[x] = f"{base}{_letters(count[base])}"; count[base] += 1
    log(f"  letterize {tag}: {len(nodes)} anchors, {sum(1 for x in nodes if not is_int(x))} letter sub-sites, "
        f"cycles broken {cycles}, integer order conflicts {bad}")
    return ren


def main():
    t_start = time.time()
    logf = open(os.path.join(OUT, "log.txt"), "w")
    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n")
    doms = load_big()
    rs.letters_to_decimal(doms)
    for d in doms:
        if d["cls"] == "V-IOTA": d["cls"] = "V-LIKE"
    fam = defaultdict(list)
    for d in doms:
        fam[d["domtype"]].append(d)
    log("domains", len(doms), dict(Counter(d["domtype"] for d in doms)))
    models, fam_frames, seqs = {}, {}, {}
    for f in ("V", "C", "F", "G", "S", "A"):
        bycls = defaultdict(list)
        for d in fam[f]:
            bycls[d["cls"]].append(d)
        log(f"== family {f}: {dict(Counter({c: len(v) for c, v in bycls.items()}))}")
        classes = {}
        for c, ds in bycls.items():
            if len(ds) >= MIN_CLASS or c == REF[f]:
                m, anc = class_model(ds, f, log, c)
                m = rs.anchors_only(m, anc)
                classes[c] = (m, anc, len(ds))
        core = G_CORE if f == "G" else IG_CORE
        uni, uni_anc, maps = family_frame(f, classes, log, core)
        for c, (m, anc, n) in classes.items():
            mp = maps[c]
            seqs[(f, c)] = [mp[x] for x in m.names]
            mm = Consensus([mp[x] for x in m.names], m.xyz, m.cbu)
            mm.stats = {mp[x]: v for x, v in m.stats.items()}
            models[(f, c)] = (mm, {mp[x] for x in anc})
        fam_frames[f] = (uni, uni_anc, set(bycls) - set(classes))
    # --- Ig frame: express C and FN3 family frames in V's names
    ig, ig_anc, _ = fam_frames["V"]
    tmaps = {"V": {x: x for x in ig.names}}
    for f in ("C", "F"):
        u, a, _ = fam_frames[f]
        ig, ig_anc, tmaps[f], _ = merge(ig, ig_anc, u, a, IG_CORE, log, f"{f} -> Ig")
    for (f, c), (m, anc) in list(models.items()):
        if f in ("C", "F"):
            mp = tmaps[f]
            mm = Consensus([mp[x] for x in m.names], m.xyz, m.cbu)
            mm.stats = {mp[x]: v for x, v in m.stats.items()}
            models[(f, c)] = (mm, {mp[x] for x in anc})
            seqs[(f, c)] = [mp[x] for x in seqs[(f, c)]]
    # --- letter sub-sites: one global order per frame
    ren = {}
    for grp, fs in (("Ig", ("V", "C", "F")), ("G", ("G",)), ("S", ("S",)), ("A", ("A",))):
        u = ig if grp == "Ig" else fam_frames[fs[0]][0]
        r = letterize([v for (f, c), v in seqs.items() if f in fs], log, grp, coords=dict(zip(u.names, u.xyz)))
        for f in fs:
            ren[f] = r
    for (f, c), (m, anc) in list(models.items()):
        r = ren[f]
        mm = Consensus([r[x] for x in m.names], m.xyz, m.cbu)
        mm.stats = {r[x]: v for x, v in m.stats.items()}
        models[(f, c)] = (mm, {r[x] for x in anc})
    def ren_uni(f, uni, anc):
        r = ren[f]
        keep = [k for k, x in enumerate(uni.names) if x in r]
        return Consensus([r[uni.names[k]] for k in keep], uni.xyz[keep], uni.cbu[keep]), {r[x] for x in anc if x in r}
    ig, ig_anc = ren_uni("V", ig, ig_anc)
    frame_ig = rs.make_frame(ig.names)
    # names owned by each class (its own decimals are offered to it as loop names)
    own = {k: set(m.names) for k, (m, a) in models.items()}
    log(f"== Ig frame: {len(ig_anc)} anchors, frame size {len(frame_ig)}")
    # --- renumber every domain against its class model (small classes: family frame model)
    out = []
    final = defaultdict(list)
    for f in ("V", "C", "F", "G", "S", "A"):
        uni, uni_anc, small = fam_frames[f]
        if f in ("C", "F"):
            uni = Consensus([tmaps[f][x] for x in uni.names], uni.xyz, uni.cbu)
            uni_anc = {tmaps[f][x] for x in uni_anc}
        uni, uni_anc = ren_uni(f, uni, uni_anc)
        frame = frame_ig if f in ("V", "C", "F") else rs.make_frame(uni.names)
        all_anc = ig_anc if f in ("V", "C", "F") else uni_anc
        for d in fam[f]:
            key = (f, d["cls"])
            if key in models:
                m, anc = models[key]; mine = own[key]
            else:
                m, anc = uni, uni_anc; mine = set(uni.names)
            view = rs.anchors_only(m, anc)
            frame_c = sorted({x for x in frame if is_int(x)} | set(anc), key=lab_key)
            other = [x for x in all_anc if x not in anc and not is_int(x)]
            mt, R, t = assign(d, view, dmax=3.0)
            site_of = {i: view.names[j] for i, j, _ in mt}
            lab = number_domain(len(d["residues"]), site_of, view.names, frame_c, exclude=other)
            d["new"] = [lab[i] for i in range(len(d["residues"]))]
            d["site"] = [i in site_of for i in range(len(d["residues"]))]
            d["ss"], part = dssp_lite(d, partners=True)
            d["partner"] = part.tolist()
            final["Ig" if f in "VCF" else f].append(d)
    ooo = sum(any(lab_key(b) <= lab_key(a) for a, b in zip(d["new"], d["new"][1:])) for ds in final.values() for d in ds)
    log(f"domains with labels out of chain order: {ooo}")
    json.dump([dict(id=d["id"], pdb=d["pdb"], chain=d["chain"], domtype=d["domtype"], cls=d["cls"], gene=d["gene"],
                    species=d["species"], ident=d["ident"], aa=d["seq"], auth=[r["auth"] for r in d["residues"]],
                    imgt=d["lab"], new=d["new"], site=d["site"], ss=d["ss"], partner=d["partner"])
               for ds in final.values() for d in ds], open(os.path.join(OUT, "renumbered.json"), "w"))
    for (f, c), (m, anc) in models.items():
        j = m.to_json(); j["anchors"] = sorted(anc, key=lab_key)
        json.dump(j, open(os.path.join(OUT, f"model_{f}_{re.sub(r'[^A-Za-z0-9-]', '_', c)}.json"), "w"))
    json.dump(dict(frame_Ig=frame_ig, anchors_Ig=sorted(ig_anc, key=lab_key), type_maps=tmaps,
                   letter_map={f: r for f, r in ren.items()},
                   anchors={f: sorted({x for (ff, c), (m, a) in models.items() if ff == f for x in a}, key=lab_key) for f in ("G", "S", "A")}),
              open(os.path.join(OUT, "frames.json"), "w"), indent=0)
    log(f"renumbered {sum(len(v) for v in final.values())} domains in {time.time() - t_start:.0f}s")
    for f, ds in final.items():
        agg = rs.evaluate(ds, k=20 if len(ds) > 100 else len(ds))
        log(f"== evaluation {f}: ALL {rs.fmt(agg['ALL'])}")
        for key in sorted(k for k in agg if k != "ALL"):
            if agg[key][0] > 2000:
                log(f"   {key[0]:>16} x {key[1]:<16} {rs.fmt(agg[key])}")
        json.dump({("|".join(k) if k != "ALL" else k): v.tolist() for k, v in agg.items()}, open(os.path.join(OUT, f"eval_{f}.json"), "w"))
    log(f"done {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
