#!/usr/bin/env python3
"""Independent check that the 15 AL-amyloid entries are not Ig-folded and that their SAbR/ANARCI
disagreement is a detector artifact rather than a register error.

For every chain with a cached mmCIF (scratchpad/sabr/dl) and a per-residue SAbR/ANARCI record:
  * Q_native: fraction of consensus Ig-V Calpha contacts (built from clean, Ig-folded control light chains,
    positions indexed by sequence-based ANARCI IMGT numbers) that are present in the chain (< 8 A + 1.5 A slack);
  * flat: SD of Calpha coords along the 3rd principal axis (fibril layers are ~planar);
  * frac_dis: fraction of residues where SAbR IMGT != ANARCI IMGT;
  * frac_oor: fraction of residues SAbR places outside IMGT 1-128.
Groups: amyloid (15 entries), non-amyloid error chains (err_first), clean light chains (controls).
"""
import json, gzip, random, sys, warnings
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser

warnings.filterwarnings("ignore")
SP = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM0_amyloid_artifact_integrity"
AMY = ['6ic3', '6z1o', '7nsl', '9faa', '9fab', '9fac', '9els', '9liv', '9liw', '9lix', '9liy', '9lj0', '9lja', '9ljb', '9v91']

c = pd.read_csv(f"{SP}/deeper/chains.csv.gz")
import os
cached = {f[:-4] for f in os.listdir(f"{SP}/sabr/dl") if f.endswith(".cif")}

rec = {}
for fn in ["validation.jsonl", "snapshot_rest.jsonl", "new.jsonl"]:
    with open(f"{SP}/sabr/out/{fn}") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("status") != "ok":
                continue
            rec[(d["pdb"], d["chain"])] = d

c["amy"] = c.pdb.isin(AMY)
c["has"] = c.pdb.isin(cached).values & np.array([(p, ch) in rec for p, ch in zip(c.pdb, c.chain)])
light = c.ctype.isin(["L", "K"])
amy = c[c.amy & c.has]
err = c[(~c.amy) & c.err_first & c.has & (c.method == "EM")]
ctl = c[(~c.amy) & (~c.err_any) & c.has & light & (c.method == "EM")]
random.seed(1)
# one chain per entry for controls/errors to limit compute
ctl = ctl.groupby("pdb").head(1)
ctl = ctl.sample(min(150, len(ctl)), random_state=1)
err = err.groupby("pdb").head(1)
print("chains: amyloid", len(amy), "error(non-amy, EM, 1/entry)", len(err), "clean light controls", len(ctl), file=sys.stderr)

parser = MMCIFParser(QUIET=True)
_cache = {}


def get_struct(pdb):
    if pdb not in _cache:
        _cache.clear()
        _cache[pdb] = parser.get_structure(pdb, f"{SP}/sabr/dl/{pdb}.cif")
    return _cache[pdb]


def coords(pdb, chain):
    s = get_struct(pdb)
    m = next(iter(s))
    if chain not in m:
        return None
    ch = m[chain]
    d = rec[(pdb, chain)]
    out = []
    for r in d["residues"]:
        an, ic, rn, si, a_imgt, a_ins, s_imgt, s_ins = r
        key = (" ", an, ic if ic else " ")
        res = ch[key] if key in ch else None
        if res is None or "CA" not in res:
            continue
        out.append(dict(a=(a_imgt, a_ins or ""), s=(s_imgt, s_ins or ""), xyz=res["CA"].coord.astype(float)))
    return out


def feats(rows):
    X = np.array([r["xyz"] for r in rows])
    Xc = X - X.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False) / np.sqrt(len(X))
    dis = np.mean([r["a"] != r["s"] for r in rows])
    oor = np.mean([(r["s"][0] is None) or (r["s"][0] < 1) or (r["s"][0] > 128) for r in rows])
    return dict(sd1=sv[0], sd2=sv[1], flat=sv[2], frac_dis=dis, frac_oor=oor, n_ca=len(rows))


def cmap(rows):
    """dict of (imgt_i, imgt_j) -> distance using ANARCI (sequence-based) positions, no insertions"""
    pos = {r["a"][0]: r["xyz"] for r in rows if r["a"][0] is not None and not r["a"][1]}
    ks = sorted(pos)
    D = {}
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if b - a >= 4:
                D[(a, b)] = float(np.linalg.norm(pos[a] - pos[b]))
    return D


rows = []
maps = {}
for grp, df in [("amyloid", amy), ("error_nonamy", err), ("clean_light", ctl)]:
    for p, ch in zip(df.pdb, df.chain):
        try:
            r = coords(p, ch)
        except Exception as ex:
            print("fail", p, ch, ex, file=sys.stderr)
            continue
        if not r or len(r) < 40:
            continue
        f = feats(r)
        f.update(pdb=p, chain=ch, group=grp)
        rows.append(f)
        maps[(p, ch)] = cmap(r)

F = pd.DataFrame(rows)
# consensus contacts from clean controls: pairs in contact (<8 A) in >=80% of controls where both present
ctl_keys = [(p, ch) for p, ch, g in zip(F.pdb, F.chain, F.group) if g == "clean_light"]
cnt, tot = {}, {}
for k in ctl_keys:
    for pr, dd in maps[k].items():
        tot[pr] = tot.get(pr, 0) + 1
        if dd < 8:
            cnt[pr] = cnt.get(pr, 0) + 1
cons = {pr for pr, n in tot.items() if n >= 0.5 * len(ctl_keys) and cnt.get(pr, 0) / n >= 0.8}
print("consensus contacts:", len(cons), file=sys.stderr)


def q(k):
    m = maps[k]
    have = [pr for pr in cons if pr in m]
    if len(have) < 30:
        return np.nan
    return float(np.mean([m[pr] < 9.5 for pr in have]))


F["Q_native"] = [q((p, ch)) for p, ch in zip(F.pdb, F.chain)]
F.to_csv(f"{OUT}/fold_check_chains.csv", index=False)
pd.set_option("display.width", 200)
summ = F.groupby("group")[["Q_native", "flat", "sd1", "frac_dis", "frac_oor", "n_ca"]].describe(percentiles=[0.05, 0.5, 0.95]).T
print(summ.to_string())
print()
print(F[F.group == "amyloid"].groupby("pdb")[["Q_native", "flat", "frac_dis", "frac_oor"]].mean().round(3).to_string())
# how many non-amyloid error / clean chains look like the amyloid ones?
thrQ = F[F.group == "amyloid"].Q_native.max()
print("\nmax amyloid Q:", round(thrQ, 3))
for g in ["error_nonamy", "clean_light"]:
    s = F[F.group == g]
    print(g, "n", len(s), "Q<=max amyloid Q:", int((s.Q_native <= thrQ).sum()), " min Q:", round(s.Q_native.min(), 3))
low = F[(F.group != "amyloid") & (F.Q_native < 0.6)]
print("non-amyloid chains with Q<0.6:\n", low.to_string())
