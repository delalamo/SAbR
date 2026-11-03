#!/usr/bin/env python3
"""Detector-independent fold test. For each V chain, take CA atoms of residues that ANARCI (sequence-based) numbers,
and compute shape descriptors that do not use SAbR at all:
  thick  = SD along the 3rd principal axis of the CA cloud (an Ig V domain is a ~25x25x40 A sandwich; a fibril
           layer is a flat ~4.8 A-thick sheet);
  rg     = radius of gyration;
  dAG    = distance between CA centroids of IMGT 1-15 (strand A) and IMGT 118-128 (strand G): adjacent in the
           Ig fold, far apart in an extended fibril layer.
Groups: AL fibril chains (15 entries in the set + 6HUD/8CPE/6Z1I outside it), clean EM chains (sample), and
non-amyloid EM first-occurrence error chains."""
import json, os, sys, numpy as np, pandas as pd
B = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM0_amyloid_artifact_confounding"
CACHE = f"{B}/deeper/cache/verify_PM0"
AMY = ['6ic3','6z1o','7nsl','9els','9faa','9fab','9fac','9liv','9liw','9lix','9liy','9lj0','9lja','9ljb','9v91']
EXTRA = ['6hud', '8cpe', '6z1i']
C = pd.read_csv(f"{B}/deeper/chains.csv.gz")
have = {f.split('.')[0].lower() for f in os.listdir(f"{B}/sabr/dl")}
rng = np.random.default_rng(11)
em = C[C.method == "EM"].copy()
em["grp"] = np.where(em.pdb.isin(AMY), "AL_fibril", np.where(em.err_first.astype(bool), "error_nonamy",
                     np.where(~em.err_any.astype(bool), "clean", "other")))
sel = [em[em.grp == "AL_fibril"]]
ne = em[(em.grp == "error_nonamy") & em.pdb.isin(have)]
sel.append(ne.groupby("unit").head(1))
for ct, n in [("L", 60), ("K", 60), ("H", 60)]:
    cl = em[(em.grp == "clean") & (em.ctype == ct) & em.pdb.isin(have)]
    ents = rng.choice(cl.pdb.unique(), size=min(n, cl.pdb.nunique()), replace=False)
    sel.append(cl[cl.pdb.isin(ents)].groupby("pdb").head(1))
S = pd.concat(sel)[["pdb", "chain", "ctype", "grp", "year", "res", "err_modes"]]
S = pd.concat([S, pd.DataFrame([dict(pdb=p, chain=None, ctype="L", grp="AL_fibril_outside_set") for p in EXTRA])])
need = set(S.pdb)
print("chains selected:", S.grp.value_counts().to_dict())

# ANARCI numbering per (pdb, chain) from SAbR output records (field 4 = anarci_imgt; author num field 0, icode 1)
rec = {}
for f in ["new", "snapshot_rest", "validation"]:
    for l in open(f"{B}/sabr/out/{f}.jsonl"):
        if l[9:13].lower() not in need and json.loads(l)["pdb"] not in need:
            continue
        d = json.loads(l)
        if d.get("residues"):
            rec[(d["pdb"], d["chain"])] = d["residues"]

def read_ca(path):
    cols, rows, inloop = [], [], False
    with open(path) as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                cols.append(line.strip().split(".", 1)[1]); inloop = True; continue
            if inloop and (line.startswith("ATOM") or line.startswith("HETATM")):
                rows.append(line.split())
            elif inloop and cols and line.startswith("#"):
                break
    ix = {c: i for i, c in enumerate(cols)}
    out = {}
    for r in rows:
        if r[ix["label_atom_id"]] != "CA" or r[ix["pdbx_PDB_model_num"]] != "1":
            continue
        if r[ix.get("label_alt_id", 0)] not in (".", "A", "?"):
            continue
        ch = r[ix["auth_asym_id"]]; num = int(r[ix["auth_seq_id"]])
        ic = r[ix["pdbx_PDB_ins_code"]]; ic = "" if ic in ("?", ".") else ic
        out.setdefault(ch, {})[(num, ic)] = np.array([float(r[ix["Cartn_x"]]), float(r[ix["Cartn_y"]]), float(r[ix["Cartn_z"]])])
    return out

res = []
for pdb, g in S.groupby("pdb"):
    path = f"{B}/sabr/dl/{pdb}.cif" if os.path.exists(f"{B}/sabr/dl/{pdb}.cif") else f"{CACHE}/{pdb}.cif"
    if not os.path.exists(path):
        continue
    ca = read_ca(path)
    for _, row in g.iterrows():
        chains = [row.chain] if isinstance(row.chain, str) else [k[1] for k in rec if k[0] == pdb][:1]
        if not chains and pdb == "6z1i":
            chains = sorted(ca)[:1]
        for ch in chains:
            r = rec.get((pdb, ch))
            cad = ca.get(ch, {})
            if r is not None:
                pts = [(x[4], cad.get((x[0], x[1]))) for x in r if x[4] is not None]
            else:  # 6z1i: not scanned; use the first 120 modelled residues, no IMGT numbers
                pts = [(None, v) for k, v in sorted(cad.items())][:120]
            pts = [(n, v) for n, v in pts if v is not None]
            if len(pts) < 30:
                continue
            X = np.array([v for _, v in pts]); X0 = X - X.mean(0)
            sv = np.sqrt(np.linalg.eigvalsh(np.cov(X0.T)))[::-1]
            rg = float(np.sqrt((X0 ** 2).sum(1).mean()))
            A = [v for n, v in pts if n is not None and 1 <= n <= 15]
            G = [v for n, v in pts if n is not None and 118 <= n <= 128]
            dAG = float(np.linalg.norm(np.mean(A, 0) - np.mean(G, 0))) if len(A) >= 3 and len(G) >= 3 else np.nan
            res.append(dict(pdb=pdb, chain=ch, grp=row.grp, ctype=row.ctype, year=row.year, res=row.res, n_ca=len(X),
                            s1=sv[0], s2=sv[1], thick=sv[2], rg=rg, dAG=dAG))
T = pd.DataFrame(res)
T.to_csv(f"{OUT}/fold_geometry_chains.csv", index=False)
pd.set_option("display.width", 200)
print(T.groupby("grp")[["n_ca", "thick", "rg", "dAG"]].describe(percentiles=[.0, .5, 1.0]).round(1).T.to_string())
print(T[T.grp.str.startswith("AL")].round(1).to_string())
lo_thick = T[~T.grp.str.startswith("AL")].thick.min()
print("min thickness among non-AL chains:", round(lo_thick, 2), " max thickness among AL:", round(T[T.grp.str.startswith('AL')].thick.max(), 2))
print("non-AL chains with thick<5 or dAG>25:", T[(~T.grp.str.startswith('AL')) & ((T.thick < 5) | (T.dAG > 25))][["pdb", "chain", "grp", "thick", "rg", "dAG"]].to_string())
