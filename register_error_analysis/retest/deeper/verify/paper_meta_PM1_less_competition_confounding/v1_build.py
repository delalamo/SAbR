#!/usr/bin/env python3
"""Verification (confounding lens) of paper_meta PM1 'less competition'.

Builds a paper-level table on top of paper_meta/competition_table.csv (read-only) with extra covariates:
  * finer time: deposit year, continuous deposit date;
  * complex size: number of antigen UniProt accessions, number of entries;
  * target class / fiducial status (from antigen_competition lens, per entry -> paper);
  * lab experience (prior antibody entries of the lab, antigen_competition lens);
  * new V regions (no identical V earlier in the PDB; pdb_meta lens);
  * local map fit of V chains (lq_z from chains.csv);
  * a re-computed competition measure using ONLY the antigen accessions the antibodies actually contact
    (tgt_keys), excluding partner/fiducial accessions (G proteins, BRIL, ACE2...);
  * all-time popularity of the antigen (other labs any time) to separate 'contemporaneous' from 'popular'.
Writes paper_table_verify.csv.
"""
import json, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
DEEP = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"
BASE = "/home/user/SAbR/register_error_analysis/retest/deeper"
OUT = f"{BASE}/verify/paper_meta_PM1_less_competition_confounding"

X = pd.read_csv(f"{BASE}/paper_meta/competition_table.csv", parse_dates=["dep_first", "pub_date"])
c = pd.read_csv(f"{DEEP}/chains.csv.gz")
F = pd.read_csv(f"{BASE}/antigen_competition/tables/focal_entries_antigen_features.csv.gz")
LE = pd.read_csv(f"{BASE}/antigen_competition/tables/lab_experience.csv")
NV = pd.read_csv(f"{BASE}/pdb_meta/tables/chain_novel_v.csv.gz")
F = F.merge(LE, on="pdb", how="left")
F["cls2"] = F.tgt_cls.replace({"viral other": "enzyme"})
fe = F.set_index("pdb")
c = c.merge(NV[["pdb", "chain", "novel_v"]], on=["pdb", "chain"], how="left")
cg = {p: g for p, g in c.groupby("pdb")}

# ---------- SIFTS / PDB-wide index (same construction as competition.py) ----------
vchains = set(zip(c.pdb.str.lower(), c.chain.astype(str)))
S = pd.read_csv("/tmp/supp/pdbwide/pdb_chain_uniprot.tsv.gz", sep="\t", comment="#", usecols=[0, 1, 2], dtype=str)
S.columns = ["pdb", "chain", "uniprot"]
S = S.drop_duplicates()
S["pdb"] = S.pdb.str.lower()
ig_acc = set(S[[(p, ch) in vchains for p, ch in zip(S.pdb, S.chain)]].uniprot)
S = S[~S.uniprot.isin(ig_acc)]
S = S[[(p, ch) not in vchains for p, ch in zip(S.pdb, S.chain)]]
ent_acc = S.groupby("pdb").uniprot.apply(set)
info = {}
with open("/tmp/supp/pdbwide/cite_all.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        a = r.get("rcsb_accession_info") or {}
        info[r["rcsb_id"].lower()] = (((r.get("rcsb_primary_citation") or {}).get("pdbx_database_id_DOI") or "").lower() or None,
                                      a.get("deposit_date", "")[:10], a.get("initial_release_date", "")[:10])
last = {}
with open("/tmp/supp/pdbwide/entries.idx", encoding="latin-1") as fh:
    for i, line in enumerate(fh):
        if i < 2:
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) > 5:
            au = [x.strip() for x in re.split(r",\s(?=[A-Z][^,]*\.)", f[5]) if x.strip()]
            last[f[0].lower()] = re.sub(r"[^a-z]", "", au[-1].split(",")[0].lower()) if au else ""
E = pd.DataFrame([(p, *info.get(p, (None, None, None)), last.get(p, "")) for p in ent_acc.index],
                 columns=["pdb", "doi", "dep", "rel", "last"])
E["dep"] = pd.to_datetime(E.dep, errors="coerce")
E["acc"] = E.pdb.map(ent_acc)
EX = E.explode("acc")
by_acc = {a: g for a, g in EX.groupby("acc")}
MAXDEP = E.dep.max()
print("latest deposit in SIFTS-mapped released PDB:", MAXDEP)
acc_n_entries = EX.groupby("acc").pdb.nunique()


def comp(accs, pdbs, plast, doi, dep0):
    """distinct other-lab DOIs with an entry carrying one of accs within +/-180 d; plus all-time other labs."""
    accs = [a for a in accs if a in by_acc]
    if not accs:
        return dict(n=np.nan, n_before=np.nan, n_after=np.nan, labs_alltime=np.nan, labs_prior=np.nan, labs_after=np.nan)
    G = pd.concat([by_acc[a] for a in accs]).drop_duplicates("pdb")
    G = G[~G.pdb.isin(pdbs)]
    o = G[(G["last"] != plast) & ((G.doi != doi) | G.doi.isna())]
    key = o.doi.fillna("pdb:" + o.pdb)
    dd = (o.dep - dep0).dt.days
    return dict(n=key[dd.abs() <= 180].nunique(), n_before=key[(dd < 0) & (dd >= -180)].nunique(),
                n_after=key[(dd >= 0) & (dd <= 180)].nunique(),
                labs_alltime=o["last"].nunique(), labs_prior=o[dd < -180]["last"].nunique(),
                labs_after=o[dd > 180]["last"].nunique())


rows = []
for _, p in X.iterrows():
    pdbs = p.pdbs.split(";")
    ff = fe.reindex([x for x in pdbs if x in fe.index])
    cc = pd.concat([cg[x] for x in pdbs if x in cg]) if any(x in cg for x in pdbs) else pd.DataFrame()
    r = dict(pkey=p.pkey)
    r["n_focal"] = len(ff)
    r["cls_mode"] = ff.cls2.mode().iat[0] if len(ff) and ff.cls2.notna().any() else "NA"
    r["cap"] = int((ff.cls2 == "viral capsid").any())
    r["enz"] = int((ff.cls2 == "enzyme").any())
    r["vsg"] = int((ff.cls2 == "viral surface glycoprotein").any())
    r["gpcr_chan"] = int(ff.cls2.isin(["GPCR", "channel/transporter"]).any())
    r["fid_only"] = int(len(ff) > 0 and (ff.ab_status == "fiducial-only").all())
    r["any_fid"] = int((ff.ab_status == "fiducial-only").any())
    r["unbound_only"] = int(len(ff) > 0 and (ff.ab_status == "unbound").all())
    r["viral_org"] = int((ff.tgt_org == "viral").any())
    r["org_mode"] = ff.tgt_org.mode().iat[0] if len(ff) and ff.tgt_org.notna().any() else "NA"
    r["lab_prior_ab"] = ff.lab_prior_ab.min() if len(ff) else np.nan
    # antibody-level
    if len(cc):
        u = cc.groupby("unit").agg(nv=("novel_v", "max"), vhh=("vhh", "max"), mouse=("species", lambda s: (s == "mouse").any()),
                                   vid=("v_id", "mean"), atyp=("L3_atyp", "max"))
        r["n_units_chk"] = len(u)
        r["n_novel_units"] = int(u.nv.fillna(False).astype(bool).sum())
        r["novel_frac"] = u.nv.fillna(False).astype(bool).mean()
        r["any_vhh"] = int((u.vhh == 1).any())
        r["any_mouse"] = int(u.mouse.any())
        r["vid_min"] = u.vid.min()
        # local map fit: all chains, and chains with no disagreement run at all
        r["lqz_all"] = cc.lq_z.median()
        r["lqz_clean_chains"] = cc[~cc.err_any.astype(bool)].lq_z.median()
        r["res_min_v"] = cc.res.min()
    # contact-antigen (primary target) competition
    keys = set()
    for k in ff.tgt_keys.dropna():
        keys |= {x for x in str(k).split("|") if not x.startswith("g")}
    r["n_tgt_acc"] = len(keys)
    pr = comp(keys, pdbs, p.plast, p.doi, p.dep_first)
    r.update({f"tgt_{k}": v for k, v in pr.items()})
    # all-accession competition, recomputed with split before/after and all-time popularity
    accs = set().union(*[ent_acc.get(x, set()) for x in pdbs])
    pa = comp(accs, pdbs, p.plast, p.doi, p.dep_first)
    r.update({f"all_{k}": v for k, v in pa.items()})
    r["acc_max_entries"] = max([acc_n_entries.get(a, 0) for a in accs]) if accs else np.nan
    r["days_to_maxdep"] = (MAXDEP - p.dep_first).days
    rows.append(r)

V = pd.DataFrame(rows)
T = X.merge(V, on="pkey", how="left")
T.to_csv(f"{OUT}/paper_table_verify.csv", index=False)
print(T.shape)
print("check recomputed all_n == comp_dep180:", (T.all_n.fillna(-1) == T.comp_dep180.fillna(-1)).mean())
print(T.groupby("err")[["n_focal", "cap", "enz", "fid_only", "unbound_only", "novel_frac", "lab_prior_ab", "n_tgt_acc", "tgt_n", "all_n", "all_labs_alltime"]].mean().T)
