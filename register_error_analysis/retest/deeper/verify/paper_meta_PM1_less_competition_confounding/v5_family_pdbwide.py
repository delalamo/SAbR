#!/usr/bin/env python3
"""PM1 confounding, part 4: the investigator's PDB-wide +/-180 d competitor count, but with the antigen defined at the
RCSB 30% and 70% sequence-identity cluster level instead of the (strain-specific) UniProt accession.
Antigen chains = SIFTS-mapped non-immunoglobulin, non-V chains of the paper's EM entries (as in competition.py);
chain -> entity from the antigen_competition GraphQL cache; entity -> cluster from RCSB clusters-by-entity files.
Competitors = distinct DOIs (other last author, other DOI) with ANY entity in the same cluster deposited within +/-180 d.
Writes tests_confounding_4.csv and family_competition.csv"""
import json, re, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
DEEP = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"
CACHE = f"{DEEP}/cache/verify_PM1"
BASE = "/home/user/SAbR/register_error_analysis/retest/deeper"
OUT = f"{BASE}/verify/paper_meta_PM1_less_competition_confounding"
T = pd.read_csv(f"{OUT}/paper_table_model2.csv", parse_dates=["dep_first"])

c = pd.read_csv(f"{DEEP}/chains.csv.gz", usecols=["pdb", "chain"])
vchains = set(zip(c.pdb.str.lower(), c.chain.astype(str)))
S = pd.read_csv("/tmp/supp/pdbwide/pdb_chain_uniprot.tsv.gz", sep="\t", comment="#", usecols=[0, 1, 2], dtype=str)
S.columns = ["pdb", "chain", "uniprot"]
S = S.drop_duplicates()
S["pdb"] = S.pdb.str.lower()
ig_acc = set(S[[(p, ch) in vchains for p, ch in zip(S.pdb, S.chain)]].uniprot)
S = S[~S.uniprot.isin(ig_acc)]
S = S[[(p, ch) not in vchains for p, ch in zip(S.pdb, S.chain)]]
ag_chains = S.groupby("pdb").chain.apply(set)

# chain -> entity for the focal entries
need = set(x for s in T.pdbs for x in s.split(";"))
chain2ent = {}
with open(f"{DEEP}/cache/antigen_competition/gql_entries.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        p = r["rcsb_id"].lower()
        if p not in need:
            continue
        for e in r.get("polymer_entities") or []:
            ids = e.get("rcsb_polymer_entity_container_identifiers") or {}
            for ch in ids.get("auth_asym_ids") or []:
                chain2ent[(p, ch)] = e["rcsb_id"].upper()
print("focal entries with entity map:", len({p for p, _ in chain2ent}), "of", len(need))

info, last = {}, {}
with open("/tmp/supp/pdbwide/cite_all.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        a = r.get("rcsb_accession_info") or {}
        info[r["rcsb_id"].lower()] = (((r.get("rcsb_primary_citation") or {}).get("pdbx_database_id_DOI") or "").lower() or None,
                                      pd.Timestamp(a["deposit_date"][:10]) if a.get("deposit_date") else pd.NaT)
with open("/tmp/supp/pdbwide/entries.idx", encoding="latin-1") as fh:
    for i, line in enumerate(fh):
        if i < 2:
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) > 5:
            au = [x.strip() for x in re.split(r",\s(?=[A-Z][^,]*\.)", f[5]) if x.strip()]
            last[f[0].lower()] = re.sub(r"[^a-z]", "", au[-1].split(",")[0].lower()) if au else ""

res = {}
for lev in [30, 70]:
    ent2cl, cl2pdb = {}, {}
    with open(f"{CACHE}/clusters-by-entity-{lev}.txt") as fh:
        for i, line in enumerate(fh):
            ents = line.split()
            cl2pdb[i] = sorted({e.split("_")[0].lower() for e in ents})
            for e in ents:
                ent2cl[e.upper()] = i
    out = []
    for _, p in T.iterrows():
        pdbs = p.pdbs.split(";")
        cls = set()
        for x in pdbs:
            for ch in ag_chains.get(x, set()):
                e = chain2ent.get((x, ch))
                if e and e in ent2cl:
                    cls.add(ent2cl[e])
        if not cls:
            out.append(dict(pkey=p.pkey, **{f"fam{lev}_n": np.nan, f"fam{lev}_ncl": 0}))
            continue
        cand = set().union(*[cl2pdb[k] for k in cls]) - set(pdbs)
        keys = set()
        for q in cand:
            doi, dep = info.get(q, (None, pd.NaT))
            if pd.isna(dep) or abs((dep - p.dep_first).days) > 180:
                continue
            if last.get(q, "") == p.plast or (doi is not None and doi == p.doi):
                continue
            keys.add(doi or "pdb:" + q)
        out.append(dict(pkey=p.pkey, **{f"fam{lev}_n": len(keys), f"fam{lev}_ncl": len(cls)}))
    res[lev] = pd.DataFrame(out)
for lev in res:
    T = T.merge(res[lev], on="pkey")
T[["pkey", "err", "comp_dep180", "fam30_n", "fam70_n", "fam30_ncl", "viral_org", "sars2", "cap", "enz"]].to_csv(f"{OUT}/family_competition.csv", index=False)


def fit(formula, d, var, label="", subset="all EM papers"):
    d = d.dropna(subset=[var])
    try:
        m = smf.glm(formula, d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["lab"])[0]})
        ci = np.exp(m.conf_int().loc[var])
        return dict(test=label, subset=subset, var=var, n=len(d), n_err=int(d.err.sum()), err_mean=d[d.err == 1][var].mean(),
                    ctrl_mean=d[d.err == 0][var].mean(), OR=np.exp(m.params[var]), lo=ci[0], hi=ci[1], p=m.pvalues[var])
    except Exception as ex:
        return dict(test=label, subset=subset, var=var, n=len(d), OR=np.nan, p=np.nan, note=str(ex)[:100])


B = "C(period) + l2_units + res_median"
PARS = B + " + cap + enz + fid_only + sars2"
R = []
for lev in [30, 70]:
    T[f"any_fam{lev}"] = np.where(T[f"fam{lev}_n"].notna(), (T[f"fam{lev}_n"] > 0).astype(float), np.nan)
    T[f"l_fam{lev}"] = np.log2(1 + T[f"fam{lev}_n"])
    for var in [f"any_fam{lev}", f"l_fam{lev}"]:
        R.append(fit(f"err ~ {var} + {B}", T, var, label=f"{lev}% cluster, base"))
        R.append(fit(f"err ~ {var} + {PARS}", T, var, label=f"{lev}% cluster, parsimonious"))
        for nm, d in [("viral non-SARS2", T[(T.viral_org == 1) & (T.sars2 == 0)]), ("non-viral", T[T.viral_org == 0]),
                      ("non-SARS2", T[T.sars2 == 0])]:
            R.append(fit(f"err ~ {var} + {B}", d, var, label=f"{lev}% cluster, base", subset=nm))
# accession-level, same complete-case sample for comparison
cc = T.dropna(subset=["any_fam30"])
R.append(fit(f"err ~ any_comp + {B}", cc, "any_comp", label="UniProt accession (investigator), same sample as 30% cluster"))
R.append(fit(f"err ~ any_comp + {B}", cc[(cc.viral_org == 1) & (cc.sars2 == 0)], "any_comp", label="UniProt accession, same sample", subset="viral non-SARS2"))
Rt = pd.DataFrame(R)
Rt.to_csv(f"{OUT}/tests_confounding_4.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 80, "display.max_rows", 200)
print(Rt.round(4).to_string())
print(T.groupby("err")[["comp_dep180", "fam30_n", "fam70_n"]].median())
print(pd.crosstab([T.viral_org, T.sars2], [T.err, T.any_fam30.fillna(-1)]))
