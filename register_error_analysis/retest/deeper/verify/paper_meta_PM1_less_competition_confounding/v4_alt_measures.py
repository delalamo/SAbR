#!/usr/bin/env python3
"""PM1 confounding, part 3: alternative measures and other populations.
 a) family-level (30% identity) competition from the antigen_competition lens (antibody structures only),
    which removes the strain-specific-UniProt undercount for viruses;
 b) complete-window papers only (deposited before 2025-01-01, so +180 d window is fully released);
 c) X-ray papers (same measure recomputed);
 d) copied/artifact-only EM papers counted as controls.
Writes tests_confounding_3.csv"""
import json, re, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
DEEP = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"
BASE = "/home/user/SAbR/register_error_analysis/retest/deeper"
OUT = f"{BASE}/verify/paper_meta_PM1_less_competition_confounding"
T = pd.read_csv(f"{OUT}/paper_table_model2.csv", parse_dates=["dep_first"])
F = pd.read_csv(f"{BASE}/antigen_competition/tables/focal_entries_antigen_features.csv.gz")
fe = F.set_index("pdb")
B = "C(period) + l2_units + res_median"
PARS = B + " + cap + enz + fid_only + sars2"


def fit(formula, d, var, cluster="lab", label="", subset="all EM papers"):
    d = d.dropna(subset=[var])
    try:
        m = smf.glm(formula, d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
        ci = np.exp(m.conf_int().loc[var])
        return dict(test=label, subset=subset, var=var, n=len(d), n_err=int(d.err.sum()), err_mean=d[d.err == 1][var].mean(),
                    ctrl_mean=d[d.err == 0][var].mean(), OR=np.exp(m.params[var]), lo=ci[0], hi=ci[1], p=m.pvalues[var])
    except Exception as ex:
        return dict(test=label, subset=subset, var=var, n=len(d), OR=np.nan, p=np.nan, note=str(ex)[:100])


R = []
# a) family / accession, antibody-structure competition (other labs, +/-6 and 12 months), paper = max over its entries
agg = []
for _, p in T.iterrows():
    ff = fe.reindex([x for x in p.pdbs.split(";") if x in fe.index])
    agg.append(dict(pkey=p.pkey, **{k: ff[k].max() for k in ["labs_acc_6m", "labs_fam_6m", "labs_acc_12m", "labs_fam_12m", "labs_fam_contact_6m"]}))
T = T.merge(pd.DataFrame(agg), on="pkey")
for k in ["labs_acc_6m", "labs_fam_6m", "labs_acc_12m", "labs_fam_12m", "labs_fam_contact_6m"]:
    T["any_" + k] = np.where(T[k].notna(), (T[k] > 0).astype(float), np.nan)
    R.append(fit(f"err ~ any_{k} + {B}", T, "any_" + k, label=f"any other-lab antibody structure, {k} [base]"))
    R.append(fit(f"err ~ any_{k} + {PARS}", T, "any_" + k, label=f"any other-lab antibody structure, {k} [parsimonious]"))
    d = T[(T.viral_org == 1) & (T.sars2 == 0)]
    R.append(fit(f"err ~ any_{k} + {B}", d, "any_" + k, label=f"{k} [base]", subset="viral non-SARS2"))
# b) complete windows
d = T[T.dep_first < "2025-01-01"]
R.append(fit(f"err ~ any_comp + {B}", d, "any_comp", label="base", subset="deposited before 2025 (full +180 d window)"))
R.append(fit(f"err ~ any_comp + {PARS}", d, "any_comp", label="parsimonious", subset="deposited before 2025 (full +180 d window)"))

# c)+d) recompute PDB-wide accession competition for X-ray papers and copied-only EM papers
c = pd.read_csv(f"{DEEP}/chains.csv.gz", usecols=["pdb", "chain"])
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
        info[r["rcsb_id"].lower()] = (((r.get("rcsb_primary_citation") or {}).get("pdbx_database_id_DOI") or "").lower() or None, a.get("deposit_date", "")[:10])
last = {}
with open("/tmp/supp/pdbwide/entries.idx", encoding="latin-1") as fh:
    for i, line in enumerate(fh):
        if i < 2:
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) > 5:
            au = [x.strip() for x in re.split(r",\s(?=[A-Z][^,]*\.)", f[5]) if x.strip()]
            last[f[0].lower()] = re.sub(r"[^a-z]", "", au[-1].split(",")[0].lower()) if au else ""
E = pd.DataFrame([(p, *info.get(p, (None, None)), last.get(p, "")) for p in ent_acc.index], columns=["pdb", "doi", "dep", "last"])
E["dep"] = pd.to_datetime(E.dep, errors="coerce")
E["acc"] = E.pdb.map(ent_acc)
by_acc = {a: g for a, g in E.explode("acc").groupby("acc")}
P = pd.read_csv(f"{BASE}/paper_meta/paper_analysis_table.csv", parse_dates=["dep_first"])
P = P.dropna(subset=["res_median"])
P["plast"] = P.last_author.fillna("").str.split(",").str[0].str.lower().str.replace(r"[^a-z]", "", regex=True)
sel = P[((P.method == "X-ray") & P.group.isin(["error", "clean"])) | ((P.method == "EM") & (P.group == "copied_only"))].copy()
vals = []
for _, p in sel.iterrows():
    pdbs = p.pdbs.split(";")
    accs = [a for a in set().union(*[ent_acc.get(x, set()) for x in pdbs]) if a in by_acc]
    if not accs:
        vals.append(np.nan)
        continue
    G = pd.concat([by_acc[a] for a in accs]).drop_duplicates("pdb")
    G = G[~G.pdb.isin(pdbs)]
    o = G[(G["last"] != p.plast) & ((G.doi != p.doi) | G.doi.isna())]
    key = o.doi.fillna("pdb:" + o.pdb)
    vals.append(key[(o.dep - p.dep_first).dt.days.abs() <= 180].nunique())
sel["comp_dep180"] = vals
sel["any_comp"] = np.where(sel.comp_dep180.notna(), (sel.comp_dep180 > 0).astype(float), np.nan)
sel["lab"] = sel.plast
sel["err"] = (sel.group == "error").astype(int)
X = sel[sel.method == "X-ray"]
R.append(fit(f"err ~ any_comp + {B}", X, "any_comp", label="X-ray papers, base", subset="X-ray error vs clean papers (2018+)"))
R.append(fit(f"err ~ any_comp + {B} + sars2", X, "any_comp", label="X-ray papers, + sars2", subset="X-ray error vs clean papers (2018+)"))
co = sel[sel.method == "EM"].copy()
co["err"] = 0
print("copied-only EM papers:", len(co), "with competitor:", co.any_comp.mean())
T2 = pd.concat([T[["err", "any_comp", "period", "l2_units", "res_median", "lab", "sars2"]], co[["err", "any_comp", "period", "l2_units", "res_median", "lab", "sars2"]]])
T2["sars2"] = T2.sars2.astype(int)
R.append(fit(f"err ~ any_comp + {B}", T2, "any_comp", label="copied/artifact-only papers added as controls, base", subset="EM"))
Rt = pd.DataFrame(R)
Rt.to_csv(f"{OUT}/tests_confounding_3.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 80, "display.max_rows", 200)
print(Rt.round(4).to_string())
