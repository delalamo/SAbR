#!/usr/bin/env python3
"""Was the antigen being published by other labs at the same time? (paper level, PDB-wide)

Antigen of a paper = UniProt accessions (SIFTS) of the non-antibody chains in its EM entries (chains listed as
V chains in chains.csv are excluded; immunoglobulin UniProts are excluded by name pattern via SIFTS absence).
For each paper, PDB-wide competitors = other DOIs (different last author in entries.idx) whose entries contain
one of those UniProts, with
  * deposition within +/-180 days of the paper's first deposition (race at the bench), and
  * release date within the 180 days BEFORE the paper's publication date (scooped / pressure to publish).
Also: first-in-PDB (no earlier entry of any lab with that UniProt) and the paper's rank among same-antigen papers.
Test: error vs clean EM papers, logit adjusted for period + log2 n_units + resolution, and additionally sars2.
"""
import gzip, json, re, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
DEEP = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta"
D = pd.read_csv(f"{OUT}/paper_analysis_table.csv", parse_dates=["dep_first", "pub_date", "rel_first"])
c = pd.read_csv(f"{DEEP}/chains.csv.gz", usecols=["pdb", "chain"])
vchains = set(zip(c.pdb.str.lower(), c.chain.astype(str)))

S = pd.read_csv("/tmp/supp/pdbwide/pdb_chain_uniprot.tsv.gz", sep="\t", comment="#", usecols=[0, 1, 2], dtype=str)
S.columns = ["pdb", "chain", "uniprot"]
S = S.drop_duplicates()
S["pdb"] = S.pdb.str.lower()
# accessions that ever map to an antibody V chain in our scan are immunoglobulins -> drop everywhere
ig_acc = set(S[[(p, ch) in vchains for p, ch in zip(S.pdb, S.chain)]].uniprot)
S = S[~S.uniprot.isin(ig_acc)]
S = S[[(p, ch) not in vchains for p, ch in zip(S.pdb, S.chain)]]
ent_acc = S.groupby("pdb").uniprot.apply(set)

# PDB-wide entry info: deposit, release, doi, last author
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
E["rel"] = pd.to_datetime(E.rel, errors="coerce")
E["acc"] = E.pdb.map(ent_acc)
EX = E.explode("acc")
by_acc = {a: g for a, g in EX.groupby("acc")}

X = D[(D.method == "EM") & D.group.isin(["error", "clean"])].dropna(subset=["res_median"]).copy()
X["plast"] = X.last_author.fillna("").str.split(",").str[0].str.lower().str.replace(r"[^a-z]", "", regex=True)
rows = []
for _, p in X.iterrows():
    pdbs = p.pdbs.split(";")
    accs = set().union(*[ent_acc.get(x, set()) for x in pdbs])
    rec = dict(pkey=p.pkey, n_antigen_acc=len(accs))
    if not accs:
        rows.append(rec)
        continue
    G = pd.concat([by_acc[a] for a in accs if a in by_acc]).drop_duplicates("pdb")
    G = G[~G.pdb.isin(pdbs)]
    other = G[(G.last != p.plast) & ((G.doi != p.doi) | G.doi.isna())]
    key = other.doi.fillna("pdb:" + other.pdb)
    win = (other.dep - p.dep_first).dt.days.abs() <= 180
    rec["comp_dep180"] = key[win].nunique()
    if pd.notna(p.pub_date):
        pre = (other.rel < p.pub_date) & (other.rel >= p.pub_date - pd.Timedelta(days=180))
        rec["comp_rel180_before_pub"] = key[pre].nunique()
    rec["first_in_pdb"] = not (G.dep < p.dep_first).any()
    rec["prior_labs"] = G[G.dep < p.dep_first]["last"].nunique()
    rows.append(rec)
CT = pd.DataFrame(rows)
X = X.merge(CT, on="pkey", how="left")
X.to_csv(f"{OUT}/competition_table.csv", index=False)
for v in ["comp_dep180", "comp_rel180_before_pub", "prior_labs"]:
    X["l_" + v] = np.log2(1 + X[v])
X["any_comp_dep180"] = np.where(X.comp_dep180.notna(), (X.comp_dep180 > 0).astype(float), np.nan)
out = []
for v, cov in [(v, cov) for v in ["l_comp_dep180", "any_comp_dep180", "l_comp_rel180_before_pub", "first_in_pdb", "l_prior_labs"]
               for cov in ["C(period) + l2_units + res_median", "C(period) + l2_units + res_median + sars2"]]:
    d = X.dropna(subset=[v]).copy()
    d[v] = d[v].astype(float)
    m = smf.logit(f"err ~ {v} + {cov}", d).fit(disp=0)
    e, cc = d[d.err == 1][v], d[d.err == 0][v]
    out.append(dict(feature=v, covariates=cov, n_err=len(e), n_ctrl=len(cc), err_mean=e.mean(), ctrl_mean=cc.mean(),
                    err_median=e.median(), ctrl_median=cc.median(),
                    adj_OR=np.exp(m.params[v]), ci_lo=np.exp(m.conf_int().loc[v][0]), ci_hi=np.exp(m.conf_int().loc[v][1]),
                    p=m.pvalues[v]))
R = pd.DataFrame(out)
R.to_csv(f"{OUT}/tests_competition.csv", index=False)
pd.set_option("display.width", 250)
print(R.to_string())
# non-SARS-CoV-2 papers only
d = X[~X.sars2.astype(bool)].dropna(subset=["l_comp_dep180"])
m = smf.logit("err ~ l_comp_dep180 + C(period) + l2_units + res_median", d).fit(disp=0)
print("non-SARS2 only: OR per doubling", np.exp(m.params["l_comp_dep180"]), m.pvalues["l_comp_dep180"], len(d), d.err.sum())
print(X.groupby("err")[["n_antigen_acc", "comp_dep180", "comp_rel180_before_pub", "prior_labs"]].median())

# ---- sensitivity: antigen class and tool papers
X["viral_title"] = X.title_pdb.fillna("").str.contains(r"(?i)virus|viral|virion|capsid|spike|\bEnv\b|hemagglutinin|haemagglutinin|glycoprotein|SARS|HIV|influenza|ebola|lassa|nipah|RSV|dengue|zika|norovirus|rotavirus|polio|enterovirus|parvovirus|phage")
X["l_comp"] = np.log2(1 + X.comp_dep180)
X["any_comp"] = (X.comp_dep180 > 0).astype(float)
sens = []
for name, d, cov in [
        ("mapped antigen, + sars2 + viral", X[X.n_antigen_acc > 0], "C(period) + l2_units + res_median + sars2 + viral_title"),
        ("non-tool papers, + sars2 + viral", X[(X.n_antigen_acc > 0) & (X.frac_aux_units == 0)], "C(period) + l2_units + res_median + sars2 + viral_title"),
        ("viral non-SARS2 only", X[(X.n_antigen_acc > 0) & X.viral_title & ~X.sars2.astype(bool)], "C(period) + l2_units + res_median"),
        ("non-viral only", X[(X.n_antigen_acc > 0) & ~X.viral_title], "C(period) + l2_units + res_median"),
        ("SARS2 only", X[(X.n_antigen_acc > 0) & X.sars2.astype(bool)], "C(period) + l2_units + res_median")]:
    for v in ["l_comp", "any_comp"]:
        try:
            m = smf.logit(f"err ~ {v} + {cov}", d.dropna(subset=[v])).fit(disp=0)
            sens.append(dict(subset=name, feature=v, n=len(d), n_err=int(d.err.sum()),
                             err_mean=d[d.err == 1][v].mean(), ctrl_mean=d[d.err == 0][v].mean(),
                             adj_OR=np.exp(m.params[v]), p=m.pvalues[v]))
        except Exception as ex:
            sens.append(dict(subset=name, feature=v, n=len(d), n_err=int(d.err.sum()), adj_OR=np.nan, p=np.nan))
S2 = pd.DataFrame(sens)
S2.to_csv(f"{OUT}/tests_competition_sensitivity.csv", index=False)
print(S2.to_string())
print(pd.crosstab(X.viral_title, X.err, normalize="columns"))
