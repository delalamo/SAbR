#!/usr/bin/env python3
"""Do error papers from DIFFERENT labs share people, institutions/facilities or funders?

Set: published cryo-EM papers (EM entries deposited 2018+), error vs clean (copied-only excluded).
1. Global test: pairs of error papers with different last authors (PDB last_author and paper last author both differ)
   that share >=1 author. Null: permute the error label within strata of period x n_units-bin x resolution-bin
   (keeps each paper's baseline risk), 2000 permutations.
2. Per-author enrichment (authors with >=3 papers): observed error papers vs expected from the baseline model
   (period + log2 n_units + resolution), Poisson-binomial upper tail; BH over authors. Also restricted to the
   author's papers where they are NOT the last author (middle-author link across labs).
3. Same for institution tokens from affiliations (all authors) and for individual funder names.
Author key: ORCID when available, else normalised surname + initials.
"""
import re, unicodedata, itertools, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta"
rng = np.random.default_rng(7)
D = pd.read_csv(f"{OUT}/paper_analysis_table.csv")
A = pd.read_csv(f"{OUT}/paper_authors.csv.gz")
X = D[(D.method == "EM") & D.group.isin(["error", "clean"]) & D.published & D.n_authors.notna()].dropna(subset=["res_median"]).copy()
X = X.reset_index(drop=True)
m = smf.logit("err ~ C(period) + l2_units + res_median", X).fit(disp=0)
X["p_hat"] = m.predict(X)
X["stratum"] = X.period + "|" + pd.cut(X.n_units, [0, 1, 2, 4, 1000]).astype(str) + "|" + pd.cut(X.res_median, [0, 3.3, 3.7, 6]).astype(str)


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


A = A[A.doi.isin(X.doi)].copy()
A["akey"] = np.where(A.orcid.notna(), "orcid:" + A.orcid.astype(str), "n:" + A["last"].map(norm) + "_" + A.initials.fillna("").map(norm))
# map name-keys to ORCID where the same name has exactly one ORCID in the data
nm = A.assign(nk="n:" + A["last"].map(norm) + "_" + A.initials.fillna("").map(norm))
one = nm[nm.orcid.notna()].groupby("nk").orcid.nunique()
one = one[one == 1].index
o_map = nm[nm.orcid.notna() & nm.nk.isin(one)].groupby("nk").orcid.first()
A["akey"] = np.where(A.orcid.isna() & nm.nk.isin(o_map.index), "orcid:" + nm.nk.map(o_map).astype(str), A.akey)
A["is_last"] = A.pos == A.n - 1
A["name"] = A["last"].astype(str) + " " + A.initials.fillna("").astype(str)

doi_idx = {d: i for i, d in enumerate(X.doi)}
lab = X.last_author.fillna("").map(norm).values  # PDB last author
plast = A[A.is_last].set_index("doi").akey.reindex(X.doi).values  # paper last author
paper_auth = A.groupby("doi").akey.apply(set).reindex(X.doi)

# ---------- 1. global co-authorship test across labs
inv = {}
for i, s in enumerate(paper_auth):
    for a in (s if isinstance(s, set) else []):
        inv.setdefault(a, []).append(i)
share = set()
for a, idx in inv.items():
    if len(idx) > 1 and len(idx) < 200:
        for i, j in itertools.combinations(idx, 2):
            if lab[i] != lab[j] and plast[i] != plast[j]:
                share.add((i, j))
share = np.array(sorted(share))
err = X.err.values.astype(bool)


def stat(e):
    return int((e[share[:, 0]] & e[share[:, 1]]).sum())


obs = stat(err)
strata = X.stratum.values
groups = [np.where(strata == s)[0] for s in np.unique(strata)]
null = []
for _ in range(2000):
    e2 = err.copy()
    for g in groups:
        e2[g] = rng.permutation(err[g])
    null.append(stat(e2))
null = np.array(null)
glob = dict(test="error-paper pairs from different labs sharing >=1 author", observed=obs,
            null_mean=null.mean(), null_sd=null.std(), ratio=obs / null.mean(), p_perm=(1 + (null >= obs).sum()) / 2001,
            n_pairs_sharing=len(share), n_papers=len(X), n_error=int(err.sum()))
print(glob)

# same but counting only links via a NON-last author in both papers (people who are not the PI)
nl_auth = A[~A.is_last].groupby("doi").akey.apply(set).reindex(X.doi)
inv2 = {}
for i, s in enumerate(nl_auth):
    for a in (s if isinstance(s, set) else []):
        inv2.setdefault(a, []).append(i)
share2 = set()
for a, idx in inv2.items():
    if 1 < len(idx) < 200:
        for i, j in itertools.combinations(idx, 2):
            if lab[i] != lab[j] and plast[i] != plast[j]:
                share2.add((i, j))
share2 = np.array(sorted(share2))
obs2 = int((err[share2[:, 0]] & err[share2[:, 1]]).sum())
null2 = []
for _ in range(2000):
    e2 = err.copy()
    for g in groups:
        e2[g] = rng.permutation(err[g])
    null2.append(int((e2[share2[:, 0]] & e2[share2[:, 1]]).sum()))
null2 = np.array(null2)
glob2 = dict(test="... sharing >=1 non-last author (middle-author link)", observed=obs2, null_mean=null2.mean(),
             null_sd=null2.std(), ratio=obs2 / null2.mean(), p_perm=(1 + (null2 >= obs2).sum()) / 2001,
             n_pairs_sharing=len(share2), n_papers=len(X), n_error=int(err.sum()))
print(glob2)

# same-lab baseline for comparison: error pairs within the same PDB last author
same = []
for a, idx in pd.Series(range(len(X))).groupby(lab).apply(list).items():
    if a and len(idx) > 1:
        same += list(itertools.combinations(idx, 2))
same = np.array(same)
obs3 = int((err[same[:, 0]] & err[same[:, 1]]).sum())
null3 = []
for _ in range(2000):
    e2 = err.copy()
    for g in groups:
        e2[g] = rng.permutation(err[g])
    null3.append(int((e2[same[:, 0]] & e2[same[:, 1]]).sum()))
null3 = np.array(null3)
glob3 = dict(test="error-paper pairs within the SAME lab (PDB last author), for comparison", observed=obs3,
             null_mean=null3.mean(), null_sd=null3.std(), ratio=obs3 / null3.mean(), p_perm=(1 + (null3 >= obs3).sum()) / 2001,
             n_pairs_sharing=len(same), n_papers=len(X), n_error=int(err.sum()))
print(glob3)
pd.DataFrame([glob, glob2, glob3]).to_csv(f"{OUT}/network_global.csv", index=False)


# ---------- 2/3. per-entity enrichment
def pb_upper(ps, k):
    """P(X >= k) for a Poisson-binomial with probabilities ps"""
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return float(dist[k:].sum())


def enrich(memb, label, min_papers=3):
    """memb: DataFrame doi, entity"""
    memb = memb.drop_duplicates()
    memb = memb[memb.doi.isin(doi_idx)]
    out = []
    for ent, g in memb.groupby("entity"):
        idx = [doi_idx[d] for d in g.doi]
        if len(idx) < min_papers:
            continue
        o = int(err[idx].sum())
        ps = X.p_hat.values[idx]
        labs_err = len({lab[i] for i in idx if err[i]})
        out.append(dict(kind=label, entity=ent, n_papers=len(idx), n_err=o, expected=ps.sum(),
                        p_upper=pb_upper(ps, o) if o > 0 else 1.0, n_labs=len({lab[i] for i in idx}),
                        n_err_labs=labs_err))
    R = pd.DataFrame(out)
    if len(R):
        R["q"] = multipletests(R.p_upper, method="fdr_bh")[1]
    return R.sort_values("p_upper")


# authors (any position) and authors in non-last position only
names = A.groupby("akey").name.agg(lambda s: s.value_counts().index[0])
ra = enrich(A[["doi", "akey"]].rename(columns={"akey": "entity"}), "author_any")
rb = enrich(A[~A.is_last][["doi", "akey"]].rename(columns={"akey": "entity"}), "author_nonlast")
for r in (ra, rb):
    r["name"] = r.entity.map(names)

# institutions: comma-separated affiliation segments naming an institution, normalised
INST = re.compile(r"(Universit|Institut|College|School|Hospital|Center|Centre|Academy|Laborator|Facility|Inc\b|Ltd|GmbH|Foundation|Council|NIH|NIAID|Agency|Corporation|Pharma)", re.I)


def inst_tokens(aff):
    toks = set()
    if not isinstance(aff, str):
        return toks
    aff = re.sub(r"\S+@\S+|Electronic address:.*", "", aff)
    for seg in re.split(r"[,;]", aff):
        seg = seg.strip().strip(".")
        if INST.search(seg) and 4 < len(seg) < 90 and not re.match(r"(?i)^(department|division|school of|laboratory of|program|graduate)", seg):
            toks.add(re.sub(r"\s+", " ", seg))
    return toks


IM = [(d, t) for d, a in zip(A.doi, A.aff) for t in inst_tokens(a)]
IM = pd.DataFrame(IM, columns=["doi", "entity"])
ri = enrich(IM, "institution", 5)
# EM facility mentions in any affiliation
FAC = {"PNCC (Pacific Northwest Cryo-EM Center)": r"Pacific Northwest Cryo|PNCC", "SLAC/S2C2": r"SLAC|Stanford-SLAC|S2C2",
       "NCI/Frederick cryo-EM (NCEF)": r"Frederick National|National Cryo-EM Facility|NCEF",
       "NYSBC/Simons": r"New York Structural Biology|Simons Electron|NYSBC", "Janelia": r"Janelia",
       "eBIC/Diamond": r"Diamond Light|eBIC|Electron Bio-Imaging", "EMBL": r"\bEMBL\b|European Molecular Biology Laboratory",
       "ESRF": r"ESRF|European Synchrotron", "MRC-LMB": r"MRC Laboratory of Molecular Biology|MRC-LMB|LMB",
       "Any 'cryo-EM facility/center'": r"(?i)cryo-?EM (facility|center|centre|core)|cryo-electron microscopy (facility|center|centre|core)|electron microscopy (facility|core|center|centre)"}
FM = [(d, f) for d, a in zip(A.doi, A.aff) if isinstance(a, str) for f, pat in FAC.items() if re.search(pat, a)]
rf = enrich(pd.DataFrame(FM, columns=["doi", "entity"]), "facility", 3)
# funders (individual names)
FU = []
for d, t in zip(X.doi, X.funders_text):
    if isinstance(t, str):
        for f in {x.strip() for x in t.split(" | ") if x.strip()}:
            FU.append((d, re.sub(r"\s+", " ", f)))
rfu = enrich(pd.DataFrame(FU, columns=["doi", "entity"]), "funder", 5)
# journals as an entity too, for completeness
rj = enrich(pd.DataFrame({"doi": X.doi, "entity": X.journal_n}), "journal", 10)

ALL = pd.concat([ra, rb, ri, rf, rfu, rj], ignore_index=True)
ALL.to_csv(f"{OUT}/network_enrichment.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 60)
for k, r in ALL.groupby("kind"):
    print("==", k, "tested:", len(r), " q<0.1:", (r.q < 0.1).sum())
    print(r.sort_values("p_upper").head(10)[["entity", "name", "n_papers", "n_err", "expected", "p_upper", "q", "n_labs", "n_err_labs"]].to_string())
