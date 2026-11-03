#!/usr/bin/env python3
"""Fetch PDB obsolete/supersede records (pdbx_database_PDB_obs_spr) for every EM entry of the analysed papers
(RCSB GraphQL), to see whether superseded (re-deposited, possibly corrected) entries are concentrated in highly
cited papers and whether this explains the citation association."""
import json, os, subprocess, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM2_lower_citations_confounding"
CACHE = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/verify_PM2"
X = pd.read_csv(f"{OUT}/v01_analysis_frame.csv.gz")
ids = sorted({p.upper() for s in X.pdbs for p in s.split(";")})
fn = f"{CACHE}/obs_spr.json"
if os.path.exists(fn):
    res = json.load(open(fn))
else:
    res = {}
    for i in range(0, len(ids), 150):
        chunk = ids[i:i + 150]
        q = "{ entries(entry_ids:[%s]) { rcsb_id pdbx_database_PDB_obs_spr { id pdb_id replace_pdb_id date details } } }" % \
            ",".join(f'\\"{c}\\"' for c in chunk)
        body = '{"query":"%s"}' % q
        r = subprocess.run(["curl", "-sS", "-X", "POST", "https://data.rcsb.org/graphql", "-H", "Content-Type: application/json",
                            "-d", body], capture_output=True, text=True, timeout=120)
        js = json.loads(r.stdout)
        for e in js["data"]["entries"]:
            res[e["rcsb_id"]] = e["pdbx_database_PDB_obs_spr"]
    json.dump(res, open(fn, "w"))
print("entries", len(ids), "fetched", len(res), "with obs_spr", sum(1 for v in res.values() if v))
X["superseding"] = [float(any(res.get(p.upper()) for p in s.split(";"))) for s in X.pdbs]
print(X.groupby("err").superseding.agg(["mean", "sum", "size"]))
XV = "l_cites_per_year"
BASE = "C(period) + l2_units + res_median"
out = []
d = X
if d.superseding.sum() >= 5:
    m = smf.logit(f"superseding ~ {XV} + {BASE}", d).fit(disp=0)
    out.append(dict(test="paper has a superseding (replacement) entry ~ cites", OR=np.exp(m.params[XV]), p=m.pvalues[XV], n=len(d)))
    print("superseding ~ cites", np.exp(m.params[XV]), m.pvalues[XV])
m = smf.logit(f"err ~ {XV} + {BASE}", d[d.superseding == 0]).fit(disp=0, cov_type="cluster",
                                                                  cov_kwds={"groups": pd.factorize(d[d.superseding == 0].lab)[0]})
ci = m.conf_int().loc[XV]
out.append(dict(test="citation OR, papers without superseding entries", OR=np.exp(m.params[XV]), lo=np.exp(ci[0]), hi=np.exp(ci[1]),
                p=m.pvalues[XV], n=int((d.superseding == 0).sum()), n_err=int(d[d.superseding == 0].err.sum())))
print(out[-1])
pd.DataFrame(out).to_csv(f"{OUT}/v05_superseded.csv", index=False)
rows = [dict(pdb=k, obs_spr=json.dumps(v)) for k, v in res.items() if v]
pd.DataFrame(rows).to_csv(f"{OUT}/v05_superseding_entries.csv", index=False)
