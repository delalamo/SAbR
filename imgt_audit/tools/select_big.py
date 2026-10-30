"""Build the large-scale job list: antibodies (SAbDab2), TCRs (STCRDab), other families (RCSB Pfam reps)."""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from rcsb_search import pfam_reps

BIG = os.path.join(os.path.dirname(__file__), "..", "big")
jobs = {}


def add(pdb, types, chains=None, entities=None, src=""):
    j = jobs.setdefault(pdb.lower(), {"types": set(), "chains": set(), "entities": set(), "src": set()})
    j["types"] |= set(types); j["src"].add(src)
    if chains: j["chains"] |= {c for c in chains if isinstance(c, str) and c not in ("NA", "")}
    if entities: j["entities"] |= set(entities)


# antibodies
s = pd.read_csv(os.path.join(BIG, "sab_all.tsv"), low_memory=False)
s = s[(s.method == "XRAY") & (pd.to_numeric(s.resolution, errors="coerce") <= 2.5) & (s.model == 0)]
s["pdb"] = s.PDB.str[-4:]
s["stratum"] = s.heavy_subclass.fillna("noH") + "|" + s.light_subclass.fillna("noL") + "|" + s.light_ctype.fillna("-")
s = s.sort_values("resolution").drop_duplicates("VH")
picked = s.groupby("stratum", group_keys=False).head(3)
picked = picked.sample(n=min(380, len(picked)), random_state=1)
for _, r in picked.iterrows():
    add(r.pdb, "VC", chains=[r.Hchain, r.Lchain], src="sabdab")
print("antibody entries", len({r.pdb for _, r in picked.iterrows()}), "strata", s.stratum.nunique())

# TCRs
t = pd.read_csv(os.path.join(BIG, "stcrdab_summary.tsv"), sep="\t")
t = t[(t.method.str.contains("X-RAY")) & (pd.to_numeric(t.resolution, errors="coerce") <= 3.0)]
t = t.sort_values("resolution")
ab = t[t.TCRtype == "abTCR"].drop_duplicates(["alpha_subgroup", "beta_subgroup"]).head(150)
gd = t[t.TCRtype == "gdTCR"].drop_duplicates("pdb")
for _, r in pd.concat([ab, gd]).iterrows():
    add(r.pdb, "VCG", chains=[r.Achain, r.Bchain, r.Gchain, r.Dchain, r.mhc_chain1, r.mhc_chain2], src="stcrdab")
print("tcr entries", len(ab) + len(gd))

# other families
fams = {"PF07686": ("VC", 120), "PF07654": ("CG", 160), "PF00047": ("VC", 30), "PF13927": ("VC", 60),
        "PF13895": ("VC", 31), "PF07679": ("VC", 72), "PF05790": ("VC", 8), "PF00129": ("CG", 48),
        "PF00993": ("CG", 10), "PF00969": ("CG", 21), "PF00041": ("F", 76), "PF00530": ("S", 14),
        "PF15494": ("S", 4), "PF00024": ("A", 7)}
for pf, (types, cap) in fams.items():
    _, reps = pfam_reps(pf, 95, 3.0)
    for rep in reps[:cap]:
        pdb, ent = rep.split("_")
        add(pdb, types, entities=[ent], src=pf)
    print(pf, len(reps[:cap]))
out = [dict(pdb=k, types="".join(sorted(v["types"])), chains=sorted(v["chains"]), entities=sorted(v["entities"]),
            src=sorted(v["src"])) for k, v in jobs.items()]
json.dump(out, open(os.path.join(BIG, "jobs.json"), "w"), indent=0)
print("total entries", len(out))
