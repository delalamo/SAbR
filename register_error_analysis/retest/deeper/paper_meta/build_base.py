#!/usr/bin/env python3
"""Paper-level table of PDB-derived attributes for antibody entries deposited 2018 onward.

One row per (paper, method-set). The main analysis set is EM: a paper's EM entries deposited >= 2018.
Error paper  = at least one EM entry with a first-occurrence error (err_first).
Clean paper  = none of its EM entries has any SAbR/ANARCI disagreement run (err_any False).
Copied-only  = runs only of copied/artifact type (excluded from the main contrast, kept as a 3rd group).
The same is done for X-ray entries as a secondary set.
"""
import json
import numpy as np
import pandas as pd

DEEP = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta"

e = pd.read_csv(f"{DEEP}/entries.csv")
c = pd.read_csv(f"{DEEP}/chains.csv.gz")
e["dep"] = pd.to_datetime(e.deposit)
e["rel"] = pd.to_datetime(e.release)
e = e[e.dep >= "2018-01-01"].copy()
e["doi_l"] = e.doi.str.strip().str.lower()
e["pkey"] = e.doi_l.fillna("nodoi:" + e.pdb)
c = c[c.pdb.isin(e.pdb)].copy()
c = c.merge(e[["pdb", "pkey"]], on="pdb")

# total PDB entries citing each DOI (any resolution / method) from the PDB-wide citation dump
tot = {}
with open("/tmp/supp/pdbwide/cite_all.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        d = ((r.get("rcsb_primary_citation") or {}).get("pdbx_database_id_DOI") or "").strip().lower()
        if d:
            tot[d] = tot.get(d, 0) + 1

rows = []
for method in ["EM", "X-ray"]:
    em = e[e.method == method]
    cm = c[c.method == method]
    for pk, g in em.groupby("pkey"):
        cg = cm[cm.pdb.isin(g.pdb)]
        units = cg.groupby("unit").agg(err=("err_first", "any"), role=("role", "first"),
                                       aux=("aux_reagent", "any"), vhh=("vhh", "max"),
                                       sars2=("sars2", "any"), species=("species", "first"))
        roles = units.role.value_counts()
        n_units = len(units)
        n_aux_units = int(((units.role == "Auxiliary") | units.aux).sum())
        row = dict(
            pkey=pk, method=method, doi=g.doi_l.iloc[0] if g.doi_l.notna().any() else None,
            journal_pdb=g.journal.iloc[0], last_author=g.last_author.iloc[0],
            n_entries=len(g), n_entries_pdb_total=tot.get(pk, len(g)) if not pk.startswith("nodoi:") else 1,
            n_vchains=int(g.n_V_chains.sum()), n_units=n_units,
            n_err_units=int(units.err.sum()),
            n_err_entries=int(g.err_first.sum()), n_errany_entries=int(g.err_any.sum()),
            err_first=bool(g.err_first.any()), err_any=bool(g.err_any.any()),
            res_median=float(g.res.median()), res_min=float(g.res.min()),
            dep_first=g.dep.min(), dep_last=g.dep.max(), rel_first=g.rel.min(), rel_last=g.rel.max(),
            dep_span_days=int((g.dep.max() - g.dep.min()).days),
            hold_days_median=float((g.rel - g.dep).dt.days.median()),
            sars2=bool(g.sars2.any()),
            frac_aux_units=n_aux_units / n_units if n_units else np.nan,
            tool_paper=bool(n_units and n_aux_units == n_units),
            any_vhh=bool(units.vhh.max() == 1) if n_units else False,
            any_mouse=bool((units.species == "mouse").any()),
            role_alone=int(roles.get("Antibody alone", 0)),
            novel_target=float(cg.novel_target.max()) if cg.novel_target.notna().any() else np.nan,
            competitors=float(cg.competitors.max()) if cg.competitors.notna().any() else np.nan,
            tier=cg.tier.iloc[0] if len(cg) else None,
            title_pdb=g.title.iloc[0],
            pdbs=";".join(sorted(g.pdb)),
            err_pdbs=";".join(sorted(g.pdb[g.err_first])),
            modes=";".join(sorted({m for x in g.modes.dropna() for m in str(x).split(";") if m})),
        )
        row["amyloid_fibril"] = bool(g.title.str.contains(r"(?i)amyloid fibril|AL amyloid|amyloidosis|cardiac amyloid|amyloid AL").any())
        row["group"] = "error" if row["err_first"] else ("clean" if not row["err_any"] else "copied_only")
        if row["amyloid_fibril"]:
            # light-chain amyloid fibrils are not Ig-folded: SAbR-vs-ANARCI disagreement there is not a register error
            row["group"] = "amyloid_artifact"
        rows.append(row)

P = pd.DataFrame(rows)
P["dep_year"] = P.dep_first.dt.year
P["period"] = pd.cut(P.dep_year, [2017, 2019, 2020, 2021, 2023, 2026],
                     labels=["2018-19", "2020", "2021", "2022-23", "2024-26"]).astype(str)
P.to_csv(f"{OUT}/paper_base.csv", index=False)
print(P.groupby(["method", "group"]).size())
print(pd.crosstab([P.method, P.period], P.group))
