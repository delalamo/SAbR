#!/usr/bin/env python3
"""Reproduce the PM0 counts: 15 AL fibril entries, all err_first; units; runs; modes; papers."""
import pandas as pd, numpy as np
B = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/verify/paper_meta_PM0_amyloid_artifact_confounding"
C = pd.read_csv(f"{B}/chains.csv.gz"); E = pd.read_csv(f"{B}/entries.csv")
R = pd.read_csv(f"{B}/runs_first_occurrence.csv"); RA = pd.read_csv(f"{B}/runs_all.csv")
AMY = ['6ic3','6z1o','7nsl','9els','9faa','9fab','9fac','9liv','9liw','9lix','9liy','9lj0','9lja','9ljb','9v91']
# independent identification: title regex (broad) and chain description
reg = r"(?i)amyloid(?!-beta| beta|\s*\()|fibril"
hit = set(E.pdb[E.title.str.contains(reg, na=False)])
print("title regex hits:", sorted(hit))
print("hits not in AMY:", sorted(hit - set(AMY)), " AMY not hit:", sorted(set(AMY) - hit))
EA = E[E.pdb.isin(AMY)]
print("method:", EA.method.value_counts().to_dict(), " err_first:", EA.err_first.sum(), "/", len(EA))
print("entry modes:", EA.modes.value_counts().to_dict())
# EM error entries / units / runs
em_err_entries = E[(E.method == "EM") & E.err_first]
print("EM error entries:", len(em_err_entries), " amyloid among them:", em_err_entries.pdb.isin(AMY).sum())
CE = C[C.err_first.astype(bool)]
u_all = CE.groupby("unit").agg(method=("method", "first"), amy=("pdb", lambda s: s.isin(AMY).any()))
print("error units all:", len(u_all), " EM:", (u_all.method == "EM").sum(), " amyloid units:", u_all.amy.sum(),
      " amyloid EM units:", (u_all.amy & (u_all.method == "EM")).sum())
print("first-occurrence runs:", len(R), " amyloid:", R.pdb.isin(AMY).sum(), f"({R.pdb.isin(AMY).mean():.1%})")
print("EM first-occurrence runs:", (R.method == "EM").sum(), " amyloid share of EM runs:",
      f"{R[R.method=='EM'].pdb.isin(AMY).mean():.1%}")
print("all runs:", len(RA), " amyloid:", RA.pdb.isin(AMY).sum())
# units for amyloid
print(C[C.pdb.isin(AMY)].groupby("unit").pdb.apply(lambda s: ";".join(sorted(set(s)))))
# mode counts by run among amyloid
print("run modes amyloid:", R[R.pdb.isin(AMY)].groupby("mode").pdb.nunique().to_dict())
# mode share of amyloid among all first-occurrence runs, by mode (entry level)
em = R[R.method == "EM"]
t = em.groupby("mode").agg(entries=("pdb", "nunique"), amy_entries=("pdb", lambda s: s[s.isin(AMY)].nunique()),
                           runs=("pdb", "size"), amy_runs=("pdb", lambda s: s.isin(AMY).sum()))
t["amy_entry_share"] = t.amy_entries / t.entries
print(t)
t.to_csv(f"{OUT}/mode_share_amyloid_EM.csv")
