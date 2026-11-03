"""Quick checks used by the completeness critic (no new claims; reproduces numbers cited in the gap list).

1. kappa Pro8 'fingerprint' by period at antibody-in-paper unit level (EM, chains with Pro8 at IMGT 8).
2. Unit-level and paper-level EM error rate by deposition half-year (amyloid excluded), and the labs carrying peaks.
3. Units per mode and artifact-ledger flag (how many units lack any SAbR-independent confirmation).
"""
import re
import pandas as pd

SP = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"
D = "/home/user/SAbR/register_error_analysis/retest/deeper"
OUT = D + "/completeness_critic"

ch = pd.read_csv(SP + "/chains.csv.gz")
amy = set(pd.read_csv(D + "/pdb_meta/tables/amyloid_fibril_chains.csv").pdb.str.upper())

# 1. fingerprint by period
cs = pd.read_csv(D + "/pdb_meta/tables/chain_cis_sheet.csv.gz")


def has(s, p):
    if pd.isna(s):
        return False
    return p in [re.sub("[A-Z]+$", "", x.strip()) for x in str(s).split(";")]


k = cs[(cs.ctype == "K") & cs.method.str.contains("EM")].copy()
k = k[k.v_pro_pos.apply(lambda s: has(s, "8"))]
k["cis8"] = k.v_cis_pos.apply(lambda s: has(s, "8"))
k = k.merge(ch[["pdb", "chain", "unit", "year", "irm", "sw_ISOLDE"]], on=["pdb", "chain"], how="left")
k["period"] = pd.cut(k.year, [0, 2019, 2021, 2023, 2030], labels=["<=2019", "2020-21", "2022-23", ">=2024"])
k = k[(k.err_first == 1) | (k.err_any == 0)]
u = k.groupby("unit").agg(period=("period", "first"), all_trans=("cis8", lambda x: (~x).all()),
                          err=("err_first", "max")).reset_index()
t1 = u.groupby(["period", "all_trans"], observed=True).agg(units=("err", "size"), err_units=("err", "sum"),
                                                            err_rate=("err", "mean")).reset_index()
t1["share_of_period"] = t1.units / t1.groupby("period", observed=True).units.transform("sum")
t1.to_csv(OUT + "/kappa_pro8_by_period_units.csv", index=False)
print(t1.round(3).to_string())
print("ISOLDE recorded, cis-Pro8 share (EM kappa, >=2022):",
      k[k.year >= 2022].groupby("sw_ISOLDE").cis8.agg(["size", "mean"]).round(3).to_dict())

# 2. time trend
c = ch[ch.method.str.contains("EM") & ~ch.pdb.str.upper().isin(amy)].copy()
d = pd.to_datetime(c.deposit)
c["half"] = d.dt.year.astype(str) + "H" + ((d.dt.month > 6) + 1).astype(str)
uu = c.groupby("unit").agg(half=("half", "min"), err=("err_first", "max"), anyr=("err_any", "max"),
                           paper=("paper", "first"), lab=("last_author", "first")).reset_index()
uu = uu[(uu.err == 1) | (uu.anyr == 0)]
t2 = uu.groupby("half").agg(units=("err", "size"), err_units=("err", "sum"), unit_rate=("err", "mean"),
                            papers=("paper", "nunique")).reset_index()
pp = uu.groupby(["half", "paper"]).err.max().reset_index()
t2 = t2.merge(pp.groupby("half").err.agg(err_papers="sum", paper_rate="mean").reset_index(), on="half")
top = (uu[uu.err == 1].groupby(["half", "lab"]).size().rename("n").reset_index()
       .sort_values(["half", "n"], ascending=[True, False]).groupby("half").head(1))
t2 = t2.merge(top.rename(columns={"lab": "top_lab", "n": "top_lab_err_units"}), on="half", how="left")
t2 = t2[t2.units >= 30]
t2.to_csv(OUT + "/em_error_rate_by_half_year.csv", index=False)
print(t2.round(3).to_string())

# 3. ledger
r = pd.read_csv(D + "/synthesis/artifact_ledger_runs.csv")
t3 = r.groupby(["mode", "flag"]).unit.nunique().unstack(fill_value=0)
t3.to_csv(OUT + "/ledger_units_by_mode_flag.csv")
print(t3.to_string())
