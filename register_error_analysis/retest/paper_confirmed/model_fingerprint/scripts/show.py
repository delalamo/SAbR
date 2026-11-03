"""Compact printer for result tables: python3 show.py table.csv [filter-substring]"""
import sys, pandas as pd, numpy as np
R = pd.read_csv(sys.argv[1])
if len(sys.argv) > 2: R = R[R.test.str.contains(sys.argv[2], regex=False)]
f = lambda x: '' if pd.isna(x) else f'{x:.3g}'
for _, r in R.iterrows():
    s = f"{r.test[:70]} | {r.case_set} {r.method} | papers {f(r.get('case_papers'))} | {r.case_val} vs {r.ctrl_val}"
    if 'crude_OR' in r and pd.notna(r.get('crude_OR')): s += f" | crude {f(r.crude_OR)} ({f(r.crude_lo)}-{f(r.crude_hi)}) p={f(r.fisher_p)}"
    if 'crude_diff' in r and pd.notna(r.get('crude_diff')): s += f" | diff {f(r.crude_diff)} MWp={f(r.mw_p)}"
    for t in ['adj', 'pw', 'paper']:
        k = f'{t}_OR' if f'{t}_OR' in r and pd.notna(r.get(f'{t}_OR')) else f'{t}_est'
        if k in r and pd.notna(r.get(k)): s += f" | {t} {f(r[k])} ({f(r[t+'_lo'])},{f(r[t+'_hi'])}) p={f(r[t+'_p'])}"
    if pd.notna(r.get('paper_mw_p')): s += f" | paperMW p={f(r.paper_mw_p)}"
    for k in ["mde_OR_chains", "mde_OR_papers", "mde_80pct"]:
        if pd.notna(r.get(k)): s += f" | {k} {f(r[k])}"
    h = [f"{k[6:]}={f(r[k])}" for k in ['het_p_z', 'het_p_crude', 'het_p_3group'] if pd.notna(r.get(k))]
    if h: s += ' | het ' + ' '.join(h)
    print(s)
