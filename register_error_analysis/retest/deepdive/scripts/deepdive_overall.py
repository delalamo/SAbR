"""Overall share of first-occurrence errors with (a) unbuilt residues within 5 positions or a truncated N-terminus
(mode A), (b) the adjacent CDR of atypical length for its chain type, (c) bottom-quartile local map fit at the site;
vs correct units at the same location (EM). Plus per-mode membership: papers, labs, years."""
import numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
O = '/home/user/SAbR/register_error_analysis/retest/deepdive/tables'
D = pd.read_pickle(f'{SP}/unexpl/deepdive_units_full.pkl'); D = D[D.method == 'EM'].copy()
ADJ = {'A': None, 'F': 'L1_atyp', 'B': 'L2_atyp', 'C': 'L3_atyp', 'D+': 'L3_atyp', 'D-': 'L3_atyp', 'E': 'L3_atyp'}
out = []
for m, g in D.groupby('mode'):
    q25 = g[~g.case].win_fit_pct.quantile(0.25)
    gap = (g.unmod_near.fillna(0) > 0) | (g.nterm_missing.fillna(False).astype(bool) if m == 'A' else False)
    adj = g[ADJ[m]].fillna(False).astype(bool) if ADJ[m] else pd.Series(False, index=g.index)
    poor = g.win_fit_pct < q25
    g = g.assign(gap=gap, adj=adj, poor=poor, gap_or_poor=gap | poor, any3=gap | poor | adj)
    out.append(g)
G = pd.concat(out)
S = G.groupby('case')[['gap', 'poor', 'gap_or_poor', 'adj', 'any3']].mean().mul(100).round(1)
print('all modes pooled (EM units): % with feature'); print(S.to_string())
print('case units', int(G.case.sum()))
mem = G[G.case]
print('\nper mode: papers / labs / years / SARS-CoV-2 / nanobody / mouse')
print(mem.groupby('mode').agg(units=('pdb', 'size'), papers=('paper', 'nunique'), labs=('last_author', 'nunique'), top_lab_units=('last_author', lambda x: x.value_counts().iloc[0]),
                              y_med=('year', 'median'), sars2=('sars2', 'mean'), vhh=('vhh', 'mean'), mouse=('species', lambda x: (x == 'mouse').mean())).round(2).to_string())
S.to_csv(f'{O}/deepdive_overall.csv')
