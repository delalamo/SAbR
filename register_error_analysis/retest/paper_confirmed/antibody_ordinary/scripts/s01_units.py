"""Unit frame (antibody x paper, = 'unit' of chains.csv) for the sequence-level tests.
Features are taken verbatim from seq_novelty/tables/units.csv.gz (earliest-deposited chain of the unit; feature
definitions as in seq_novelty/scripts/s06_tests.py) and literature_dean/tables/units_literature.csv.gz (CDR length
rarity vs the X-ray <=3 A reference, any X-ray <=3 A structure of the same V domain = has_hq).
Labels (aoutil.unit_labels): paper_strict, paper_incl, control (all chains paper_control), rest (deeper first-occurrence,
not paper-confirmed, >=1 non-artifact run, no amyloid), rest_win (rest released in the paper window), clean_all.
Method: EM if any case chain (for case units) / any chain (other units) is EM.
Output: tables/units_frame.csv.gz"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from aoutil import DEEP, TAB, chain_groups, unit_labels, ybin_win, ybin_full

C = chain_groups()
L = unit_labels(C)
U = pd.read_csv(f'{DEEP}/seq_novelty/tables/units.csv.gz', low_memory=False).set_index('unit')
LD = pd.read_csv(f'{DEEP}/literature_dean/tables/units_literature.csv.gz', low_memory=False).set_index('unit')
U = U.join(L, how='inner').join(LD[['rL1', 'rL2', 'rL3', 'fL1', 'fL2', 'fL3', 'rare_any', 'logf_min', 'has_hq']], how='left')
# method of the case chains for case units
cm = C[C.paper_case_incl_copies | C.paper_case_first].groupby('unit').EM.any()
rm = C[C.rest].groupby('unit').EM.any()
U['EM'] = U.method_u.eq('EM')
U.loc[cm.index.intersection(U.index), 'EM'] = cm
U.loc[rm.index.intersection(U.index).difference(cm.index), 'EM'] = rm
U['method'] = np.where(U.EM, 'EM', 'X-ray')
# ---- features (seq_novelty s06 definitions)
lg = lambda x: np.log10(1 + x)
F = pd.DataFrame(index=U.index)
for k in ['V', 'J', 'VJ', 'L123']:
    F[f'first_{k}'] = (U[f'gprior_{k}'] == 0).astype(float)
F['log_prior_V'] = lg(U.gprior_V)
F['nn_any_V'] = U.nn_any_V; F['nn_diff_V'] = U.nn_diff_V
F['prior_same_ab'] = (U.n_prior_same_ab > 0).astype(float)
F['v_id_hm'] = U.q_v_id_hm
F['low_vid85'] = (U.q_v_id_hm < 0.85).astype(float).where(U.q_v_id_hm.notna())
F['mut_total'] = U[[c for c in U.columns if c.startswith('mut_')]].sum(1)
F['humanised_sig'] = U.humanised_sig.astype(float)
for c in ['rare_canonical', 'rare_vernier', 'rare_interface']:
    F[c + '_any'] = (U[c] > 0).astype(float)
F['nonhuman_mouse'] = (~U.q_species_all.isin(['human', 'mouse'])).astype(float)
F['nanobody'] = U.vhh_u.fillna(0).astype(float)
F['rabbit'] = U.q_species_all.eq('rabbit').astype(float)
F['bovine'] = U.q_species_all.eq('cow').astype(float)
F['alpaca'] = U.q_species_all.eq('alpaca').astype(float)
F['rare_cdr_len5'] = U.rare_any.astype(float)
F['has_hq_xray'] = U.has_hq.astype(float)
F.columns = ['f_' + c for c in F.columns]
keep = ['paper', 'pdb', 'chain', 'ctype', 'method', 'EM', 'res_med', 'dep_year', 'year', 'release', 'sars2_u', 'vhh_u',
        'q_species_all', 'v_gene', 'n_prior_same_ab', 'n_chains', 'err_first_u', 'err_any_u', 'modes_u',
        'paper_strict', 'paper_incl', 'rest', 'rest_win', 'control', 'clean_all', 'any_amyloid']
X = pd.concat([U[keep], F], axis=1)
X['yb_w'] = ybin_win(X.dep_year); X['yb_f'] = ybin_full(X.dep_year)
X['sars'] = X.sars2_u.astype(float); X['vhh'] = X.vhh_u.fillna(0).astype(float)
X['nonhm'] = X.f_nonhuman_mouse
X['ct'] = np.where(X.vhh == 1, 'VHH', X.ctype)
X.reset_index().to_csv(f'{TAB}/units_frame.csv.gz', index=False)
print('units', len(X))
for k in ['paper_strict', 'paper_incl', 'control', 'rest', 'rest_win', 'clean_all']:
    d = X[X[k]]
    print(f'{k:13s} units {len(d):5d}  EM {int(d.EM.sum()):5d}  X-ray {int((~d.EM).sum()):5d}  papers {d.paper.nunique():4d}')
print('paper_strict units by paper:', X[X.paper_strict].paper.value_counts().head(6).to_dict())
print('overlap control & paper', int((X.control & X.paper_incl).sum()), 'control & rest', int((X.control & X.rest).sum()))
