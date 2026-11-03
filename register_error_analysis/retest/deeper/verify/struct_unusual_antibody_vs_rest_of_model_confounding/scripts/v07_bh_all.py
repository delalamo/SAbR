"""Benjamini-Hochberg q over every test run in this verification (population, within-paper, sample re-analyses)."""
import pandas as pd
from statsmodels.stats.multitest import multipletests
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_antibody_vs_rest_of_model_confounding/tables'
parts = []
for fn in ['population_tests', 'population_within_paper', 'sample_reanalysis', 'sample_restfix']:
    x = pd.read_csv(f'{OUT}/{fn}.csv'); x.insert(0, 'table', fn); parts.append(x)
A = pd.concat(parts, ignore_index=True)
A['q_BH_all'] = multipletests(A.p, method='fdr_bh')[1]
A.to_csv(f'{OUT}/all_tests_bh.csv', index=False)
key = A[A.analysis.fillna('').str.contains('error vs all clean|within paper|all covariates: V chains without|all covariates: V excluding within 10|other chains % clash|other chains % allowed|V minus rest Q', regex=True)
        | (A.table == 'sample_restfix') & (A.cluster == 'paper') & A.y.isin(['gap_rama_allowed', 'gap_clash', 'rest_clash'])]
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print(len(A), 'tests'); print(key[['table', 'analysis', 'subset', 'y', 'n_case', 'n_ctrl', 'effect', 'p', 'q_BH_all']].round(5).to_string())
