"""Combine all DE2 verification tests and apply one Benjamini-Hochberg correction across every p-value
(Spearman descriptives and leave-one-out ranges excluded)."""
import pandas as pd, numpy as np
from statsmodels.stats.multitest import multipletests
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE2_framework_distortion_confounding'
parts = []
for f in ['tests_all.csv', 'tests_extra.csv', 'tests_entry_level.csv', 'tests_within_paper_ct.csv']:
    t = pd.read_csv(f'{OUT}/{f}'); t['source'] = f; parts.append(t)
T = pd.concat(parts, ignore_index=True)
T = T.drop(columns=[c for c in ['q', 'q_within_file'] if c in T])
ok = T.p.notna() & ~T.test.str.contains('Spearman|confirmed vs not|leave-one', regex=True)
T.loc[ok, 'q_BH_all'] = multipletests(T.loc[ok, 'p'], method='fdr_bh')[1]
T.to_csv(f'{OUT}/all_tests_BH.csv', index=False)
print(len(T), ok.sum())
