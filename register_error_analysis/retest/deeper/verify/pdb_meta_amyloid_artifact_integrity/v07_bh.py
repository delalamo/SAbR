"""BH q across the verification tests run in v02/v04 (p-values copied from their logs)."""
import pandas as pd
from statsmodels.stats.multitest import multipletests
T = pd.DataFrame([
 ('antibody-alone, EM entries, with fibrils (Fisher)', 1e-11),
 ('antibody-alone, EM entries, with fibrils (adjusted)', 1.6e-07),
 ('antibody-alone, EM entries, without fibrils (Fisher)', 1.0),
 ('antibody-alone, EM entries, without fibrils (adjusted)', 0.89),
 ('lambda L1=8 vs 9, chains all methods, with fibrils', 1.67e-15),
 ('lambda L1=8 vs 9, chains all methods, without fibrils', 0.0195),
 ('lambda L1=8 vs 9, units all methods, with fibrils', 0.0020),
 ('lambda L1=8 vs 9, units all methods, without fibrils', 0.0419),
 ('lambda L1=8 vs 9, units EM, without fibrils', 0.107),
 ('lambda L1=8 vs 9, EM chains adjusted, with fibrils', 0.00237),
 ('lambda L1=8 vs 9, EM chains adjusted, without fibrils', 0.277),
 ('lambda L1=8 vs 9, EM units adjusted, without fibrils', 0.125),
], columns=['test', 'p'])
T['q_BH'] = multipletests(T.p, method='fdr_bh')[1]
T.to_csv('verification_tests_bh.csv', index=False)
print(T.to_string(index=False))
