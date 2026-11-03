# Verify the SARS-CoV-2 sub-claims: median other labs, "ordinary" error rate (adjusted), earliest-25 all clean
import warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
F = pd.read_csv(f'{INV}/tables/focal_entries_antigen_features.csv.gz')
F['sars2'] = F.sars2.astype(bool)
EM = F[F.method == 'EM'].copy()
S = EM[EM.sars2]
print('SARS2 EM analysis entries', len(S), 'errors', S.y.sum(), 'rate %.3f' % S.y.mean(), '| median other labs acc 12m %.1f (all), %.1f (clean), %.1f (error)' % (
    S.labs_acc_12m.median(), S[S.y == 0].labs_acc_12m.median(), S[S.y == 1].labs_acc_12m.median()))
print('non-SARS2 EM rate %.3f (%d/%d)' % (EM[~EM.sars2].y.mean(), EM[~EM.sars2].y.sum(), (~EM.sars2).sum()))
def fit(D, term, extra):
    mod = smf.glm(f'y ~ {term} + {extra}', D, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.paper)[0]})
    k = [x for x in mod.params.index if x.startswith(term)][0]; ci = mod.conf_int().loc[k]
    return 'OR %.2f (%.2f-%.2f) p=%.3g n=%d err=%d' % (np.exp(mod.params[k]), np.exp(ci[0]), np.exp(ci[1]), mod.pvalues[k], len(D), D.y.sum())
EM['s2'] = EM.sars2.astype(int)
print('SARS2 vs other EM, unadjusted   :', fit(EM, 's2', '1'))
print('SARS2 vs other EM, year+res     :', fit(EM, 's2', 'C(ycat) + res'))
for per in ['2020-21', '2022+']:
    D = EM[EM.period == per]
    print(f'SARS2 vs other EM {per}, year+res:', fit(D, 's2', 'C(ycat) + res'), '| rates %.3f vs %.3f' % (D[D.sars2].y.mean(), D[~D.sars2].y.mean()))
# against other contested (>=1 other lab) non-SARS2 antigens only, same period
D = EM[(EM.period == '2020-21') & (EM.sars2 | (EM.labs_acc_12m > 0))]
print('2020-21 SARS2 vs contested non-SARS2:', fit(D, 's2', 'C(ycat) + res'), '| rates %.3f vs %.3f' % (D[D.sars2].y.mean(), D[~D.sars2].y.mean()))
# spike family test as in investigator
EM['fam1'] = EM.tgt_fam.fillna('none').str.split('|').str[0]
EM['spikefam'] = (EM.fam1 == 'g30_4905_30').astype(int)
print('spike 30% family (g30_4905_30) vs other EM, year+res:', fit(EM, 'spikefam', 'C(ycat) + res'), '| rates %.3f vs %.3f' % (EM[EM.spikefam == 1].y.mean(), EM[EM.spikefam == 0].y.mean()))
print('  spike family entries that are not flagged sars2:', int(((EM.spikefam == 1) & ~EM.sars2).sum()), '| sars2 entries outside family:', int(((EM.spikefam == 0) & EM.sars2).sum()))

# earliest SARS2 antibody cryo-EM entries: use ALL entries.csv (before exclusions) so that excluded ones are visible
e = pd.read_csv(f'{SP}/deeper/entries.csv'); e['dep'] = pd.to_datetime(e.deposit)
s = e[(e.method == 'EM') & e.sars2.astype(bool)].sort_values(['dep', 'pdb'])
print('\nfirst 30 SARS2 EM entries in entries.csv (all, incl. excluded):')
print(s.head(30)[['pdb', 'deposit', 'res', 'last_author', 'err_first', 'err_any', 'modes']].to_string())
k = 0
for i, (_, r) in enumerate(s.iterrows()):
    if r.err_first: k = i; break
print('index of first err_first SARS2 EM entry:', k, '-> deposited', s.iloc[k].deposit, s.iloc[k].pdb, s.iloc[k].last_author)
print('SARS2 EM entries deposited before 2020-09-01:', (s.dep < '2020-09-01').sum(), 'err_first', s[s.dep < '2020-09-01'].err_first.sum(), 'err_any', s[s.dep < '2020-09-01'].err_any.sum(),
      'papers', s[s.dep < '2020-09-01'].doi.nunique())
first25 = s.head(25)
print('first 25: deposit range', first25.deposit.min(), first25.deposit.max(), '| err_first', first25.err_first.sum(), '| err_any', first25.err_any.sum(), '| distinct papers', first25.doi.nunique(), '| labs', first25.last_author.nunique())
# chance of 25 clean entries given period rate, treating papers as the independent unit
r2020 = EM[(EM.period == '2020-21')].y.mean(); rpap = EM[EM.period == '2020-21'].groupby('paper').y.max().mean()
npap = first25.doi.nunique()
print('P(all clean | 2020-21 EM entry rate %.3f, 25 independent entries) = %.3f' % (r2020, (1 - r2020) ** 25))
print('P(all clean | 2020-21 EM paper rate %.3f, %d independent papers) = %.3f' % (rpap, npap, (1 - rpap) ** npap))
# what about the next window
for a, b in [('2020-01-01', '2020-09-01'), ('2020-09-01', '2021-01-01'), ('2021-01-01', '2021-07-01'), ('2021-07-01', '2022-01-01'), ('2022-01-01', '2023-01-01'), ('2023-01-01', '2027-01-01')]:
    w = s[(s.dep >= a) & (s.dep < b)]; wn = e[(e.method == 'EM') & ~e.sars2.astype(bool) & (e.dep >= a) & (e.dep < b)]
    print(f'{a}..{b}: SARS2 EM entries {len(w)}, err_first {w.err_first.sum()} ({w.err_first.mean():.3f}) | non-SARS2 EM {len(wn)} err_first {wn.err_first.sum()} ({wn.err_first.mean():.3f})')
