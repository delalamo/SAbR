"""Publication-level trends (paper_meta lens; deeper README 4.4 and 6): citations, authors per antibody, journal/tier,
funders, open access, country, preprints and timing. Unit = paper (pkey) and method, as in paper_meta
(entries deposited from 2018; paper_analysis_table.csv). Model as the lens: err ~ x + period + log2(n antibody units)
+ median resolution, SEs clustered by last author (as PM2 verification). Window groups use year category <=2019/2020/2021+."""
import sys; sys.path.insert(0, '.')
from ww_common import *
from scipy.stats import chi2_contingency
E = entry_flags()
P = pd.read_csv(f'{D}/paper_meta/paper_analysis_table.csv', low_memory=False)
km = P[['pkey', 'method', 'pdbs']].assign(pdb=P.pdbs.str.split(';')).explode('pdb')
km['uk'] = km.pkey + '|' + km.method
P['uk'] = P.pkey + '|' + P.method
P = P.merge(unit_flags(E, km[['pdb', 'uk']], 'uk'), on='uk', how='left')
for c in ['case', 'case_i', 'ctrl', 'gen', 'rest', 'rest_w', 'win_all', 'clean']:
    P[c] = P[c].fillna(False).astype(bool)
P = add_yc(P, 'dep_year'); P['lab'] = P.last_author.fillna(P.pkey)
P = P[P.amyloid_fibril != True]
print(P.groupby('method')[['case', 'case_i', 'ctrl', 'rest_w', 'rest', 'clean']].sum())
print('paper-case papers missing from paper_meta (deposited before 2018):', sorted(set(E[E.case].paper) - set(P[P.case].pkey)))
for c in ['pre_to_pub', 'dep_to_pub', 'rec_to_acc', 'acc_to_pub', 'hold_days_median', 'dep_minus_received']:
    P[c + '_100d'] = P[c] / 100
P['tier_top'] = P.tier.isin(['CNS', 'top']).astype(float) if P.tier.dtype == object else np.nan
P['oa'] = P.open_access.astype(float)
P['n_auth_l2'] = np.log2(P.n_authors)
B = '{YC} + l2_units + res_median'
TESTS = [('l_cites_per_year', 'citations per year (log1p) OR 0.68 (0.54-0.86), between-lab only; within-lab 0.88 [paper_meta, PM2 verify]', 'pub'),
         ('cite_pct', 'within-year citation percentile 0.49 vs 0.56 [paper_meta]', 'pub'),
         ('l_authors', 'authors per antibody: log2 authors adj for units, OR 0.70 per doubling q=0.052; not in 2018-20 [paper_meta]', 'pub'),
         ('j_CNS', 'CNS journals: null [paper_meta]', 'pub'), ('j_sister', 'CNS sister journals: null', 'pub'),
         ('j_natcommun', 'Nat Commun: null', 'pub'), ('j_open_megajournal', 'mega journals: null', 'pub'),
         ('oa', 'open access: null [paper_meta]', 'pub'),
         ('fund_NIH', 'NIH funding: null', 'pub'), ('fund_HHMI', 'HHMI: null', 'pub'), ('fund_Gates', 'Gates: null', 'pub'),
         ('fund_Wellcome', 'Wellcome: null', 'pub'), ('fund_China_funders', 'Chinese funders: null', 'pub'), ('fund_EU_ERC', 'EU/ERC: null', 'pub'),
         ('fund_UKRI_MRC', 'MRC/UKRI: null', 'pub'), ('fund_DFG', 'DFG 5/110 vs 13/991, OR 5.0, q=0.052, likely chance', 'pub'),
         ('aff_industry', 'industry affiliation: null', 'pub'),
         ('last_China', 'last-author region: USA 11.0%, China 10.0%, Europe 9.9%, other 3.2%; p=0.24 [paper_meta]', 'pub'),
         ('last_USA', 'see last_China', 'pub'), ('last_Europe', 'see last_China', 'pub'),
         ('has_preprint', 'preprint exists 32.7% vs 30.4%, p=0.79', 'pub'),
         ('pre_to_pub_100d', 'preprint->publication 211 vs 179 d, p=0.10', 'pub'),
         ('preprint_lt90d_after_dep', 'preprint within 90 d of deposition 83% vs 62.5%, OR 2.6 p=0.04 q=0.41', 'pub'),
         ('dep_to_pub_100d', 'first deposition->publication 178 vs 194 d, p=0.55', 'pub'),
         ('rec_to_acc_100d', 'review time 144 vs 149 d, p=0.86', 'pub'), ('fast_review_lt60d', 'review <60 d 12.7% vs 12.4%', 'pub'),
         ('acc_to_pub_100d', 'acceptance->publication 24 vs 26 d', 'pub'),
         ('dep_after_submission', 'first structure deposited >7 d after submission 27% vs 32%, p=0.29', 'pub'),
         ('deplast_after_submission', 'structure added after submission 32% vs 35%, p=0.36', 'pub'),
         ('released_before_pub', 'released before paper 30% vs 30%', 'pub'),
         ('hold_days_median_100d', 'hold time 207 vs 231 d, p=0.48', 'all'),
         ('pt_letter_brief', 'letter/brief 6.4% vs 4.4%, p=0.22', 'pub'),
         ('tool_paper', 'tool paper OR 0.90 after size adjustment', 'all'),
         ('published', 'published OR 1.46 p=0.19', 'all')]
for meth in ['EM', 'X-ray']:
    for x, ref, sub in TESTS:
        d = P[P.method == meth]
        if sub == 'pub': d = d[d.published == True]
        d = d.copy()
        if d[x].dtype == bool or d[x].dtype == object: d[x] = d[x].astype(float)
        record(f'publication: {x}', 'paper', meth, x, B + ', lab-clustered', run_groups(d, x, B, 'lab'), ref if meth == 'EM' else ('X-ray: ' + ref))
# citations: one paper per lab robustness and within-lab note
d = P[(P.method == 'EM') & (P.published == True)].copy()
record('publication: l_cites_per_year (unclustered)', 'paper', 'EM', 'l_cites_per_year', B, run_groups(d, 'l_cites_per_year', B, None), 'OR 0.683 p=0.0013 unclustered [PM2]')
record('publication: l_cites_per_year, crude', 'paper', 'EM', 'l_cites_per_year', '{YC}', run_groups(d, 'l_cites_per_year', '{YC}', 'lab'), 'crude 0.85 p=0.12; period alone 0.73 [PM2]')

record('publication: l_cites_per_year (+last_China)', 'paper', 'EM', 'l_cites_per_year', B + ' + last_China', run_groups(d, 'l_cites_per_year', B + ' + last_China', 'lab'), 'composition check')
record('publication: l_cites_per_year (non-China last author)', 'paper', 'EM', 'l_cites_per_year', B, run_groups(d[d.last_China == 0], 'l_cites_per_year', B, 'lab'), 'composition check')
record('publication: l_cites_per_year (+sars2)', 'paper', 'EM', 'l_cites_per_year', B + ' + sars2', run_groups(d.assign(sars2=d.sars2.astype(float)), 'l_cites_per_year', B + ' + sars2', 'lab'), 'composition check')
record('publication: l_authors (+last_China)', 'paper', 'EM', 'l_authors', B + ' + last_China', run_groups(d, 'l_authors', B + ' + last_China', 'lab'), 'composition check')
record('publication: fund_NIH (+last_China)', 'paper', 'EM', 'fund_NIH', B + ' + last_China', run_groups(d.assign(fund_NIH=d.fund_NIH.astype(float)), 'fund_NIH', B + ' + last_China', 'lab'), 'composition check')
# within-lab: conditional logit on labs with both a paper-case and a window-control paper is not feasible (count printed)
w = d[d.case | (d.ctrl & ~d.gen & ~d.case_i)]
print('labs with both a paper-case and a window-control EM paper:', int((w.groupby('lab').case.nunique() > 1).sum()))
# case-only contrast: China-led share, paper case vs rest_w (Fisher)
from scipy.stats import fisher_exact
a = d[d.case].last_China; b = d[d.rest_w].last_China
print(f'China-led share, EM papers: paper-strict {int(a.sum())}/{a.notna().sum()} vs rest_w {int(b.sum())}/{b.notna().sum()}, Fisher p={fisher_exact([[a.sum(), a.notna().sum()-a.sum()], [b.sum(), b.notna().sum()-b.sum()]])[1]:.3f}')
a = d[d.case_i & ~d.case].last_China
print(f'  copies-only papers: {int(a.sum())}/{a.notna().sum()}')

# journal heterogeneity (journal groups with >=10 window papers), paper-case vs window controls, EM+X-ray
print('\n== journal groups: paper-case papers vs window controls (published, both methods)')
d = P[(P.published == True) & (P.case | (P.ctrl & ~P.gen & ~P.case_i))].copy()
d['jg'] = d.journal_n.fillna('other')
top = d.jg.value_counts(); keep = top[top >= 10].index
d['jg2'] = np.where(d.jg.isin(keep), d.jg, 'other')
ct = pd.crosstab(d.jg2, d.case); chi = chi2_contingency(ct)
rng = np.random.default_rng(0); obs = chi[0]; perm = []
for i in range(2000):
    perm.append(chi2_contingency(pd.crosstab(d.jg2, rng.permutation(d.case.values)))[0])
pperm = float(np.mean(np.array(perm) >= obs))
print(ct.assign(rate=lambda t: (100 * t[True] / t.sum(axis=1)).round(1)).sort_values(True, ascending=False).to_string())
print(f'chi2={obs:.1f}, df={chi[2]}, permutation p={pperm:.3f}  (full set: heterogeneity across 19 journal groups p=0.36)')
ROWS.append(dict(trend='publication: journal group heterogeneity', level='paper', method='EM+X-ray', x='journal group', group='case', p=pperm, note=f'chi2 {obs:.1f} df {chi[2]} permutation; full set p=0.36'))
ct.to_csv(f'{T}/w04_journal_groups.csv')
save_rows(f'{T}/w04_paper_meta_tests.csv')
P[['pkey', 'method', 'dep_year', 'case', 'case_i', 'ctrl', 'rest', 'rest_w', 'gen', 'clean', 'journal_n', 'tier', 'last_country', 'l_cites_per_year', 'l_authors',
   'has_preprint', 'rec_to_acc', 'dep_to_pub', 'hold_days_median']].to_csv(f'{T}/w04_paper_frame.csv.gz', index=False)
