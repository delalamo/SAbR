"""Composition checks: (a) no-funding-record excess vs China-led last author; (b) lab experience (antigen_competition
measure) in non-China-led papers; (c) what drives the enzyme incl.-copies OR; (d) early SARS-CoV-2 rank table check."""
import numpy as np, pandas as pd, warnings
from scipy.stats import fisher_exact
import statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D = '/home/user/SAbR/register_error_analysis/retest/deeper'
WW = '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/who_where_when/tables'
CH = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False).merge(pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz'), on=['pdb', 'chain'])
R = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv').merge(pd.read_csv(f'{D}/synthesis/artifact_ledger_runs.csv')[['pdb', 'chain', 'imgt0', 'imgt1', 'flag']], on=['pdb', 'chain', 'imgt0', 'imgt1'], how='left')
G = set(map(tuple, R[~R.flag.isin({'amyloid_fibril', 'indel_gap_artifact_confirmed', 'DE_chain_end_artifact', 'near_copy', 'not_a_register_shift'})][['pdb', 'chain']].values))
CH['gen'] = [k in G for k in zip(CH.pdb, CH.chain)]
E = CH.groupby('pdb').agg(case=('paper_case_strict', 'any'), case_i=('paper_case_incl_copies', 'any'), ctrl=('paper_control', 'all'), win=('in_paper_window', 'all'), gen=('gen', 'any')).reset_index()
E['rest_w'] = E.gen & ~E.case_i & E.win; E['ctl'] = E.ctrl & ~E.gen & ~E.case_i
P = pd.read_csv(f'{D}/paper_meta/paper_analysis_table.csv', low_memory=False)
china = P[P.method == 'EM'].assign(pdb=P.pdbs.str.split(';')).explode('pdb')[['pdb', 'last_China']].drop_duplicates('pdb')
F = pd.read_csv(f'{D}/pdb_meta/tables/em_entry_analysis_frame_with_amyloid.csv.gz', low_memory=False).merge(E, on='pdb').merge(china, on='pdb', how='left')
F['paper'] = F.doi.fillna(F.pdb); F['yc'] = np.where(F.dep_year <= 2019, 'a', np.where(F.dep_year == 2020, 'b', 'c'))
print('(a) no funding record, EM entries, by China-led')
for g in ['case', 'rest_w']:
    d = F[F[g] | F.ctl].dropna(subset=['last_China']).copy(); d['y'] = d[g].astype(float)
    print(f'  {g}:', d.groupby(['last_China', 'y']).fund_none.agg(['sum', 'count']).to_dict('index'))
    for rhs in ['C(yc) + res', 'C(yc) + res + last_China']:
        m = smf.glm(f'y ~ fund_none + {rhs}', d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.paper)[0]})
        print(f'    {rhs:28s} OR {np.exp(m.params.fund_none):.2f} ({np.exp(m.conf_int().loc["fund_none"][0]):.2f}-{np.exp(m.conf_int().loc["fund_none"][1]):.2f}) p={m.pvalues.fund_none:.3g}  n={int(d.y.sum())}/{int((d.y==0).sum())}')
    dn = d[d.last_China == 0]
    t = pd.crosstab(dn.y, dn.fund_none).reindex(index=[1.0, 0.0], columns=[1, 0], fill_value=0).values
    print(f'    non-China only: {t[0,0]}/{t[0].sum()} vs {t[1,0]}/{t[1].sum()}  Fisher OR {fisher_exact(t)[0]:.2f} p={fisher_exact(t)[1]:.3g}')
print('  control China share of no-funding entries: %.2f' % F[F.ctl & (F.fund_none == 1)].last_China.mean())

print('\n(b) lab experience (antigen_competition lab_prior_ab), EM entries, by China-led')
A = pd.read_csv(f'{D}/verify/antigen_competition_no_competitive_pressure_confounding/entries_frame.csv.gz', low_memory=False).drop(columns=['clean'])
A = A.merge(E, on='pdb').merge(china, on='pdb', how='left').merge(pd.read_csv(f'{D}/antigen_competition/tables/lab_experience.csv'), on='pdb', how='left')
A = A[(A.method == 'EM') & A.ab_status2.isin(['bound', 'fiducial-only'])].copy()
A['lexp'] = np.log2(1 + A.lab_prior_ab); A['yc'] = np.where(A.dyear <= 2019, 'a', np.where(A.dyear == 2020, 'b', 'c'))
for sub, dsub in [('non-China', A[A.last_China == 0]), ('China', A[A.last_China == 1]), ('all, +China', A.dropna(subset=['last_China']))]:
    for g in ['case', 'rest_w']:
        d = dsub[dsub[g] | dsub.ctl].dropna(subset=['lexp', 'res']).copy(); d['y'] = d[g].astype(float)
        rhs = 'C(yc) + res' + (' + last_China' if sub.startswith('all') else '')
        try:
            m = smf.glm(f'y ~ lexp + {rhs}', d, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.paper)[0]})
            print(f'  {sub:12s} {g:7s} {int(d.y.sum())}/{int((d.y==0).sum())}: OR {np.exp(m.params.lexp):.2f} ({np.exp(m.conf_int().loc["lexp"][0]):.2f}-{np.exp(m.conf_int().loc["lexp"][1]):.2f}) p={m.pvalues.lexp:.3g}')
        except Exception as e:
            print('  ', sub, g, 'fail', str(e)[:50])
print('  median lab_prior_ab: case %.0f | rest_w %.0f | ctl %.0f' % (A[A.case].lab_prior_ab.median(), A[A.rest_w].lab_prior_ab.median(), A[A.ctl].lab_prior_ab.median()))

print('\n(c) enzyme, EM entries: case / case_i / ctl')
A['enz'] = A.cls2.eq('enzyme')
for g in ['case', 'case_i', 'ctl']:
    d = A[A[g]]
    print(f'  {g}: {int(d.enz.sum())}/{len(d)}', d[d.enz][['pdb', 'paper']].values.tolist() if g != 'ctl' else '')
w = pd.read_csv(f'{WW}/w01_antigen_competition_tests.csv')
print(w[(w.x == 'enz') & (w.group.isin(['case', 'case_i']))][['trend', 'level', 'method', 'group', 'n_case', 'n_ctrl', 'OR', 'lo', 'hi', 'p']].to_string())

print('\n(d) early SARS-CoV-2 race: investigator rank bins')
print(pd.read_csv(f'{WW}/w02_sars2_rank_bins.csv').head(8).to_string())
