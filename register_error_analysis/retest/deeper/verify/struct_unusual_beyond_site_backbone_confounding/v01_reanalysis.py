"""Adversarial re-analysis (confounding lens) of the struct_unusual finding 'beyond_site_backbone':
error chains have more Ramachandran 'allowed', clash and rotamer-outlier residues outside the error site.

Uses the investigator's per-unit table (unit_metrics.csv; metrics computed from wwPDB validation XML) and re-fits
the case-control models with:
  - cluster-robust SEs by entry (original), by paper, by last author;
  - paper-level aggregation (one case value and one control value per paper);
  - extra covariates: SARS-CoV-2, continuous deposition date, resolution band (categorical), refinement software flags,
    starting-model type and fitting protocol, antibody role (auxiliary reagent), unbuilt residues / chain gaps,
    side-chain truncation, local Q-score, the non-antibody rest of the same entry (entry-level modelling quality);
  - restriction to 2020-2022, to non-SARS-CoV-2, to 'antibody of interest';
Output: tables/v01_models.csv
"""
import warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SU = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/struct_unusual_beyond_site_backbone_confounding'


def load():
    U = pd.read_csv(f'{SU}/unit_metrics.csv', dtype={'chain': str, 'partner': str})
    s = pd.read_csv(f'{SU}/sample.csv', dtype={'chain': str})
    d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str})
    meta = d[['pdb', 'chain', 'last_author', 'deposit', 'sars2', 'role', 'aux_reagent', 'irm', 'fit', 'species',
              'sw_ISOLDE', 'sw_Rosetta', 'sw_Coot', 'sw_Phenix', 'sw_REFMAC', 'sw_BUSTER', 'tier', 'novel_target',
              'v_id', 'L1_atyp', 'L2_atyp', 'L3_atyp', 'competitors', 'lq_z']]
    U = U.merge(meta, on=['pdb', 'chain'], how='left')
    U['case'] = (U.role_cc == 'case').astype(int)
    U['sars2'] = U.sars2.astype(float)
    U['paper_c'] = U.paper.fillna(U.pdb)
    U['author_c'] = U.last_author.fillna(U.paper_c)
    dep = pd.to_datetime(U.deposit)
    U['dep_y'] = dep.dt.year + dep.dt.dayofyear / 366.0
    U['rband'] = pd.cut(U.res, [0, 3.25, 3.5, 3.75, 4.0, 4.4, 9]).astype(str)
    U['aux'] = (U.role == 'Auxiliary').astype(int)
    U['irm_c'] = U.irm.fillna('none recorded')
    U['fit_c'] = U.fit.fillna('not recorded')
    E = pd.read_csv(f'{SU}/entry_vrpt.csv')
    U = U.merge(E[['pdb', 'rest_rama_allowed', 'rest_clash', 'rest_rota_out', 'rest_q', 'e_clashscore',
                   'rest_n']], on='pdb', how='left')
    U['any_gap'] = (U.n_gaps_int > 0).astype(int)
    U['log_unbuilt'] = np.log1p(U.unbuilt_int.fillna(0))
    U['first_apos_late'] = (U.first_apos.fillna(1) > 3).astype(int)
    return U


BASE = 'C(cclass) + C(ybin) + res'


def fit(df, y, rhs, group, label, rows, metric_label=None):
    cols = [y, 'case', group]
    for tok in rhs.replace('C(', ' ').replace(')', ' ').replace('+', ' ').replace('*', ' ').split():
        if tok in df.columns:
            cols.append(tok)
    x = df.dropna(subset=list(dict.fromkeys(cols))).copy()
    if x.case.sum() < 5 or (1 - x.case).sum() < 5:
        return
    g = pd.factorize(x[group])[0]
    m = smf.ols(f'{y} ~ case + {rhs}', x).fit(cov_type='cluster', cov_kwds={'groups': g})
    lo, hi = m.conf_int().loc['case']
    rows.append(dict(metric=metric_label or y, model=label, cluster=group, n_case=int(x.case.sum()),
                     n_ctrl=int((1 - x.case).sum()), n_clusters=len(set(g)), case_mean=x[x.case == 1][y].mean(),
                     ctrl_mean=x[x.case == 0][y].mean(), effect=m.params['case'], ci_lo=lo, ci_hi=hi,
                     p=m.pvalues['case']))


def main():
    U = load()
    rows = []
    Ys = ['rama_allowed_out', 'clash_out', 'rota_out_out']
    for y in Ys:
        fit(U, y, BASE, 'pdb', 'base (reproduction)', rows)
        fit(U, y, BASE, 'paper_c', 'base, cluster=paper', rows)
        fit(U, y, BASE, 'author_c', 'base, cluster=last author', rows)
        fit(U, y, 'C(cclass) + dep_y + I(dep_y**2) + C(rband)', 'author_c', 'continuous year (quadratic) + resolution band', rows)
        fit(U, y, BASE + ' + sars2', 'author_c', '+ SARS-CoV-2', rows)
        fit(U, y, BASE + ' + sw_Phenix + sw_Coot + sw_ISOLDE + sw_Rosetta + sw_REFMAC + sw_BUSTER', 'author_c',
            '+ refinement software flags', rows)
        fit(U, y, BASE + ' + C(irm_c) + C(fit_c)', 'author_c', '+ starting model + fitting protocol', rows)
        fit(U, y, BASE + ' + aux + C(tier)', 'author_c', '+ auxiliary-reagent role + journal tier', rows)
        fit(U, y, BASE + ' + n_gaps_int + log_unbuilt + first_apos_late + sc_trunc_out', 'author_c',
            '+ unbuilt residues/gaps/N-term truncation + side-chain truncation', rows)
        fit(U, y, BASE + ' + q_out', 'author_c', '+ local Q outside site', rows)
        fit(U, y, BASE + ' + rest_rama_allowed + rest_clash + rest_rota_out', 'author_c',
            '+ same-entry non-V geometry (entry modelling quality)', rows)
        fit(U, y, BASE + ' + sars2 + sw_Phenix + sw_Coot + sw_ISOLDE + sw_Rosetta + sw_REFMAC + C(irm_c) + C(fit_c)'
            ' + aux + n_gaps_int + log_unbuilt + first_apos_late + sc_trunc_out + q_out + rest_rama_allowed + rest_clash',
            'author_c', 'kitchen sink (all of the above)', rows)
        fit(U[U.year.between(2020, 2022)], y, BASE, 'author_c', '2020-2022 only', rows)
        fit(U[U.sars2 == 0], y, BASE, 'author_c', 'non-SARS-CoV-2 only', rows)
        fit(U[U.role == 'Antibody of interest'], y, BASE, 'author_c', "'antibody of interest' only", rows)
        fit(U[U.aux == 1], y, BASE, 'author_c', 'auxiliary reagents only', rows)
        fit(U[U.n_gaps_int.fillna(0) == 0], y, BASE, 'author_c', 'chains with no internal gap only', rows)
        # Phenix-only (the dominant refinement program; Ramachandran restraints available)
        fit(U[U.sw_Phenix == 1], y, BASE, 'author_c', 'Phenix-refined only', rows)
        fit(U[U.sw_Phenix == 0], y, BASE, 'author_c', 'Phenix not recorded only', rows)
    # ---- paper-level aggregation: mean per (paper, role) -> one row per paper per role
    for y in Ys:
        P = U.dropna(subset=[y]).groupby(['paper_c', 'case']).agg(
            y=(y, 'mean'), res=('res', 'mean'), dep_y=('dep_y', 'mean'), author_c=('author_c', 'first'),
            frac_VL=('cclass', lambda c: np.mean(np.isin(c, ['K', 'L']))),
            frac_VHH=('cclass', lambda c: np.mean(c == 'VHH')), n=('unit', 'size')).reset_index()
        x = P
        g = pd.factorize(x.author_c)[0]
        m = smf.ols('y ~ case + dep_y + I(dep_y**2) + res + frac_VL + frac_VHH', x).fit(cov_type='cluster',
                                                                                     cov_kwds={'groups': g})
        lo, hi = m.conf_int().loc['case']
        rows.append(dict(metric=y, model='paper-level means (1 row per paper x role)', cluster='author_c',
                         n_case=int(x.case.sum()), n_ctrl=int((1 - x.case).sum()), n_clusters=len(set(g)),
                         case_mean=x[x.case == 1].y.mean(), ctrl_mean=x[x.case == 0].y.mean(),
                         effect=m.params['case'], ci_lo=lo, ci_hi=hi, p=m.pvalues['case']))
        # author-level aggregation
        A = U.dropna(subset=[y]).groupby(['author_c', 'case']).agg(
            y=(y, 'mean'), res=('res', 'mean'), dep_y=('dep_y', 'mean'),
            frac_VL=('cclass', lambda c: np.mean(np.isin(c, ['K', 'L']))),
            frac_VHH=('cclass', lambda c: np.mean(c == 'VHH'))).reset_index()
        m = smf.ols('y ~ case + dep_y + I(dep_y**2) + res + frac_VL + frac_VHH', A).fit(cov_type='HC3')
        lo, hi = m.conf_int().loc['case']
        rows.append(dict(metric=y, model='last-author-level means (1 row per lab x role)', cluster='none (HC3)',
                         n_case=int(A.case.sum()), n_ctrl=int((1 - A.case).sum()), n_clusters=len(A),
                         case_mean=A[A.case == 1].y.mean(), ctrl_mean=A[A.case == 0].y.mean(),
                         effect=m.params['case'], ci_lo=lo, ci_hi=hi, p=m.pvalues['case']))
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/tables/v01_models.csv', index=False)
    pd.set_option('display.width', 250)
    print(T[['metric', 'model', 'cluster', 'n_case', 'n_ctrl', 'n_clusters', 'case_mean', 'ctrl_mean', 'effect',
             'ci_lo', 'ci_hi', 'p']].round(4).to_string())
    # descriptive: covariate balance
    bal = U.groupby('case')[['res', 'dep_y', 'sars2', 'sw_Phenix', 'sw_Coot', 'sw_ISOLDE', 'sw_Rosetta', 'sw_REFMAC',
                             'aux', 'n_gaps_int', 'unbuilt_int', 'sc_trunc_out', 'q_out', 'rest_rama_allowed',
                             'rest_clash', 'first_apos_late']].mean().T
    bal.to_csv(f'{OUT}/tables/v01_balance.csv')
    print(bal.round(3))


if __name__ == '__main__':
    main()
