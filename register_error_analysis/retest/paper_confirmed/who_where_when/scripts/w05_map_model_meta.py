"""Microscope, processing, map-intrinsic local resolution at the site, entry size, deposition completeness (pdb_meta lens),
and recorded software / starting model / fitting protocol (deepdive, junction), on the paper-confirmed set.
Reuses pdb_meta/tables/em_entry_analysis_frame_with_amyloid.csv.gz (one row per EM entry), pdb_meta/tables/local_resolution_sites.csv,
and the per-chain software/starting-model fields in scratchpad/deeper/chains.csv.gz."""
import sys; sys.path.insert(0, '.')
from ww_common import *
from scipy.stats import binomtest, wilcoxon, mannwhitneyu, fisher_exact
C = chains(); E = entry_flags(C)
F = pd.read_csv(f'{D}/pdb_meta/tables/em_entry_analysis_frame_with_amyloid.csv.gz', low_memory=False)
F = F.merge(E[['pdb', 'case', 'case_f', 'case_i', 'ctrl', 'gen', 'rest', 'rest_w', 'win_all', 'clean', 'win']], on='pdb', how='inner')
F = add_yc(F, 'dep_year'); F['paper'] = F.doi.fillna(F.pdb)
print('EM entries:', F[['case', 'case_i', 'ctrl', 'rest_w', 'rest', 'clean']].sum().to_dict())
B = '{YC} + res + l2nV'
REF = {'krios': 'Krios 67.4% vs 78.1%, OR 0.53 (0.34-0.83); <=2019 0.19, 2020-21 0.29, 2022+ 0.79 [pdb_meta]',
       'kv300': '300 kV 79% vs 86%, OR 0.59 p=0.05 [pdb_meta]',
       'pix_gt_1': 'pixel size 1.07 vs 1.02 A, OR 1.31 [pdb_meta] (here: pixel >1.0 A)',
       'any_local_or_focused': 'local/focused refinement 13.4% vs 13.7%: null', 'kw_deepemhancer': 'DeepEMhancer: null', 'kw_densmod': 'density modification: null',
       'sw_cryosparc_final': 'cryoSPARC final 74% vs 75%: null', 'sw_relion_final': 'RELION final: null', 'log10_particles': 'particle number: null',
       'startup_pdb': 'EM start-up model: null', 'det_K2': 'detector K2: null', 'has_half_maps': 'half maps 62% vs 76%, OR 0.58 (n.s. once mandatory)',
       'log2_tot_mw': 'total mass OR 0.69 per doubling p=0.005 [pdb_meta]', 'ab_small_part_lt15pct': 'antibody <15% of mass 18% vs 25%, OR 0.57',
       'kw_membrane': 'membrane keywords 19% vs 32%, OR 0.47', 'big_gt500kDa': '>500 kDa 14.0% vs 12.7%: null',
       'fund_none': 'no funding record 13.5% vs 4.6%, OR 2.4 (2020-21 only; not paper-weighted)', 'emdb_n_fields': 'EMDB fields 371 vs 412 p=0.014 (per 10 fields here)',
       'vem_Q_score': 'entry Q-score 0.40 vs 0.43 p=0.11: null'}
F['pix_gt_1'] = (F.map_pix > 1.0).astype(float).where(F.map_pix.notna())
F['emdb_n_fields'] = F.emdb_n_fields / 10
for x in REF:
    d = F.copy()
    if d[x].dtype == bool or d[x].dtype == object: d[x] = d[x].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0}).astype(float)
    record(f'map/processing/size: {x}', 'entry', 'EM', x, B + ', paper-clustered', run_groups(d, x, B, 'paper'), REF[x])
# one per paper (paper-level aggregation) for the headline items
for x in ['krios', 'kv300', 'log2_tot_mw', 'kw_membrane', 'fund_none']:
    d = F.copy(); d[x] = d[x].astype(float)
    P = d.groupby('paper').agg(x=(x, 'mean'), res=('res', 'median'), dep_year=('dep_year', 'min'), l2nV=('l2nV', 'mean')).reset_index()
    P = add_yc(P.merge(unit_flags(E, d[['pdb', 'paper']], 'paper'), on='paper'), 'dep_year')
    if x in ('krios', 'kv300', 'kw_membrane', 'fund_none'): P['x'] = (P.x >= 0.5).astype(float)
    record(f'map/processing/size: {x}', 'paper', 'EM', x, B + ' (paper majority/mean)', run_groups(P, 'x', B, None), REF[x])
# Krios by period within the window (case vs controls)
d = F[F.case | (F.ctrl & ~F.gen & ~F.case_i)]
print('Krios share by year category (case vs window control):', d.groupby(['yc3', 'case']).krios.mean().unstack().round(2).to_dict())
print('paper-strict case entries not on a Krios:', d[d.case & (d.krios == 0)].groupby('paper').pdb.apply(list).to_dict(),
      '| microscopes:', d[d.case & (d.krios == 0)].im_microscope_model.value_counts().to_dict())

# ---- map-intrinsic local resolution at the error site (EMDB validation), case-only percentiles vs clean chains at the same window
LR = pd.read_csv(f'{D}/pdb_meta/tables/local_resolution_sites.csv')
LR = LR.merge(C[['pdb', 'chain', 'paper_case_strict', 'paper_case_incl_copies', 'in_paper_window', 'gen_first', 'paper']], on=['pdb', 'chain'], how='left')
LR['grp'] = np.select([LR.paper_case_strict == True, LR.paper_case_incl_copies == True, (LR.gen_first == True) & LR.in_paper_window, LR.gen_first == True],
                      ['case', 'copy', 'rest_w', 'rest_out'], 'artifact/other')
rows = []
for g in ['case', 'copy', 'rest_w', 'rest_out', 'artifact/other']:
    x = LR[LR.grp == g].dropna(subset=['pct_worse_than_clean'])
    if len(x) == 0: continue
    xp = x.groupby('paper').pct_worse_than_clean.median()
    q = x.q_pct_worse_than_clean.dropna()
    rows.append(dict(group=g, n_sites=len(x), n_papers=x.paper.nunique(), median_pct_lr=x.pct_worse_than_clean.median(),
                     worst_quartile_pct=100 * (x.pct_worse_than_clean >= 75).mean(), p_binom_vs25=binomtest(int((x.pct_worse_than_clean >= 75).sum()), len(x), 0.25).pvalue,
                     p_wilcoxon_vs50=wilcoxon(x.pct_worse_than_clean - 50).pvalue if len(x) > 5 else np.nan,
                     paper_median_pct_lr=xp.median(), n_paper_medians_above50=int((xp > 50).sum()),
                     median_pct_q=q.median(), worst_quartile_q_pct=100 * (q >= 75).mean()))
LRT = pd.DataFrame(rows); print('\nlocal resolution at the site (percentile of site local resolution among clean chains, higher = worse):\n', LRT.round(3).to_string(index=False))
a = LR[LR.grp == 'case'].pct_worse_than_clean.dropna(); b = LR[LR.grp.isin(['rest_w', 'rest_out'])].pct_worse_than_clean.dropna()
print(f'case vs rest (all years) Mann-Whitney p={mannwhitneyu(a, b).pvalue:.3f}')
LRT.to_csv(f'{T}/w05_local_resolution_site.csv', index=False)
ROWS.append(dict(trend='map-intrinsic local resolution at the site', level='site', method='EM', x='pct_worse_than_clean', group='case',
                 n_case=int(LRT.set_index('group').loc['case', 'n_sites']), med_case=LRT.set_index('group').loc['case', 'median_pct_lr'],
                 p=LRT.set_index('group').loc['case', 'p_wilcoxon_vs50'], full_ref='48th percentile, 11% worst quartile (p=0.05 toward better) [pdb_meta]'))

# ---- recorded software, starting model, fitting protocol (entry level, any V chain; EM and X-ray)
C['sw_any_build'] = C[['sw_Coot', 'sw_ISOLDE', 'sw_Rosetta']].any(axis=1)
C['irm_pdb'] = C.irm.eq('experimental (PDB) model'); C['irm_af'] = C.irm.eq('AlphaFold model'); C['irm_none'] = C.irm.eq('none recorded')
C['fit_rigid'] = C.fit.eq('rigid body only'); C['fit_flex'] = C.fit.eq('flexible fit')
cols = ['sw_Coot', 'sw_Phenix', 'sw_ISOLDE', 'sw_Rosetta', 'sw_REFMAC', 'sw_any_build', 'irm_pdb', 'irm_af', 'irm_none', 'fit_rigid', 'fit_flex']
EE = C.groupby('pdb').agg(**{c: (c, 'max') for c in cols}, res=('res', 'first'), year=('year', 'first'), method=('method', 'first'), paper=('paper', 'first'), nV=('chain', 'size')).reset_index()
EE = add_yc(EE.merge(E[['pdb', 'case', 'case_f', 'case_i', 'ctrl', 'gen', 'rest', 'rest_w', 'win_all', 'clean', 'win']], on='pdb'), 'year')
EE['l2nV'] = np.log2(EE.nV)
for c in cols: EE[c] = EE[c].astype(float)
REF2 = {'sw_Rosetta': 'deepdive 2018-21: entries recording Rosetta 1.5% errors vs 9.6% base (p=0.015)',
        'sw_Phenix': 'deepdive 2018-21: Phenix 7.6% vs 9.6% (p=0.035)', 'sw_Coot': 'deepdive 2018-21: Coot 4.7% vs 9.6% (p=0.047)',
        'sw_ISOLDE': 'deepdive 2018-21: ISOLDE 0/16 (n.s.); README: methods-text ISOLDE null', 'sw_REFMAC': 'not reported',
        'sw_any_build': 'Coot/ISOLDE/Rosetta recorded (deepdive: fewer errors)', 'irm_pdb': 'junction: PDB template 1.09% vs none 1.70% (CDR3/FR4, OR 0.56)',
        'irm_af': 'junction: AlphaFold start 0.34% (OR 0.34, 0.05-2.6)', 'irm_none': 'none recorded is the reference category (65/70 kappa error chains none)',
        'fit_rigid': 'deepdive D-: rigid-body-only 30% vs 11%; enzyme lens 28% vs 10%', 'fit_flex': 'flexible fit: no full-set claim'}
for meth in ['EM', 'X-ray']:
    for c in cols:
        if meth == 'X-ray' and c.startswith('fit_'): continue
        record(f'software/start model: {c}', 'entry', meth, c, '{YC}+res+l2nV, paper-clustered', run_groups(EE[EE.method == meth], c, '{YC} + res + l2nV', 'paper'), REF2[c])
# 2018-21 EM, as the deepdive strata
d = EE[(EE.method == 'EM') & EE.year.between(2018, 2021)]
for c in ['sw_Rosetta', 'sw_Phenix', 'sw_Coot', 'sw_any_build', 'irm_pdb', 'irm_af']:
    record(f'software/start model: {c} (EM 2018-21)', 'entry', 'EM', c, '{YC}+res+l2nV', run_groups(d, c, '{YC} + res + l2nV', 'paper'), REF2[c])
print('\nAlphaFold starting models recorded among window entries (EM+X-ray):', int(EE[EE.win].irm_af.sum()), 'of', int(EE.win.sum()),
      '| paper-strict case entries with AF start:', int(EE[EE.case].irm_af.sum()))
print('paper-strict case entries by recorded starting model:', C[C.paper_case_strict].groupby('pdb').irm.first().value_counts().to_dict())
save_rows(f'{T}/w05_map_model_meta_tests.csv')
