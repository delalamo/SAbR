"""Case vs control tests for every structural metric, adjusted for chain class, deposition-year bin and resolution
(cluster-robust SE by entry), with raw Mann-Whitney / Fisher p and Benjamini-Hochberg q over all adjusted tests.
Output tables/tests.csv"""
import warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from common import SP, TAB
warnings.filterwarnings('ignore')

s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
U = pd.read_csv(f'{TAB}/unit_metrics.csv', dtype={'chain': str, 'partner': str})
U['case'] = (U.role_cc == 'case').astype(int)
U = U.merge(s[['unit', 'pdb', 'sars2']].drop_duplicates(['unit', 'pdb']), on=['unit', 'pdb'], how='left')
U['Dmode'] = U.modes.fillna('').str.contains('D')
emeta = s.groupby('pdb').agg(res=('res', 'first'), ybin=('ybin', 'first'), role_cc=('role_cc', 'first'),
                             sars2=('sars2', 'max'))
E = pd.read_csv(f'{TAB}/entry_vrpt.csv').join(emeta, on='pdb')
E['case'] = (E.role_cc == 'case').astype(int)
FV = pd.read_csv(f'{TAB}/fv_orientation.csv', dtype={'H': str, 'L': str})
FV['case'] = FV.fv_case.astype(int)
FV['core_rms'] = FV[['rmsH_sabr', 'rmsL_sabr']].mean(1)
try:
    PF = pd.read_csv(f'{TAB}/pred_compare_fv.csv', dtype={'H': str, 'L': str})
    FV = FV.merge(PF.drop(columns=['fv_case']), on=['pdb', 'H', 'L'], how='left')
    for k in ['HL', 'HC1', 'LC1', 'HC2', 'LC2', 'dc']:
        FV[f'abs_d_{k}'] = FV[f'd_{k}'].abs()
    A = ['HL', 'HC1', 'LC1', 'HC2', 'LC2']
    ref = pd.read_csv(f'{SP}/deeper/cache/struct_unusual/abangle_src/ABangle/data/All_Angles.dat', sep='\t')
    ref = ref[ref.Method == 'X-RAY']; A6 = A + ['dc']
    mu = ref[A6].mean().values; ci = np.linalg.inv(np.cov(ref[A6].values.T))
    X = FV[[f'pred_{k}' for k in A6]].values - mu
    FV['pred_maha'] = np.sqrt(np.einsum('ij,jk,ik->i', X, ci, X))
    PC = pd.read_csv(f'{TAB}/pred_compare_chain.csv', dtype={'chain': str})
    PC = PC.merge(s.drop_duplicates('pdb')[['pdb', 'res', 'ybin']], on='pdb', how='left')
    PC['case'] = (PC.role_cc == 'case').astype(int)
    PC['cls'] = PC.cclass.replace({'K': 'VL', 'L': 'VL'})
except FileNotFoundError:
    PF = PC = None

rows = []


def test(name, df, y, kind, covars, group, desc, family='', sub=''):
    x = df[[y, 'case', group] + [c.replace('C(', '').replace(')', '') for c in covars]].dropna()
    x = x[np.isfinite(x[y].astype(float))].copy(); x[y] = x[y].astype(float)
    a, b = x[x.case == 1][y].astype(float), x[x.case == 0][y].astype(float)
    if len(a) < 5 or len(b) < 5:
        return
    covs = [c for c in covars if x[c.replace('C(', '').replace(')', '')].nunique() > 1]
    f = f'{y} ~ case' + ''.join(' + ' + c for c in covs)
    o = dict(test=name, family=family, subset=sub, metric=y, kind=kind, n_case=len(a), n_ctrl=len(b), desc=desc)
    g = pd.factorize(x[group])[0]
    if kind == 'binary':
        o.update(case_val=100 * a.mean(), ctrl_val=100 * b.mean(), p_raw=stats.fisher_exact(
            [[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1])
        try:
            m = smf.logit(f, x).fit(disp=0, cov_type='cluster', cov_kwds={'groups': g}, maxiter=200)
            o.update(effect=np.exp(m.params['case']), ci_lo=np.exp(m.conf_int().loc['case', 0]),
                     ci_hi=np.exp(m.conf_int().loc['case', 1]), p_adj=m.pvalues['case'], effect_type='adjusted OR')
            if not np.isfinite(o['p_adj']) or o['effect'] > 1e3 or o['effect'] < 1e-3:
                raise ValueError
        except Exception:
            m = smf.ols(f, x).fit(cov_type='cluster', cov_kwds={'groups': g})
            o.update(effect=100 * m.params['case'], ci_lo=100 * m.conf_int().loc['case', 0],
                     ci_hi=100 * m.conf_int().loc['case', 1], p_adj=m.pvalues['case'], effect_type='adjusted risk diff (pp)')
    else:
        o.update(case_val=a.median(), ctrl_val=b.median(), case_mean=a.mean(), ctrl_mean=b.mean(),
                 p_raw=stats.mannwhitneyu(a, b).pvalue)
        m = smf.ols(f, x).fit(cov_type='cluster', cov_kwds={'groups': g})
        o.update(effect=m.params['case'], ci_lo=m.conf_int().loc['case', 0], ci_hi=m.conf_int().loc['case', 1],
                 p_adj=m.pvalues['case'], effect_type='adjusted mean diff')
    o['adjusted_for'] = ', '.join(c.replace('C(', '').replace(')', '') for c in covs) + f'; cluster={group}'
    rows.append(o)


UC = ['C(cclass)', 'C(ybin)', 'res']
# ---------------- (c) backbone geometry beyond and at the site
for y, desc in [('rama_out_out', 'Ramachandran outliers per 100 V residues, outside the site'),
                ('rama_allowed_out', "Ramachandran 'allowed' (not favoured) per 100, outside the site"),
                ('rota_out_out', 'rotamer outliers per 100, outside the site'),
                ('clash_out', 'residues with a clash per 100, outside the site'),
                ('geo_out_out', 'residues with bond/angle outliers per 100, outside the site'),
                ('cis_nonpro_out', 'non-Pro cis peptides per 100, outside the site'),
                ('caca_out_out', 'CA-CA distance outliers (|d-3.80|>0.15, trans) per 100, outside the site'),
                ('sc_trunc_out', 'residues with truncated side chains per 100, outside the site')]:
    test(y, U, y, 'cont', UC, 'pdb', desc, 'c_geometry_outside')
for y, desc in [('rama_out_site', 'Ramachandran outlier within the site +-2'),
                ('cis_nonpro_site', 'non-Pro cis peptide within the site +-2'),
                ('caca_out_site', 'CA-CA distance outlier within the site +-2'),
                ('clash_site', 'clash within the site +-2'),
                ('rota_out_site', 'rotamer outlier within the site +-2'),
                ('sc_trunc_site', 'truncated side chain within the site +-2')]:
    test(y, U, y, 'binary', UC, 'pdb', desc, 'c_geometry_site')
U['any_rama_out_out'] = (U.rama_out_out > 0).astype(float)
U['any_cis_nonpro_out'] = (U.cis_nonpro_out > 0).astype(float)
U['any_caca_out_out'] = (U.caca_out_out > 0).astype(float)
for y, desc in [('any_rama_out_out', 'any Ramachandran outlier outside the site'),
                ('any_cis_nonpro_out', 'any non-Pro cis peptide outside the site'),
                ('any_caca_out_out', 'any CA-CA outlier outside the site')]:
    test(y, U, y, 'binary', UC, 'pdb', desc, 'c_geometry_outside')
# partner chain (only partners with no disagreement run of any kind)
PU = U[(U.partner_err_any == False)]
for y, desc in [('p_rama_out_V', 'partner V domain: Ramachandran outliers per 100'),
                ('p_rota_out_V', 'partner V domain: rotamer outliers per 100'),
                ('p_clash_V', 'partner V domain: clash residues per 100'),
                ('p_cis_nonpro_V', 'partner V domain: non-Pro cis per 100'),
                ('p_caca_out_V', 'partner V domain: CA-CA outliers per 100'),
                ('p_sc_trunc_V', 'partner V domain: truncated side chains per 100'),
                ('p_q_V', 'partner V domain: mean Q-score'),
                ('p_unbuilt_int', 'partner V domain: unbuilt internal residues')]:
    test(y, PU, y, 'cont', UC, 'pdb', desc, 'c_partner')
# ---------------- (d) map fit
for y, desc in [('q_V', 'V-domain mean Q-score'), ('q_out', 'mean Q-score outside the site'),
                ('q_V_rel', 'V-domain mean Q minus entry median Q'),
                ('q_site_minus_out', 'site mean Q minus rest-of-domain mean Q'),
                ('q_entry', 'entry median Q-score (all chains)')]:
    test(y, U, y, 'cont', UC, 'pdb', desc, 'd_mapfit')
EC = ['C(ybin)', 'res']
for y, desc in [('e_Q_score', 'entry Q-score (wwPDB)'), ('e_clashscore', 'entry clashscore'),
                ('e_percent_rama_outliers', 'entry % Ramachandran outliers'),
                ('e_percent_rota_outliers', 'entry % rotamer outliers'), ('e_angles_rmsz', 'entry angle RMSZ'),
                ('e_bonds_rmsz', 'entry bond RMSZ'), ('e_atom_inclusion_backbone', 'entry backbone atom inclusion'),
                ('rest_rama_out', 'non-V residues: % Ramachandran outliers'),
                ('rest_rota_out', 'non-V residues: % rotamer outliers'), ('rest_clash', 'non-V residues: % clash'),
                ('rest_q', 'non-V residues: mean Q-score'),
                ('rest_rama_allowed', "non-V residues: % Ramachandran 'allowed'"),
                ('v_rama_allowed', "all antibody V residues in the entry: % Ramachandran 'allowed'"),
                ('gap_rama_allowed', "within-entry gap: V minus non-V % Ramachandran 'allowed'"),
                ('gap_clash', 'within-entry gap: V minus non-V % clash residues'),
                ('gap_rota_out', 'within-entry gap: V minus non-V % rotamer outliers'),
                ('gap_q', 'within-entry gap: V minus non-V mean Q-score')]:
    test(y, E, y, 'cont', EC, 'pdb', desc, 'entry_rest_of_model')
# ---------------- (b) CDR-H3 base and CDR lengths
H = U[U.cclass.isin(['H', 'VHH'])]
test('h3_kinked_all', H, 'h3_kinked', 'binary', UC, 'pdb', 'deposited H3 base kinked (structure-based numbering)', 'b_h3')
test('h3_kinked_noD', H[(H.case == 0) | ~H.Dmode], 'h3_kinked', 'binary', UC, 'pdb',
     'deposited H3 base kinked, cases without CDR3/FR4 (D) errors', 'b_h3', 'no D-mode cases')
test('h3_kinked_D', H[(H.case == 0) | H.Dmode], 'h3_kinked', 'binary', UC, 'pdb',
     'deposited H3 base kinked, D-mode cases only', 'b_h3', 'D-mode cases')
test('h3_kinked_VH_noD', H[(H.cclass == 'H') & ((H.case == 0) | ~H.Dmode)], 'h3_kinked', 'binary', ['C(ybin)', 'res'],
     'pdb', 'deposited H3 base kinked, VH only, no D-mode cases', 'b_h3', 'VH, no D')
test('rare_cdr_len', U, 'rare_cdr_len', 'binary', UC, 'pdb', 'any non-H3 CDR length with <2% frequency', 'b_cdrlen')
test('cdr3_len', U[U.cclass.isin(['H', 'VHH'])], 'cdr3_len', 'cont', UC, 'pdb', 'CDR-H3 length (IMGT)', 'b_cdrlen')
CP = pd.read_csv(f'{TAB}/cispro95.csv')
test('kappa_L3_cisPro95', CP, 'cis', 'binary', ['C(ybin)', 'res'], 'pdb',
     'kappa 9-residue CDR-L3 with Pro95 (IMGT 115): canonical cis peptide modelled', 'b_canonical')
CPn = CP[(CP.case == 0) | ~CP.modes.fillna('').str.contains('D|E_FR4')]
test('kappa_L3_cisPro95_awayfromsite', CPn, 'cis', 'binary', ['C(ybin)', 'res'], 'pdb',
     'same, cases whose error is not in CDR3/FR4 (D or E modes excluded)', 'b_canonical', 'error away from L3')
# ---------------- (a) VH-VL orientation
FC = ['C(ybin)', 'res']
for y, desc in [('maha_sabr', 'Mahalanobis distance of ABangle orientation from the X-ray reference'),
                ('outlier99', 'orientation outside the 99% reference ellipsoid'),
                ('core_rms', 'core-set CA RMSD to the ABangle consensus (mean VH, VL)'),
                ('HL_sabr', 'HL torsion'), ('HC1_sabr', 'HC1'), ('HC2_sabr', 'HC2'), ('LC1_sabr', 'LC1'),
                ('LC2_sabr', 'LC2'), ('dc_sabr', 'dc (A)')]:
    test(y, FV, y, 'binary' if y == 'outlier99' else 'cont', FC, 'pdb', desc, 'a_orientation')
FV['absdev_HL'] = (FV.HL_sabr - FV.HL_sabr[FV.case == 0].median()).abs()
for k in ['HL', 'HC1', 'HC2', 'LC1', 'LC2', 'dc']:
    FV[f'absdev_{k}'] = (FV[f'{k}_sabr'] - FV[f'{k}_sabr'][FV.case == 0].median()).abs()
    test(f'absdev_{k}', FV, f'absdev_{k}', 'cont', FC, 'pdb', f'|{k} - control median| (spread)', 'a_orientation')
FVc = FV[FV.core_rms < FV.core_rms[FV.case == 0].quantile(0.9)]
test('maha_coreOK', FVc, 'maha_sabr', 'cont', FC, 'pdb', 'Mahalanobis distance, Fvs with undistorted cores only',
     'a_orientation', 'core_rms < control 90th pct')
# ---------------- (e) deviation from ImmuneBuilder predictions
if PC is not None:
    PCC = ['C(cls)', 'C(ybin)', 'res']
    for y, desc in [('fw_rmsd_sabr', 'framework CA RMSD deposited vs predicted (structural correspondence, site excluded)'),
                    ('cdr1_rmsd', 'CDR1 CA RMSD after framework superposition'),
                    ('cdr2_rmsd', 'CDR2 CA RMSD after framework superposition'),
                    ('cdr3_rmsd', 'CDR3 CA RMSD after framework superposition'),
                    ('cdr_bin_agree', 'fraction of non-H3 CDR residues in the same backbone bin as predicted'),
                    ('pred_err_cdr3', 'ImmuneBuilder ensemble error, CDR3 (intrinsic predictability)'),
                    ('pred_err_chain', 'ImmuneBuilder ensemble error, whole chain'),
                    ('pred_err_site', 'ImmuneBuilder ensemble error at the error site (controls: matched site)')]:
        test('vsIB_' + y, PC, y, 'cont', PCC, 'pdb', desc, 'e_prediction')
    test('vsIB_all_loops_same_bins', PC, 'all_loops_same_bins', 'binary', PCC, 'pdb',
         'all non-H3 CDRs in the predicted backbone-bin pattern (canonical-class agreement)', 'e_prediction')
    PH = PC[PC.cls.isin(['H', 'VHH'])]
    test('pred_h3_kinked', PH, 'pred_h3_kinked', 'binary', PCC, 'pdb',
         'PREDICTED H3 base kinked (sequence-intrinsic)', 'b_h3')
    test('pred_h3_kinked_VH', PH[PH.cls == 'H'], 'pred_h3_kinked', 'binary', ['C(ybin)', 'res'], 'pdb',
         'PREDICTED H3 base kinked, VH only', 'b_h3', 'VH')
    test('pred_h3_kinked_VHH', PH[PH.cls == 'VHH'], 'pred_h3_kinked', 'binary', ['C(ybin)', 'res'], 'pdb',
         'PREDICTED H3 base kinked, VHH only', 'b_h3', 'VHH')
    # chains in case Fvs that themselves carry no error vs control chains
    PN = PC[(PC.case == 0) | (PC.chain_has_error == False)]
    test('pred_fw_rmsd_errorfree_chain', PN, 'fw_rmsd_sabr', 'cont', PCC, 'pdb',
         'framework RMSD vs prediction, error-free chains of case Fvs', 'e_prediction', 'error-free chain')
    test('pred_cdr_bin_agree_errorfree_chain', PN, 'cdr_bin_agree', 'cont', PCC, 'pdb',
         'non-H3 CDR backbone-bin agreement with prediction, error-free chains of case Fvs', 'e_prediction', 'error-free chain')
    test('pred_cdr1_rmsd_errorfree_chain', PN, 'cdr1_rmsd', 'cont', PCC, 'pdb',
         'CDR1 RMSD vs prediction, error-free chains of case Fvs', 'e_prediction', 'error-free chain')
    test('pred_err_cdr3_errorfree_chain', PN, 'pred_err_cdr3', 'cont', PCC, 'pdb',
         'IB ensemble error CDR3, error-free chains of case Fvs', 'e_prediction', 'error-free chain')

    test('pred_maha', FV, 'pred_maha', 'cont', FC, 'pdb',
         'PREDICTED Fv orientation: Mahalanobis distance from the X-ray reference (sequence-intrinsic)', 'a_orientation')
    test('pred_core_rms', FV, 'pred_core_rms', 'cont', FC, 'pdb',
         'PREDICTED core-set CA RMSD to the ABangle consensus (sequence-intrinsic)', 'a_orientation')
    for k in ['HL', 'HC1', 'LC1', 'HC2', 'LC2', 'dc']:
        test(f'abs_d_{k}', FV, f'abs_d_{k}', 'cont', FC, 'pdb', f'|deposited - predicted| {k}', 'e_prediction')

T = pd.DataFrame(rows)
T['q_bh'] = multipletests(T.p_adj.fillna(1), method='fdr_bh')[1]
T.to_csv(f'{TAB}/tests.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 40)
print(T[['test', 'n_case', 'n_ctrl', 'case_val', 'ctrl_val', 'effect', 'ci_lo', 'ci_hi', 'p_adj', 'q_bh']].round(4).to_string())
