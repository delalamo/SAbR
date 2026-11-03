"""Assemble the case table (cases + matched controls) and compare cases with controls at the case-file level.

Merges: cases/controls (s01), local site (s03), context (s04), methods + templates (s05), gap-artifact flags (s10).
Adds a sequence feature at the site: an indel (relative to the chain-type norm) in the IMGT region where the case's run
lies (A: N-terminus missing in SEQRES or FR1 gap; F: FR1 gap at 11-26 or atypical CDR1 length; B: indel at IMGT 66-75
or atypical CDR2 length; C: atypical FR3 length).
Tests (entry level, Fisher exact / Mann-Whitney; BH over all tests): small n, descriptive.
Outputs: tables/case_table.csv, tables/casefile_tests.csv
"""
import json, pickle, numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF'
T = f'{OUT}/tables'
L = pd.read_csv(f'{T}/local_sites.csv'); X = pd.read_csv(f'{T}/context.csv'); M = pd.read_csv(f'{T}/methods_flags.csv')
TP = pd.read_csv(f'{T}/templates.csv'); AR = pd.read_csv(f'{T}/artifact_runs.csv')
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
AN = pickle.load(open(f'{SP}/deeper/cache/seq_novelty/anarci_num.pkl', 'rb'))
CDR_NORM = {'H': {'L1': {8}, 'L2': {7, 8}}, 'K': {'L1': {6, 7, 11, 12}, 'L2': {3}}, 'L': {'L1': {6, 8, 9}, 'L2': {3}}}
def site_indel(pdb, ch, ct, mode):
    v = C[(C.pdb == pdb) & (C.chain == ch)]
    if not len(v): return np.nan
    v = v.iloc[0]
    for s in (v.seq, v.vseq):
        for d in AN.get(s) or []:
            if (d[0] in 'KL') != (ct in 'KL'): continue
            pos = {int(p) for p, i, a in d[4] if str(i).strip() == ''}
            ins = {int(p) for p, i, a in d[4] if str(i).strip() != ''}
            L1 = sum(27 <= p <= 38 for p in pos) + sum(27 <= p <= 38 for p in ins)
            L2 = sum(56 <= p <= 65 for p in pos) + sum(56 <= p <= 65 for p in ins)
            fr3 = sum(66 <= p <= 104 for p in pos) + sum(66 <= p <= 104 for p in ins)
            norm_fr3 = {38} if ct == 'H' else ({36} if ct == 'K' else {36, 38})
            if mode == 'A_FR1Nterm': return int(min(pos) > 1 or bool({k for k in range(min(pos), 27) if k not in pos} - {10}))
            if mode == 'F_CDR1': return int(bool({k for k in range(11, 27) if k not in pos} - {10}) or L1 not in CDR_NORM[ct]['L1'])
            if mode == 'B_CDR2FR3': return int(bool({k for k in range(66, 76) if k not in pos} - {73}) or any(66 <= p <= 75 for p in ins) or L2 not in CDR_NORM[ct]['L2'])
            if mode == 'C_FR3': return int(fr3 not in norm_fr3)
    return np.nan
D = L.merge(X.drop(columns=['group', 'mode', 'case_pdb']), on=['pdb', 'chain'], how='left')
D = D.merge(M.drop(columns=['group', 'mode', 'case_pdb']).drop_duplicates('pdb'), on='pdb', how='left')
D['site_indel'] = [site_indel(p, c, ct, m) for p, c, ct, m in zip(D.pdb, D.chain, D.ctype, D['mode'])]
ab = TP[TP.is_antibody == True].copy()
best = ab.sort_values('ident', ascending=False).drop_duplicates(['pdb', 'mode'])
D = D.merge(best[['pdb', 'mode', 'template', 'ident', 'same_ab', 'length_mismatch_at_site', 'template_has_run']].rename(columns={'template': 'best_ab_template'}),
            on=['pdb', 'mode'], how='left')
D['ab_template_stated'] = D.best_ab_template.notna().astype(int)
D['other_ab_template'] = ((D.best_ab_template.notna()) & (D.same_ab != True)).astype(int)
D['tmpl_mismatch'] = ((D.other_ab_template == 1) & (D.length_mismatch_at_site == True)).astype(int)
D['artifact'] = [int(((AR.pdb == p) & (AR['mode'] == m)).any()) for p, m in zip(D.pdb, D['mode'])]
D['clash_excess'] = D.clash_win - D.clash_rest
D['lr_excess'] = D.lr_win - D.lr_dom
D['prior_hires_same_ab'] = (D.same_ab_before_hires.fillna(0) > 0).astype(int)
D['contested_ag'] = (D.ag_6m_otherlab_inscan.fillna(0) >= 5).astype(int)
D['otherlab_ag_err_rate'] = D.ag_6m_otherlab_inscan_err / D.ag_6m_otherlab_inscan.replace(0, np.nan)
D['unbuilt_near'] = (D.unbuilt_win5 > 0).astype(int)
D['nterm_unbuilt'] = (D.first_built_imgt > D.seqres_first_imgt).astype(int)
D.to_csv(f'{T}/case_table.csv', index=False)

rows = []
def bin_test(var, sub=None, label=None):
    d = D if sub is None else D[sub(D)]
    d = d[d[var].notna()]
    a = d[d.group == 'case'][var].astype(int); b = d[d.group == 'control'][var].astype(int)
    if len(a) == 0 or len(b) == 0: return
    t = [[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]]
    orr, p = fisher_exact(t)
    rows.append(dict(test=label or var, kind='binary', case=f'{a.mean():.0%} ({a.sum()}/{len(a)})', control=f'{b.mean():.0%} ({b.sum()}/{len(b)})', effect=orr, p=p))
def cont_test(var, sub=None, label=None):
    d = D if sub is None else D[sub(D)]
    a = d[d.group == 'case'][var].dropna(); b = d[d.group == 'control'][var].dropna()
    if len(a) < 3 or len(b) < 3: return
    p = mannwhitneyu(a, b).pvalue
    rows.append(dict(test=label or var, kind='median', case=f'{a.median():.2f} (n={len(a)})', control=f'{b.median():.2f} (n={len(b)})', effect=a.median() - b.median(), p=p))
cont_test('fit_pct', label='site map fit, percentile within V domain')
bin_test('unbuilt_near', label='unbuilt residues within the window +-5')
bin_test('nterm_unbuilt', sub=lambda d: d['mode'] == 'A_FR1Nterm', label='A: SEQRES N-terminal residues not built')
bin_test('site_indel', label='indel / atypical length in the run region (SEQRES-based)')
bin_test('artifact', label='run explained by a high-res-confirmed gap-placement artifact')
cont_test('clash_excess', label='clashes per residue, site minus rest of V')
cont_test('b_win_z', label='B-factor z of the site within the V domain')
cont_test('lr_excess', label='local resolution, site minus V domain (A)')
bin_test('win_contact_nonab', label='site contacts a non-antibody chain')
bin_test('win_contact_partner', label='site contacts the partner chain')
bin_test('dom_contact_nonab', label='V domain contacts a non-antibody chain')
bin_test('ab_template_stated', sub=lambda d: d.fulltext == True, label='antibody template named (paper or PDB record), full text available')
bin_test('other_ab_template', label='a different antibody named as template')
bin_test('tmpl_mismatch', label='named different-antibody template differs in length at the site')
for f in ['homology_tool', 'alphafold', 'rigid_body', 'mutated', 'isolde', 'rosetta', 'local_refine', 'fv_only', 'poor_density', 'icosahedral', 'own_crystal']:
    bin_test(f, sub=lambda d: d.fulltext == True, label=f'methods mention: {f}')
bin_test('prior_hires_same_ab', label='same V region solved before at <3 A')
bin_test('contested_ag', label='>=5 other-lab same-antigen entries in the scan within +-6 months')
bin_test('ag_first_ab_structure', label='first antibody structure of the antigen')
cont_test('v_id', label='germline identity')
cont_test('nn_diff_V', label='identity to nearest earlier V (different antibody)')
R = pd.DataFrame(rows); R['q'] = multipletests(R.p, method='fdr_bh')[1]
R.to_csv(f'{T}/casefile_tests.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
print(R.round(4).to_string())
# per-mode view of key binary features
for v in ['site_indel', 'artifact', 'unbuilt_near', 'tmpl_mismatch', 'win_contact_nonab', 'prior_hires_same_ab']:
    print(v, D.groupby(['mode', 'group'])[v].agg(lambda s: f'{int(s.sum())}/{s.notna().sum()}').unstack().to_dict())
