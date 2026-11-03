"""Collect all test rows (w01-w05) into one wide table: one row per test with paper-strict, incl-copies, rest_w (in-window
errors not paper-confirmed) and rest (all-years first-occurrence errors not paper-confirmed) estimates; heterogeneity
z-tests of log ORs (case vs rest, case vs rest_w); minimum detectable OR at 80% power for binary exposures."""
import sys, glob; sys.path.insert(0, '.')
from ww_common import *
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{T}/w0[1-5]_*tests.csv'))], ignore_index=True)
key = ['trend', 'level', 'method', 'x', 'spec']
R['spec'] = R.spec.fillna(''); R['full_ref'] = R.full_ref.fillna('')
out = []
for k, g in R.groupby(key, sort=False):
    gg = {d['group']: d for d in [dict(zip(g.columns, v)) for v in g.values]}
    row = dict(zip(key, k)); row['full_ref'] = g.full_ref.iloc[0]
    for grp in ['case', 'case_i', 'rest_w', 'win_all', 'rest', 'gen']:
        d = gg.get(grp, {})
        row[f'{grp}_n'] = f"{d.get('n_case', np.nan):.0f}/{d.get('n_ctrl', np.nan):.0f}" if pd.notna(d.get('n_case', np.nan)) else ''
        desc = ''
        if pd.notna(d.get('pct_case', np.nan)): desc = f"{d['pct_case']:.0f}% vs {d.get('pct_ctrl', np.nan):.0f}%"
        elif pd.notna(d.get('med_case', np.nan)): desc = f"median {d['med_case']:.3g} vs {d.get('med_ctrl', np.nan):.3g}"
        row[f'{grp}_desc'] = desc
        row[f'{grp}_est'] = fmt(d) if pd.notna(d.get('OR', np.nan)) else ''
        row[f'{grp}_OR'] = d.get('OR', np.nan); row[f'{grp}_p'] = d.get('p', np.nan)
    c, rr, rw = gg.get('case', {}), gg.get('rest', {}), gg.get('rest_w', {})
    row['z_het_case_vs_rest'] = z_het(c, rr) if all(pd.notna(x.get('se_log', np.nan)) for x in [c, rr]) else np.nan
    row['z_het_case_vs_rest_w'] = z_het(c, rw) if all(pd.notna(x.get('se_log', np.nan)) for x in [c, rw]) else np.nan
    if pd.notna(c.get('pct_ctrl', np.nan)) and pd.notna(c.get('n_case', np.nan)):
        p0 = c['pct_ctrl'] / 100
        row['min_det_OR80_up'] = min_det_or(int(c['n_case']), p0, int(c['n_ctrl']), direction=1)
        row['min_det_OR80_down'] = min_det_or(int(c['n_case']), p0, int(c['n_ctrl']), direction=-1)
    if pd.notna(c.get('hi', np.nan)) and c.get('lo', 0) > 0:
        row['case_CI_ratio'] = c['hi'] / c['lo']
    row['note'] = '; '.join(str(x) for x in g.note.dropna().unique()) if 'note' in g else ''
    out.append(row)
S = pd.DataFrame(out)
S.to_csv(f'{T}/summary_all_tests.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print(S[['trend', 'level', 'method', 'case_n', 'case_desc', 'case_est', 'case_i_est', 'rest_w_est', 'win_all_est', 'rest_est', 'z_het_case_vs_rest', 'z_het_case_vs_rest_w', 'min_det_OR80_up', 'min_det_OR80_down']].round(3).to_string())
