"""V01: population-level view of the claimed SAbR chain-end artifact.
For every scanned single-domain V chain (chains.csv.gz universe), take the SAbR/ANARCI per-residue records, compute
  last_imgt  : last modelled ANARCI IMGT position of the V domain
  cterm_run  : a >=3-residue SAbR-ANARCI disagreement run with SAbR > ANARCI (D+-like) that ends within 1 position of
               the last modelled residue and lies in IMGT 104-128 (computed directly from the records, so it includes
               copied/artifact runs)
and relate the D+ call rate to how far the model extends past W/F118, by method / year / chain type, and with a
cluster-robust logit (cluster = paper) adjusting for year bin, resolution, method and chain type.
Output: tables/v01_chain_end_population.csv, tables/v01_rates.csv, tables/v01_logit.csv
"""
import json, numpy as np, pandas as pd, warnings
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE1_cterm_artifact_confounding'

CH = pd.read_csv(f'{SP}/deeper/chains.csv.gz')
keys = set(zip(CH.pdb, CH.chain))
F = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
RA = pd.read_csv(f'{SP}/deeper/runs_all.csv')
I = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/tables/independent_check_DE_units.csv')


def key(n, i):
    return (int(n), i or '')


rows = []
for f in ['validation', 'snapshot_rest', 'new']:
    for l in open(f'{SP}/sabr/out/{f}.jsonl'):
        r = json.loads(l)
        if r.get('status') != 'ok' or (r['pdb'], r['chain']) not in keys:
            continue
        v = [x for x in r['residues'] if x[4] is not None and x[6] is not None]
        if not v:
            continue
        last = max(x[4] for x in v)
        a118 = [x for x in v if x[4] == 118 and not x[5]]
        # disagreement flags in residue order
        flags = [(key(x[4], x[5]) != key(x[6], x[7])) for x in v]
        runs, cur = [], []
        for k, fl in enumerate(flags):
            if fl:
                cur.append(k)
            else:
                if len(cur) >= 3: runs.append(cur)
                cur = []
        if len(cur) >= 3: runs.append(cur)
        ct_run = False; ct_delta = None; any_run_c = False
        for rr in runs:
            a0, a1 = v[rr[0]][4], v[rr[-1]][4]
            d = v[rr[-1]][6] - v[rr[-1]][4]
            if a1 >= 104 and a0 <= 128:
                any_run_c = True
                if rr[-1] >= len(v) - 2 and d > 0:
                    ct_run = True; ct_delta = d
        last_res = v[-1]
        rows.append(dict(pdb=r['pdb'], chain=r['chain'], last_imgt=last, n_after118=sum(1 for x in v if x[4] > 118),
                         last_aa=last_res[2], last_sabr=last_res[6], last_anarci=last_res[4],
                         cterm_run_dplus=ct_run, cterm_delta=ct_delta, any_run_104_128=any_run_c))
P = pd.DataFrame(rows).drop_duplicates(['pdb', 'chain'])
P = P.merge(CH[['pdb', 'chain', 'method', 'res', 'year', 'ctype', 'vhh', 'sars2', 'paper', 'unit', 'err_first', 'err_any']], on=['pdb', 'chain'])
fo = F[F['mode'] == 'Dplus_CDR3short'][['pdb', 'chain']].drop_duplicates().assign(fo_dplus=True)
P = P.merge(fo, on=['pdb', 'chain'], how='left'); P['fo_dplus'] = P.fo_dplus.fillna(False).astype(bool)
ra = RA[RA['mode'] == 'Dplus_CDR3short'][['pdb', 'chain']].drop_duplicates().assign(any_dplus=True)
P = P.merge(ra, on=['pdb', 'chain'], how='left'); P['any_dplus'] = P.any_dplus.fillna(False).astype(bool)
P['end_cat'] = pd.cut(P.last_imgt, [0, 117, 118, 119, 121, 124, 129], labels=['<=117', '118', '119', '120-121', '122-124', '>=125']).astype(str)
P['ends_le121'] = P.last_imgt <= 121
P['yb'] = pd.cut(P.year, [0, 2019, 2020, 2021, 2023, 2030], labels=['<=2019', '2020', '2021', '2022-23', '>=2024']).astype(str)
P['ct'] = np.where(P.vhh == 1, 'VHH', P.ctype)
P.to_csv(f'{OUT}/tables/v01_chain_end_population.csv', index=False)

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
out = []
for grp, g in [('all', P), ('EM', P[P.method == 'EM']), ('X-ray', P[P.method == 'X-ray'])]:
    for ec, h in g.groupby('end_cat'):
        out.append(dict(subset=grp, end_cat=ec, n_chains=len(h), n_units=h.unit.nunique(),
                        cterm_dplus_run=int(h.cterm_run_dplus.sum()), frac_cterm_dplus=h.cterm_run_dplus.mean(),
                        any_dplus_runs_all=int(h.any_dplus.sum()), first_occ_dplus=int(h.fo_dplus.sum()), frac_first_occ_dplus=h.fo_dplus.mean(),
                        any_run_104_128=int(h.any_run_104_128.sum())))
Rt = pd.DataFrame(out)
Rt.to_csv(f'{OUT}/tables/v01_rates.csv', index=False)
print(Rt.round(3).to_string())

# one chain per unit, logit for first-occurrence D+ and for any C-terminal D+-like run
U = P.sort_values(['unit', 'fo_dplus'], ascending=[True, False]).drop_duplicates('unit')
lg = []
for sub, d in [('all', U), ('EM', U[U.method == 'EM'])]:
    for y in ['fo_dplus', 'cterm_run_dplus']:
        dd = d.dropna(subset=['res']).copy(); dd['y'] = dd[y].astype(int); dd['x'] = dd.ends_le121.astype(int)
        form = 'y ~ x + C(yb) + res + C(ct)' + (' + C(method)' if sub == 'all' else '')
        try:
            m = smf.logit(form, data=dd).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(dd.paper)[0]})
            lg.append(dict(subset=sub, outcome=y, n_units=len(dd), n_le121=int(dd.x.sum()), rate_le121=dd[dd.x == 1].y.mean(),
                           rate_gt121=dd[dd.x == 0].y.mean(), OR=np.exp(m.params['x']), lo=np.exp(m.conf_int().loc['x', 0]),
                           hi=np.exp(m.conf_int().loc['x', 1]), p=m.pvalues['x'], adjusted='year bin, resolution, chain type' + (', method' if sub == 'all' else '') + '; cluster by paper'))
        except Exception as ex:
            lg.append(dict(subset=sub, outcome=y, note=str(ex)[:80]))
L = pd.DataFrame(lg); L.to_csv(f'{OUT}/tables/v01_logit.csv', index=False)
print(L.round(4).to_string())
# how do chains ending at 118 that SAbR leaves alone differ?
e118 = P[(P.last_imgt <= 119)]
print('\nchains ending <=119:', len(e118), 'cterm D+ run', int(e118.cterm_run_dplus.sum()))
print(pd.crosstab([e118.method], [e118.cterm_run_dplus]))
print('last residue SAbR minus ANARCI for chains ending 118 (W/F118 last):')
x = P[P.last_imgt == 118]
print((x.last_sabr - x.last_anarci).value_counts().sort_index())
