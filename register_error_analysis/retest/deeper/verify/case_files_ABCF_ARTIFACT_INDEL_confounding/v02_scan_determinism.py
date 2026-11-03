"""Is the SAbR/ANARCI disagreement deterministic for chains carrying each indel pattern, regardless of method,
resolution, year and lab?  Uses the existing SAbR outputs (sabr/out/{validation,new,snapshot_rest}.jsonl) for every
chain with one of the census patterns (v01), and counts residues in the claimed window where SAbR != ANARCI
(FR3 patterns: ANARCI IMGT 79-86; FR1 gap 26: ANARCI IMGT 24-37). Also records unbuilt residues inside the window and
joins the scan's error flags (chains.csv) and runs (runs_all / runs_first_occurrence).
Outputs: tables/v02_scan_chains.csv, tables/v02_scan_summary.csv
"""
import json, pandas as pd, numpy as np
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_ARTIFACT_INDEL_confounding'
CEN = pd.read_csv(f'{OUT}/tables/v01_pattern_census.csv')
key = {(p, c): pat for p, c, pat in zip(CEN.pdb, CEN.chain, CEN.pattern)}
WIN = {'FR1_26': (24, 37)}
rows = []
for fn in ['validation.jsonl', 'new.jsonl', 'snapshot_rest.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        x = json.loads(line)
        k = (x['pdb'], x['chain'])
        if k not in key or x.get('status') != 'ok' or not x.get('residues'): continue
        pat = key[k]; lo, hi = WIN.get(pat, (79, 86))
        rec = [r for r in x['residues'] if r[4] is not None and lo <= r[4] <= hi]
        if not rec:
            rows.append(dict(src=fn, pdb=k[0], chain=k[1], pattern=pat, n_win=0)); continue
        si = [r[3] for r in rec]
        gaps = (max(si) - min(si) + 1) - len(si)
        dis = [f'{r[2]}{r[4]}{r[5]}/{r[6]}{r[7]}' for r in rec if (r[4], r[5] or '') != (r[6], r[7] or '')]
        rows.append(dict(src=fn, pdb=k[0], chain=k[1], pattern=pat, n_win=len(rec), unbuilt_in_win=gaps, n_dis=len(dis),
                         dis3=len(dis) >= 3, detail=' '.join(dis)))
D = pd.DataFrame(rows).drop_duplicates(['pdb', 'chain'], keep='first')
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False, usecols=['pdb', 'chain', 'method', 'res', 'year', 'last_author', 'paper',
                                                                          'species', 'v_gene', 'err_first', 'err_modes', 'err_any', 'sars2'])
D = D.merge(C, on=['pdb', 'chain'], how='left')
RA = pd.read_csv(f'{SP}/deeper/runs_all.csv')
RF = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
def runs_in(R, p, c, lo, hi):
    s = R[(R.pdb == p) & (R.chain == c) & (R.imgt0 <= hi) & (R.imgt1 >= lo)]
    return ';'.join(f'{a}-{b}({d},{m},{o[:2]})' for a, b, d, m, o in zip(s.imgt0, s.imgt1, s.delta, s['mode'], s.origin))
D['runs_all_win'] = [runs_in(RA, p, c, *WIN.get(t, (79, 86))) for p, c, t in zip(D.pdb, D.chain, D.pattern)]
D['runs_first_win'] = [runs_in(RF, p, c, *WIN.get(t, (79, 86))) for p, c, t in zip(D.pdb, D.chain, D.pattern)]
D = D.merge(CEN.drop_duplicates(['pdb', 'chain'])[['pdb', 'chain', 'method', 'res']].rename(columns={'method': 'method_idx', 'res': 'res_idx'}),
            on=['pdb', 'chain'], how='left')
D.to_csv(f'{OUT}/tables/v02_scan_chains.csv', index=False)
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 300); pd.set_option('display.max_colwidth', 60)
print(D.sort_values(['pattern', 'res_idx'])[['pattern', 'pdb', 'chain', 'src', 'method_idx', 'res_idx', 'year', 'last_author', 'n_win', 'unbuilt_in_win',
                                             'n_dis', 'err_first', 'err_modes', 'runs_all_win', 'detail']].to_string())
D['EM'] = D.method_idx.str.contains('ELECTRON')
S = D[D.n_win > 0].groupby(['pattern', 'EM']).agg(chains=('dis3', 'size'), dis3=('dis3', 'sum'), entries=('pdb', 'nunique'),
                                                   entries_dis=('pdb', lambda s: s[D.loc[s.index, 'dis3'] == True].nunique()))
print(S); S.to_csv(f'{OUT}/tables/v02_scan_summary.csv')
