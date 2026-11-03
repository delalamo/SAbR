"""Per FR3 gap/insertion pattern: how often do SAbR and ANARCI disagree on >=3 FR3 residues in ANY chain carrying it,
separately for the high-resolution (<3 A) validation set and the 3-5 A scan, X-ray vs EM?
Pattern computed on the *modelled* residues (as in investigator's s08) but only where FR3 (74-100) is fully modelled and
contiguous. Outputs: tables/fr3_pattern_chains.csv.gz (per chain), tables/fr3_pattern_rates.csv (per pattern).
"""
import json, numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3_INDEL_C_confounding'
NORM = {'H': {73}, 'K': {73, 81, 82}, 'L': {73}}
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'chain', 'method', 'res', 'year'], low_memory=False)
meth = {(p, c): (m, r) for p, c, m, r in zip(C.pdb, C.chain, C.method, C.res)}
rows = []
for fn in ['validation.jsonl', 'new.jsonl', 'snapshot_rest.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        x = json.loads(line); rec = x.get('residues')
        if not rec or x.get('status') != 'ok': continue
        ct = x.get('chain_type') or 'H'
        if ct not in NORM: continue
        w = [r for r in rec if r[4] is not None and 64 <= r[4] <= 106]
        if len(w) < 10: continue
        si = [r[3] for r in w]
        contiguous = max(si) - min(si) + 1 == len(si)
        pres = {r[4] for r in w}
        spanned = min(pres) <= 66 and max(pres) >= 104
        miss = tuple(k for k in range(66, 105) if k not in pres and k not in NORM[ct])
        ins = tuple(sorted({f'{r[4]}{r[5]}' for r in w if r[5] and 66 <= r[4] <= 104}))
        dis = [r for r in w if 66 <= r[4] <= 104 and (r[4], r[5] or '') != (r[6], r[7] or '')]
        m, res = meth.get((x['pdb'], x['chain']), (None, None))
        rows.append(dict(set='hires' if fn.startswith('validation') else 'scan', pdb=x['pdb'], chain=x['chain'], ct=ct,
                         method=m, res=res, contiguous=contiguous, spanned=spanned,
                         pattern=str(miss) + ('+' + ','.join(ins) if ins else ''), n_dis=len(dis), dis3=len(dis) >= 3,
                         dis_pos=','.join(str(r[4]) + (r[5] or '') + '>' + str(r[6]) + (r[7] or '') for r in dis)))
D = pd.DataFrame(rows)
# resolution / method for every entry from the PDB-wide index; 'hires' = resolution < 3.0 A (any method), not the file label
RES = {}
with open('/tmp/supp/pdbwide/entries.idx', errors='replace') as f:
    for line in f:
        t = line.rstrip('\n').split('\t')
        if len(t) >= 8 and len(t[0]) == 4:
            try: RES[t[0].lower()] = (float(t[6]), t[7])
            except ValueError: RES[t[0].lower()] = (np.nan, t[7])
D['res_idx'] = [RES.get(p, (np.nan, ''))[0] for p in D.pdb]
D['method_idx'] = [('EM' if 'ELECTRON' in RES.get(p, (0, ''))[1] else ('X-ray' if 'X-RAY' in RES.get(p, (0, ''))[1] or RES.get(p, (0, ''))[1] == '' else 'other')) for p in D.pdb]
D['method'] = D.method.fillna(D.method_idx)
D['set'] = np.where(D.res_idx < 3.0, 'hires', 'scan')
D = D.sort_values('set').drop_duplicates(['pdb', 'chain'])
D.to_csv(f'{OUT}/tables/fr3_pattern_chains.csv.gz', index=False)
Dc = D[D.contiguous & D.spanned]
Dc = Dc.assign(light=Dc.ct.isin(['K', 'L']))
G = Dc.groupby(['light', 'pattern']).apply(lambda g: pd.Series(dict(
    n=len(g), dis=g.dis3.mean(), n_hires=(g.set == 'hires').sum(), dis_hires=g[g.set == 'hires'].dis3.mean() if (g.set == 'hires').any() else np.nan,
    n_xray_scan=((g.set == 'scan') & (g.method == 'X-ray')).sum(),
    dis_xray_scan=g[(g.set == 'scan') & (g.method == 'X-ray')].dis3.mean() if ((g.set == 'scan') & (g.method == 'X-ray')).any() else np.nan,
    n_em=(g.method == 'EM').sum(), dis_em=g[g.method == 'EM'].dis3.mean() if (g.method == 'EM').any() else np.nan,
    hires_examples=';'.join((g[g.set == 'hires'].pdb + g[g.set == 'hires'].chain + ':' + g[g.set == 'hires'].dis_pos.str.slice(0, 40)).head(4))))).reset_index()
G = G.sort_values('n', ascending=False)
G.to_csv(f'{OUT}/tables/fr3_pattern_rates.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 300); pd.set_option('display.max_colwidth', 120)
print(G[G.n >= 1].head(60).to_string())
