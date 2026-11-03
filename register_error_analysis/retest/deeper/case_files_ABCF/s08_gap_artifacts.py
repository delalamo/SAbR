"""Are some first-occurrence runs deterministic numbering artifacts of rare indels (gap placement), not model errors?

Case-file lead: 2x7l / 9yfu / 9bop / 30ad (C_FR3, 3-residue runs at IMGT 81-84) and 7te9/7teb/8xlk (93-95) all carry a
one-residue FR3 deletion that ANARCI places at IMGT 84 or 96, whereas clean chains with the same-length deletion have it
at 81 or 82-83. 8gse/7eak/9pur (F_CDR1, 27-35, undefined shift) carry a deletion that ANARCI places at IMGT 26.

Test: for every V chain scanned by SAbR (3-5 A scan + the <3 A high-resolution template set), record the ANARCI gap
pattern (positions absent from the IMGT numbering of the *modelled* residues) in a region, only where the modelled
chain is contiguous through the region (no unbuilt residues), and whether SAbR and ANARCI disagree on >=3 residues in
that region. A gap pattern is "deterministic" if chains carrying it disagree >=90% of the time (n>=3), including any
high-resolution (<3 A) chains. First-occurrence runs whose chain carries a deterministic pattern in the run's region
are flagged as probable numbering artifacts.
Regions: FR1 (IMGT 11-26), CDR1 (27-38), CDR2/FR3 start (55-74), FR3 (74-100). Norm gaps per chain type removed
(heavy 10, 73; light 73 and 81-82 for kappa).
Outputs: tables/gap_patterns.csv, tables/gap_artifact_runs.csv
"""
import json, numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF'
REG = {'FR1': (11, 26), 'CDR1': (27, 38), 'CDR2FR3': (55, 74), 'FR3': (74, 100)}
NORM = {'H': {10, 73}, 'K': {10, 73, 81, 82}, 'L': {73}}
rows = []
for fn in ['validation.jsonl', 'new.jsonl', 'snapshot_rest.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        x = json.loads(line); rec = x.get('residues')
        if not rec or x.get('status') != 'ok': continue
        ct = x.get('chain_type') or 'H'
        if ct not in NORM: continue
        for reg, (lo, hi) in REG.items():
            w = [r for r in rec if r[4] is not None and lo - 2 <= r[4] <= hi + 2]
            if len(w) < 4: continue
            si = [r[3] for r in w]
            if max(si) - min(si) + 1 != len(si): continue  # unbuilt residues inside: skip
            pres = {r[4] for r in w}
            plo, phi = min(pres), max(pres)
            if plo > lo or phi < hi: continue  # region not fully spanned
            miss = tuple(k for k in range(lo, hi + 1) if k not in pres and k not in NORM[ct])
            ins = tuple(sorted({f'{r[4]}{r[5]}' for r in w if r[5] and lo <= r[4] <= hi}))
            dis = sum(1 for r in w if lo <= r[4] <= hi and (r[4], r[5] or '') != (r[6], r[7] or ''))
            rows.append(dict(set='hires' if fn.startswith('validation') else 'scan', pdb=x['pdb'], chain=x['chain'], ct=ct, region=reg,
                             pattern=str(miss) + ('+' + ','.join(ins) if ins else ''), dis=dis >= 3))
D = pd.DataFrame(rows)
D['light'] = D.ct.isin(['K', 'L'])
G = D.groupby(['light', 'region', 'pattern']).agg(n=('dis', 'size'), dis_rate=('dis', 'mean'), n_hires=('set', lambda s: (s == 'hires').sum()),
                                                  hires_dis=('dis', lambda s: s[D.loc[s.index, 'set'] == 'hires'].mean() if (D.loc[s.index, 'set'] == 'hires').any() else np.nan)).reset_index()
G['deterministic'] = (G.n >= 3) & (G.dis_rate >= 0.9) & ~(G.hires_dis < 0.9)
G.sort_values(['region', 'n'], ascending=[True, False]).to_csv(f'{OUT}/tables/gap_patterns.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(G[(G.pattern != '()') & (G.n >= 3)].sort_values(['region', 'dis_rate'], ascending=[True, False]).to_string())

# ---------- map first-occurrence runs of modes A,B,C,F to their chain's pattern in the overlapping region
RF = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
amy = set(pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/amyloid_fibril_chains.csv').pdb.str.lower())
RF = RF[~RF.pdb.isin(amy)]
Dk = D.set_index(['pdb', 'chain', 'region'])
Gk = G.set_index(['light', 'region', 'pattern'])
out = []
for r in RF.itertuples():
    regs = [g for g, (lo, hi) in REG.items() if r.imgt0 <= hi and r.imgt1 >= lo]
    flag, pats = False, []
    for g in regs:
        if (r.pdb, r.chain, g) not in Dk.index: continue
        pat = Dk.loc[(r.pdb, r.chain, g)]
        pat = pat.iloc[0] if isinstance(pat, pd.DataFrame) else pat
        k = (r.ctype in 'KL', g, pat.pattern)
        if pat.pattern.startswith('()') and '+' not in pat.pattern: continue
        st = Gk.loc[k]
        pats.append(f'{g}:{pat.pattern} n={int(st.n)} dis={st.dis_rate:.2f} hires={int(st.n_hires)}')
        flag |= bool(st.deterministic)
    out.append(dict(pdb=r.pdb, chain=r.chain, mode=r.mode, imgt0=r.imgt0, imgt1=r.imgt1, delta=r.delta, length=r.length, method=r.method,
                    last_author=r.last_author, paper=r.paper, seqkey=r.seqkey, patterns=' | '.join(pats), artifact_like=flag))
O = pd.DataFrame(out); O.to_csv(f'{OUT}/tables/gap_artifact_runs.csv', index=False)
O['unit'] = O.paper.astype(str) + '|' + O.seqkey.astype(str)
s = O[O['mode'].isin(['A_FR1Nterm', 'B_CDR2FR3', 'C_FR3', 'F_CDR1'])]
print(s.groupby('mode').agg(runs=('artifact_like', 'size'), runs_artifact=('artifact_like', 'sum'),
                            units=('unit', 'nunique'), units_artifact=('unit', lambda u: s.loc[u.index][s.loc[u.index].artifact_like].unit.nunique())))
print(s[s.artifact_like].drop_duplicates('unit')[['pdb', 'chain', 'mode', 'imgt0', 'imgt1', 'delta', 'method', 'last_author', 'patterns']].to_string())
