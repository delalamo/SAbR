"""Validate the ABangle reimplementation against published All_Angles.dat values (cached structures)."""
import pandas as pd, numpy as np
from anarci import anarci
import common
a = pd.read_csv(f'{common.C}/abangle_src/ABangle/data/All_Angles.dat', sep='\t')
codes = ['3IDY_HL', '1KEN_HL', '2X7L_AB', '3IVK_HL', '1N8Z_BA', '2FGW_HL', '3DVG_BA', '2A9N_IM', '3IVK_AB', '2X7L_GI']
rows = []
for code in codes:
    p, hc, lc = code[:4].lower(), code[5], code[6]
    at = common.read_atoms(common.cif_path(p), chains={hc, lc})
    if hc not in at or lc not in at:
        print('skip', code); continue
    ca = {}
    cts = {}
    for ch in (hc, lc):
        res = [r for r in at[ch] if 'CA' in r[3] and r[2] in common.THREE]
        seq = ''.join(common.THREE[r[2]] for r in res)
        num, det, _ = anarci([('x', seq)], scheme='imgt', assign_germline=False, output=False)
        n = num[0][0]; st = n[1]; k = st; m = {}
        for (pos, ins), aa in n[0]:
            if aa != '-':
                if ins.strip() == '':
                    m[pos] = res[k][3]['CA']
                k += 1
        ca[ch] = m; cts[ch] = det[0][0]['chain_type']
    r = common.abangle(ca[hc], ca[lc], cts[lc])
    ref = a[a.Code == code].iloc[0]
    rows.append(dict(code=code, **{k: round(r[k], 2) for k in ['HL', 'HC1', 'LC1', 'HC2', 'LC2', 'dc']},
                     **{'ref_' + k: ref[k] for k in ['HL', 'HC1', 'LC1', 'HC2', 'LC2', 'dc']}))
df = pd.DataFrame(rows)
for k in ['HL', 'HC1', 'LC1', 'HC2', 'LC2', 'dc']:
    df['d_' + k] = (df[k] - df['ref_' + k]).round(2)
pd.set_option('display.width', 250)
print(df[['code'] + [c for c in df if c.startswith('d_')]])
df.to_csv(f'{common.TAB}/abangle_validation.csv', index=False)
