"""List every SAbR-scanned chain (3-5 A scan and <3 A validation set) whose detector ANARCI numbering of the modelled
residues has a non-standard heavy FR3 gap (one or two absent positions in 74-100 other than 81/82-83/83-84), with the
disagreement window. Checks the s09 claim that the gap-at-96 pattern agrees at high resolution, and looks for
high-resolution members of the gap-89 / gap-94 / gap-94+95 / gap-85 patterns that carry the remaining 'genuine' cases.
Output tables/v03_pattern_members.csv"""
import json, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_FR3_INDEL_C_integrity'
rows = []
for fn in ['validation', 'new', 'snapshot_rest']:
    for line in open(f'{SP}/sabr/out/{fn}.jsonl'):
        x = json.loads(line); rec = x.get('residues')
        if not rec or x.get('status') != 'ok' or x.get('chain_type') != 'H': continue
        an = [r for r in rec if r[4] is not None]
        if not an: continue
        pres = {r[4] for r in an}
        lo, hi = min(pres), max(pres)
        if lo > 74 or hi < 100: continue
        miss = tuple(k for k in range(74, 101) if k not in pres)
        ins = tuple(sorted({f'{r[4]}{r[5]}' for r in an if r[5] and 74 <= r[4] <= 100}))
        if not miss and not ins: continue
        if miss in [(81,), (82, 83), (83, 84), (83,), (82,), (81, 82)] and not ins: continue
        dis = [r for r in an if 66 <= r[4] <= 117 and (r[4], r[5] or '') != (r[6], r[7] or '')]
        # contiguous unbuilt check in 74-100
        si = [r[3] for r in an if 74 <= r[4] <= 100]
        unbuilt = (max(si) - min(si) + 1 - len(si)) if si else None
        rows.append(dict(set='hires(<3A)' if fn == 'validation' else 'scan(3-5A)', pdb=x['pdb'], chain=x['chain'], gaps=str(miss), ins=','.join(ins),
                         n_dis_66_117=len(dis), dis_window=(f'{min(r[4] for r in dis)}-{max(r[4] for r in dis)}' if dis else ''),
                         dis_detail=' '.join(f'{r[2][:1]}{r[4]}{r[5] or ""}/{r[6]}{r[7] or ""}' for r in dis)[:200], unbuilt_74_100=unbuilt))
D = pd.DataFrame(rows).sort_values(['gaps', 'ins', 'set'])
D.to_csv(f'{OUT}/tables/v03_pattern_members.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 120); pd.set_option('display.max_rows', 400)
print(D[D.ins == ''].to_string())
