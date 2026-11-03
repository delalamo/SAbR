"""Fetch atoms of selected kappa chains from RCSB ModelServer (cif), to recompute omega angles independently of the
wwPDB cis-peptide annotation. Selection: all EM error kappa chains with Pro8 + random 450 EM clean kappa chains with
Pro8 (seed 1) + 60 X-ray clean kappa chains with Pro8."""
import os, sys, time, urllib.request, pandas as pd
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_pro_kappa8_integrity'
C = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/verify_cis8/coords'
os.makedirs(C, exist_ok=True)
D = pd.read_csv(f'{OUT}/frame.csv.gz')
D = D[(~D.amyloid) & (D.err_first | ~D.err_any) & (D.pro8 == True)]
em = D[D.method == 'EM']
sel = pd.concat([em[em.err_first], em[~em.err_first].sample(450, random_state=1),
                 D[(D.method == 'X-ray') & (~D.err_first)].sample(60, random_state=1)])
sel[['pdb', 'chain', 'method', 'err_first']].to_csv(f'{OUT}/coord_sample.csv', index=False)
for i, (p, c) in enumerate(zip(sel.pdb, sel.chain)):
    fn = f'{C}/{p}_{c}.cif'
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        continue
    url = f'https://models.rcsb.org/v1/{p}/atoms?auth_asym_id={c}&model_nums=1&encoding=cif'
    for a in range(4):
        try:
            data = urllib.request.urlopen(url, timeout=120).read()
            open(fn, 'wb').write(data)
            break
        except Exception as ex:
            print('retry', p, c, ex, file=sys.stderr)
            time.sleep(3 * (a + 1))
    if i % 50 == 0:
        print(i, len(sel), flush=True)
print('done')
