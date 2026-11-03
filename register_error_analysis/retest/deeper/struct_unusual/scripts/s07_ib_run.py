"""Run ImmuneBuilder (ABodyBuilder2 / NanoBodyBuilder2) with the same protocol as the case predictions
(model index 0, unrefined) and also save the ensemble error estimate per residue (sqrt of mean squared deviation
of the 4 models). usage: <ab-env-python> s07_ib_run.py jobs.json outdir"""
import sys, json, os, time, warnings; warnings.filterwarnings('ignore')
import numpy as np, torch
torch.set_num_threads(int(os.environ.get('NTHREADS', '2')))
from ImmuneBuilder import ABodyBuilder2, NanoBodyBuilder2
jobs = json.load(open(sys.argv[1])); out = sys.argv[2]
AB, NB = ABodyBuilder2(), NanoBodyBuilder2()
log = open(f'{out}/times.csv', 'a')
for j in jobs:
    fn = f"{out}/{j['fv_id']}.pdb"
    if os.path.exists(fn):
        continue
    t = time.time()
    try:
        m = NB.predict({'H': j['Hseq']}) if j['mode'] == 'nanobody' else AB.predict({'H': j['Hseq'], 'L': j['Lseq']})
        m.save_single_unrefined(fn)
        np.save(f"{out}/{j['fv_id']}_err.npy", m.error_estimates.mean(0).sqrt().cpu().numpy())
        log.write(f"{j['fv_id']},{j['mode']},{time.time()-t:.1f},ok\n")
    except Exception as e:
        log.write(f"{j['fv_id']},{j['mode']},{time.time()-t:.1f},error: {str(e)[:100].replace(',', ';')}\n")
    log.flush()
