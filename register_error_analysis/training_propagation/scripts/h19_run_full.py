"""Predict every unique Fv in fvs.json with ImmuneBuilder (ABodyBuilder2 for VH/VL pairs, NanoBodyBuilder2
for heavy-only domains) and IgFold. Unrefined models. Skips models already written. Logs times.
usage: h19_run_full.py fvs.json outdir {immunebuilder|igfold} [shard k n]"""
import sys, json, os, time, warnings, traceback; warnings.filterwarnings('ignore')
import torch
fvs = json.load(open(sys.argv[1])); out = sys.argv[2]; meth = sys.argv[3]
k, n = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
torch.set_num_threads(int(os.environ.get('NTHREADS', '4')))
os.makedirs(f'{out}/{meth}', exist_ok=True)
log = open(f'{out}/{meth}_times_{k}.csv', 'a')
if meth == 'immunebuilder':
    from ImmuneBuilder import ABodyBuilder2, NanoBodyBuilder2
    AB, NB = ABodyBuilder2(), NanoBodyBuilder2()
else:
    from igfold import IgFoldRunner
    IG = IgFoldRunner()
for i, f in enumerate(fvs):
    if i % n != k: continue
    fn = f"{out}/{meth}/{f['fv_id']}.pdb"
    if os.path.exists(fn): continue
    t = time.time()
    try:
        if meth == 'immunebuilder':
            if f['mode'] == 'nanobody': NB.predict({'H': f['Hseq']}).save_single_unrefined(fn)
            else: AB.predict({'H': f['Hseq'], 'L': f['Lseq']}).save_single_unrefined(fn)
        else:
            seqs = {'H': f['Hseq']} if f['mode'] == 'nanobody' else {'H': f['Hseq'], 'L': f['Lseq']}
            IG.fold(fn, sequences=seqs, do_refine=False, do_renum=False)
        log.write(f"{f['fv_id']},{f['mode']},{time.time()-t:.1f},ok\n")
    except Exception as e:
        log.write(f"{f['fv_id']},{f['mode']},{time.time()-t:.1f},error: {str(e)[:120].replace(',', ';')}\n")
    log.flush()
