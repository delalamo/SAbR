"""Run ABodyBuilder2 and/or IgFold on cases. usage: h19_run_ab.py cases.json outdir method [ids...]"""
import sys, json, os, time, warnings; warnings.filterwarnings('ignore')
import torch; torch.set_num_threads(4)
cases=json.load(open(sys.argv[1])); out=sys.argv[2]; meth=sys.argv[3]; ids=set(sys.argv[4:])
os.makedirs(out,exist_ok=True)
if meth=='abb2':
    from ImmuneBuilder import ABodyBuilder2; P=ABodyBuilder2()
else:
    from igfold import IgFoldRunner; P=IgFoldRunner()
for c in cases:
    if ids and c['id'] not in ids: continue
    fn=f"{out}/{c['id']}.pdb"
    if os.path.exists(fn): continue
    t=time.time()
    if meth=='abb2':
        P.predict({'H':c['Hseq'],'L':c['Lseq']}).save_single_unrefined(fn)
    else:
        P.fold(fn,sequences={'H':c['Hseq'],'L':c['Lseq']},do_refine=False,do_renum=False)
    print(meth,c['id'],'%.1f s'%(time.time()-t),flush=True)
