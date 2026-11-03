"""Fetch EMDB validation-analysis JSON (per-residue local resolution, per-chain local resolution / Q-score / atom inclusion)
for all EM error entries and 3 year-matched clean controls per error entry; keep compact extracts only."""
import json, urllib.request, sys, time, os, random
import pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT=f'{SP}/cache/pdb_meta/emdb_va.jsonl'
d=pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables/em_entry_analysis_frame.csv.gz')
d=d[d.emdb_ids.notna()]
random.seed(1)
err=d[d.y==1]; ctl=d[d.y==0]
pick=list(err.pdb)
for yr,g in err.groupby('dep_year'):
    pool=list(ctl[ctl.dep_year==yr].pdb); random.shuffle(pool); pick+=pool[:3*len(g)]
targets={r.pdb:r.emdb_ids.split(';')[0] for r in d[d.pdb.isin(pick)].itertuples()}
done=set()
if os.path.exists(OUT): done={json.loads(l)['pdb'] for l in open(OUT)}
todo=[(p,e) for p,e in targets.items() if p not in done]
print(len(targets),len(todo),file=sys.stderr)
def get(pe):
    p,e=pe; num=e.split('-')[1]
    for a in range(3):
        try:
            raw=json.load(urllib.request.urlopen(f'https://www.ebi.ac.uk/emdb/api/analysis/{e}',timeout=180))
            break
        except urllib.error.HTTPError as ex:
            if ex.code==404: return {'pdb':p,'emdb':e,'status':'404'}
            time.sleep(4)
        except Exception as ex: time.sleep(4)
    else: return {'pdb':p,'emdb':e,'status':'fail'}
    x=raw.get(num) or (list(raw.values())[0] if raw else {})
    out={'pdb':p,'emdb':e,'status':'ok','resolution':(x.get('resolution') or {}).get('value')}
    try:
        k=[kk for kk,v in (x.get('residue_local_resolution') or {}).items() if isinstance(v,dict) and 'data' in v]
        for kk in k:
            v=x['residue_local_resolution'][kk]
            if p in (v.get('name') or '').lower() or len(k)==1:
                out['res_lr']=list(zip(v['data']['residue'],v['data']['localResolution']))
                out['chain_lr']={c:[w.get('value'),w.get('numberOfResidues')] for c,w in (v['data'].get('chainResolution') or {}).items()}
                break
    except Exception as ex: out['lr_err']=str(ex)[:80]
    try:
        q=[v for kk,v in (x.get('qscore') or {}).items() if isinstance(v,dict) and 'data' in v]
        for v in q:
            if p in (v.get('name') or '').lower() or len(q)==1:
                cq=v['data'].get('chainqscore'); cq=dict(cq) if isinstance(cq,list) else cq
                out['chain_q']={c:w.get('value') for c,w in (cq or {}).items()}
                out['res_q']=list(zip(v['data'].get('residue') or [],v['data'].get('qscore') or []))
                break
    except Exception as ex: out['q_err']=str(ex)[:80]
    try:
        h=(x.get('local_res_histogram') or {}).get('masked') or {}
        if h: 
            vals=np.array(h['values']); cnt=np.array(h['counts']); cs=np.cumsum(cnt)/cnt.sum()
            out['map_lr_median']=float(vals[np.searchsorted(cs,0.5)])
    except Exception: pass
    return out
with open(OUT,'a') as f, ThreadPoolExecutor(4) as ex:
    for n,o in enumerate(ex.map(get,todo)):
        f.write(json.dumps(o)+'\n'); f.flush()
        if n%50==0: print(n,o['pdb'],o['status'],file=sys.stderr,flush=True)
