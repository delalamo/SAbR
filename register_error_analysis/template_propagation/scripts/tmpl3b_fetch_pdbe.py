import sys, os, json, time, urllib.request, pandas as pd
sys.path.insert(0,'/tmp/scripts'); from ca_fetch import parse, CACHE
from concurrent.futures import ThreadPoolExecutor
S=pd.read_pickle('/tmp/tmpl/selected.pkl'); pairs=sorted(set(zip(S.pdb,S.chain)),reverse=True)
def get(pc):
    p,c=pc; fn=f'{CACHE}/{p}_{c}.json'
    if os.path.exists(fn): return
    url=f'https://www.ebi.ac.uk/pdbe/model-server/v1/{p}/atoms?auth_asym_id={c}&label_atom_id=CA&encoding=cif'
    for k in range(3):
        try:
            ca=parse(urllib.request.urlopen(url,timeout=60).read().decode())
            if os.path.exists(fn): return
            tmp=fn+'.tmp_pdbe'; json.dump(ca,open(tmp,'w')); os.replace(tmp,fn); return
        except Exception as e:
            time.sleep(1+2**k)
t0=time.time()
with ThreadPoolExecutor(16) as ex:
    for i,_ in enumerate(ex.map(get,pairs)):
        if i%1000==0: print(i,len(pairs),'%.0fs'%(time.time()-t0),flush=True)
print('done',flush=True)
