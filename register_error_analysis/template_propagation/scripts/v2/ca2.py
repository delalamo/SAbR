import os, json, time, urllib.request, sys
sys.path.insert(0,'/tmp/scripts'); from ca_fetch import parse
OLD='/tmp/tmpl/ca'; NEW='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/tmpl2/ca'
def get_ca2(p,c):
    for d in (OLD,NEW):
        fn=f'{d}/{p}_{c}.json'
        if os.path.exists(fn):
            try: return json.load(open(fn))
            except Exception: pass
    fn=f'{NEW}/{p}_{c}.json'
    for url in (f'https://www.ebi.ac.uk/pdbe/model-server/v1/{p}/atoms?auth_asym_id={c}&label_atom_id=CA&encoding=cif',
                f'https://models.rcsb.org/v1/{p}/atoms?auth_asym_id={c}&label_atom_id=CA&encoding=cif'):
        for k in range(3):
            try:
                ca=parse(urllib.request.urlopen(url,timeout=60).read().decode())
                tmp=fn+'.tmp'; json.dump(ca,open(tmp,'w')); os.replace(tmp,fn); return ca
            except Exception as e:
                if '404' in str(e) or '400' in str(e): break
                time.sleep(1+2**k)
    return None
