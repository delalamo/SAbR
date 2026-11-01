import urllib.request, time, os, json, gzip
CACHE='/tmp/tmpl/ca'; os.makedirs(CACHE,exist_ok=True)
def parse(txt):
    cols=[]; rows=[]; inloop=False
    for line in txt.splitlines():
        if line.startswith('_atom_site.'): cols.append(line.split('.',1)[1].strip()); inloop=True; continue
        if inloop:
            if line.startswith('#') or line.startswith('loop_') or line.startswith('_'):
                if rows: break
                continue
            rows.append(line.split())
    if not cols: return []
    ci={c:i for i,c in enumerate(cols)}
    out=[]; seen=set()
    for r in rows:
        if len(r)!=len(cols): continue
        if r[ci.get('pdbx_PDB_model_num',0)]!='1' and 'pdbx_PDB_model_num' in ci: continue
        sid=r[ci['label_seq_id']]
        if sid in ('.','?') or sid in seen: continue
        seen.add(sid)
        out.append((int(sid),float(r[ci['Cartn_x']]),float(r[ci['Cartn_y']]),float(r[ci['Cartn_z']])))
    return out
def get_ca(pdb,chain):
    fn=f'{CACHE}/{pdb}_{chain}.json'
    if os.path.exists(fn): return json.load(open(fn))
    url=f'https://models.rcsb.org/v1/{pdb}/atoms?auth_asym_id={chain}&label_atom_id=CA&encoding=cif'
    for k in range(4):
        try:
            txt=urllib.request.urlopen(url,timeout=60).read().decode()
            ca=parse(txt); json.dump(ca,open(fn,'w')); return ca
        except Exception as e:
            if '404' in str(e) or '400' in str(e): json.dump([],open(fn,'w')); return []
            time.sleep(1+2**k)
    return None
