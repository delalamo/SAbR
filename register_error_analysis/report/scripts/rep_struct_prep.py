"""Fetch chains, number them (ANARCI + SAbR via the scan pipeline), and save per-chain PDB text + residue labels."""
import sys, io, json, gzip, urllib.request, time
sys.path.insert(0,'/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/scripts')
from anarci_num import number_seqs
import sabr_scan_multi as S
from Bio.PDB import MMCIFParser, PDBIO, Select
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'; OUT=f'{SP}/report/struct'
chains=[('6mlk','L'),('6mlk','H'),('6c9u','L'),('7m7f','D'),('7mxl','B'),('7mxl','K'),('7mxl','A'),('7v2a','G'),('7v2a','H'),('5zs0','B'),('7mkl','H'),('6bpe','C'),('7yv1','H'),('7yv1','L')]
num={}
for p,c in chains:
    maps=S.gql_maps(p); amap,seq=maps[c]
    doms=number_seqs([seq],f'{SP}/sabr/anarci_cache.json')[seq]
    rec=S.process([p,[c],{c:doms}])[0] if doms else {'status':'no V domain','residues':[]}
    txt=S.fetch(f'https://www.ebi.ac.uk/pdbe/model-server/v1/{p}/atoms?auth_asym_id={c}&encoding=cif')
    st=MMCIFParser(QUIET=True).get_structure(p,io.StringIO(txt))
    model=next(iter(st)); ch=model[c]
    # write PDB text with chain renamed to single letter
    for r in list(ch):
        if r.id[0]!=' ': ch.detach_child(r.id)
    ch.id='A'
    io_=PDBIO(); io_.set_structure(ch); buf=io.StringIO(); io_.save(buf)
    open(f'{OUT}/{p}_{c}.pdb','w').write(buf.getvalue())
    num[f'{p}_{c}']=dict(status=rec.get('status'),residues=rec.get('residues',[]))
    print(p,c,rec.get('status'),len(rec.get('residues',[])),flush=True)
json.dump(num,open(f'{OUT}/numbering.json','w'))
# context complexes (whole assemblies, protein only)
for p in ['7cwu','8itf']:
    data=urllib.request.urlopen(f'https://files.rcsb.org/download/{p.upper()}.cif.gz',timeout=120).read()
    open(f'{OUT}/{p}.cif','w').write(gzip.decompress(data).decode()); print(p,'cif saved')
