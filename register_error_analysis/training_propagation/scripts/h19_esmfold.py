"""ESMFold (HF transformers) on Fv: H + 25xG linker + L, linker removed from output."""
import sys, json, os, time, torch, warnings; warnings.filterwarnings('ignore')
torch.set_num_threads(4)
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
cases=json.load(open(sys.argv[1])); out=sys.argv[2]; ids=set(sys.argv[3:]); os.makedirs(out,exist_ok=True)
t=time.time()
tok=AutoTokenizer.from_pretrained('facebook/esmfold_v1')
model=EsmForProteinFolding.from_pretrained('facebook/esmfold_v1',low_cpu_mem_usage=True).eval()
model.trunk.set_chunk_size(64)
print('load %.0f s'%(time.time()-t),flush=True)
L='G'*25
for c in cases:
    if ids and c['id'] not in ids: continue
    t=time.time(); seq=c['Hseq']+L+c['Lseq']
    x=tok([seq],return_tensors='pt',add_special_tokens=False)
    with torch.no_grad(): o=model(**x)
    pos=atom14_to_atom37(o['positions'][-1],o); o={k:v.cpu().numpy() for k,v in o.items()}; pos=pos.cpu().numpy()
    n=len(seq); keep=[i for i in range(n) if not (len(c['Hseq'])<=i<len(c['Hseq'])+len(L))]
    chain=[0 if i<len(c['Hseq']) else 1 for i in keep]
    import numpy as np
    p=OFProtein(aatype=o['aatype'][0][keep],atom_positions=pos[0][keep],atom_mask=o['atom37_atom_exists'][0][keep],
                residue_index=np.arange(len(keep))+1,b_factors=o['plddt'][0][keep],chain_index=np.array(chain))
    pdb=to_pdb(p).replace(' A ',' H ')  # chain 0 -> A, 1 -> B
    lines=[]
    for l in to_pdb(p).splitlines():
        if l.startswith('ATOM'): l=l[:21]+('H' if l[21]=='A' else 'L')+l[22:]
        lines.append(l)
    open(f"{out}/{c['id']}.pdb",'w').write('\n'.join(lines)+'\n')
    print('esmfold',c['id'],'%.1f s'%(time.time()-t),flush=True)
