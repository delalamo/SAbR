import json, subprocess, numpy as np, re
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
NUM=json.load(open(f'{SP}/report/struct/numbering.json'))
def pdbtext(key): return open(f'{SP}/report/struct/{key}.pdb').read()
def atoms(txt):
    out={}
    for l in txt.splitlines():
        if l.startswith('ATOM'):
            res=(int(l[22:26]),l[26].strip()); out.setdefault(res,{})[l[12:16].strip()]=np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
    return out
def labels(key):
    """author (num,icode) -> (aa3, ANARCI imgt label, SAbR imgt label)"""
    d={}
    for x in NUM[key]['residues']:
        d[(x[0],x[1])]=(x[2],(x[4],x[5] or '') if x[4] is not None else None,(x[6],x[7] or ''))
    return d
def imgt_map(key):
    """ANARCI imgt label -> author residue"""
    return {v[1]:k for k,v in labels(key).items() if v[1] is not None}
def kabsch(P,Q):
    pc,qc=P.mean(0),Q.mean(0); H=(Q-qc).T@(P-pc); U,S,Vt=np.linalg.svd(H); d=np.sign(np.linalg.det(Vt.T@U.T))
    R=Vt.T@np.diag([1,1,d])@U.T; return R,pc,qc
def superpose(mov_key, fix_key, use=lambda lab: True):
    """superpose mov onto fix using CA of residues with matching ANARCI IMGT labels passing use(); return transformed PDB text and RMSD"""
    fm=imgt_map(fix_key); mm=imgt_map(mov_key); fa=atoms(pdbtext(fix_key)); ma=atoms(pdbtext(mov_key))
    common=[l for l in fm if l in mm and use(l) and 'CA' in fa.get(fm[l],{}) and 'CA' in ma.get(mm[l],{})]
    P=np.array([fa[fm[l]]['CA'] for l in common]); Q=np.array([ma[mm[l]]['CA'] for l in common])
    R,pc,qc=kabsch(P,Q); rms=float(np.sqrt((((Q-qc)@R.T+pc-P)**2).sum(1).mean()))
    out=[]
    for l in pdbtext(mov_key).splitlines():
        if l.startswith(('ATOM','HETATM')):
            x=np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])]); y=(x-qc)@R.T+pc
            l=l[:30]+'%8.3f%8.3f%8.3f'%tuple(y)+l[54:]
        out.append(l)
    return '\n'.join(out), rms, len(common)
def render(name, models, w=900, h=620, view='', post=''):
    """models: list of dicts {pdb, js} where js is JS applying styles with variable m (model) and v (viewer)."""
    parts=[]
    for i,m in enumerate(models):
        parts.append(f"var m{i}=v.addModel({json.dumps(m['pdb'])},'{m.get('fmt','pdb')}');\n"+m.get('js','').replace('MODEL',f'm{i}').replace('MID',str(i)))
    html=f'''<html><body style="margin:0;background:#ffffff"><div id="v" style="width:{w}px;height:{h}px;position:relative"></div>
<script src="{SP}/report/lib/3Dmol-min.js"></script><script>
var v=$3Dmol.createViewer('v',{{backgroundColor:'#ffffff',antialias:true}});
{''.join(parts)}
{view}
v.render(); {post} v.render(function(){{window.done=true;}}); setTimeout(function(){{window.done=true}},4000);
</script></body></html>'''
    fn=f'{SP}/report/lib/{name}.html'; open(fn,'w').write(html)
    r=subprocess.run(['node',f'{SP}/scripts/shot.js',fn,f'{SP}/report/img/{name}.png',str(w),str(h)],capture_output=True,text=True)
    if r.returncode: print(r.stdout,r.stderr)
    return f'{SP}/report/img/{name}.png'
AA1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def frame_from(txt_key_or_text, strand_labels, key_for_labels):
    """rotation so that strand (list of IMGT labels) runs along +x and the domain centroid lies along -y"""
    txt=txt_key_or_text if '\n' in txt_key_or_text else pdbtext(txt_key_or_text)
    a=atoms(txt); m=imgt_map(key_for_labels)
    ca=np.array([a[m[l]]['CA'] for l in strand_labels if l in m and 'CA' in a[m[l]]])
    allca=np.array([v['CA'] for v in a.values() if 'CA' in v])
    x=ca[-1]-ca[0]; x/=np.linalg.norm(x)
    up=ca.mean(0)-allca.mean(0); up-=up.dot(x)*x; up/=np.linalg.norm(up)
    z=np.cross(x,up); R=np.vstack([x,up,z]); return R, ca.mean(0)
def apply(txt,R,c):
    out=[]
    for l in txt.splitlines():
        if l.startswith(('ATOM','HETATM')):
            p=np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])]); q=R@(p-c)
            l=l[:30]+'%8.3f%8.3f%8.3f'%tuple(q)+l[54:]
        out.append(l)
    return '\n'.join(out)
def label_js(txt,key,imgt_labels,color,size=15,offset=1.6,suffix=''):
    a=atoms(txt); L=labels(key); m=imgt_map(key); js=''
    for lab in imgt_labels:
        if lab not in m or 'CA' not in a[m[lab]]: continue
        r=a[m[lab]]; ca=r['CA']; d=(r['CB']-ca) if 'CB' in r else np.array([0,1.0,0]); d=d/np.linalg.norm(d)
        p=ca+d*offset+np.array([0,0,2.0])
        txtl=AA1.get(L[m[lab]][0],'X')+str(lab[0])+lab[1]+suffix
        js+=f"v.addLabel('{txtl}',{{position:{{x:{p[0]:.2f},y:{p[1]:.2f},z:{p[2]:.2f}}},fontSize:{size},fontColor:'{color}',backgroundColor:'white',backgroundOpacity:0.85,borderThickness:0,inFront:true,alignment:'center',font:'Helvetica'}});"
    return js
def resi_sel(key,imgt_labels):
    m=imgt_map(key); return [m[l][0] for l in imgt_labels if l in m]
def label_js2(txt,key,imgt_labels,color,size=16,dy=3.2,dz=1.5,fmt=None):
    a=atoms(txt); L=labels(key); m=imgt_map(key); js=''
    for lab in imgt_labels:
        if lab not in m or 'CA' not in a[m[lab]]: continue
        ca=a[m[lab]]['CA']; p=ca+np.array([0,dy,dz])
        t=(fmt or (lambda aa,l: f'{aa}{l[0]}{l[1]}'))(AA1.get(L[m[lab]][0],'X'),lab)
        js+=f"v.addLabel('{t}',{{position:{{x:{p[0]:.2f},y:{p[1]:.2f},z:{p[2]:.2f}}},fontSize:{size},fontColor:'{color}',backgroundColor:'white',backgroundOpacity:0.9,borderThickness:1,borderColor:'{color}',inFront:true,alignment:'center',font:'Helvetica'}});"
    return js
def vresi(key): return [k[0] for k,v in labels(key).items() if v[1] is not None]
