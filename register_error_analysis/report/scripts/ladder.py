import sys, numpy as np
sys.path.insert(0,'/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/scripts')
from render import *
def lab_ord(key):
    """ANARCI label -> sequence index order for the query"""
    L=labels(key); items=sorted(L.items(), key=lambda kv: kv[0])
    return items
def ladder(ref, q, lo, hi, excl=None, cut=3.0):
    excl = excl or (lo, hi)
    use=lambda l: not (excl[0]-2 <= l[0] <= excl[1]+2) and not (27<=l[0]<=38 or 56<=l[0]<=65 or 105<=l[0]<=117)
    tq,rms,n=superpose(q, ref, use=use)
    aR=atoms(pdbtext(ref)); aQ=atoms(tq); LR=labels(ref); LQ=labels(q)
    qkeys=sorted([k for k in aQ if 'CA' in aQ[k]])
    qidx={k:i for i,k in enumerate(qkeys)}
    # ANARCI label -> query author key
    qlab={v[1]:k for k,v in LQ.items() if v[1] is not None}
    rows=[]
    for k in sorted(LR):
        aa,la,ls=LR[k]
        if la is None or not (lo<=la[0]<=hi) or 'CA' not in aR.get(k,{}): continue
        d=[(np.linalg.norm(aQ[qk]['CA']-aR[k]['CA']),qk) for qk in qkeys]
        dist,qk=min(d)
        if dist>cut: rows.append(dict(pos=la, ref=AA1.get(aa,'X'), q=None, qlab=None, shift=None, dist=dist)); continue
        qaa,qla,qls=LQ.get(qk,(None,None,None))
        # shift: sequence-index difference between residue found here and the residue ANARCI says belongs here
        sh=None
        if la in qlab and qlab[la] in qidx: sh=qidx[qk]-qidx[qlab[la]]
        rows.append(dict(pos=la, ref=AA1.get(aa,'X'), q=AA1.get(qaa,'X'), qlab=qla, sab=qls, shift=sh, dist=round(dist,1)))
    return rows, rms, n
if __name__=='__main__':
    for ref,q,lo,hi in [('6mlk_L','6c9u_L',1,14),('6mlk_L','7m7f_D',1,14),('7yv1_H','7mkl_H',112,128),('7yv1_L','6bpe_C',1,14),('7yv1_H','7v2a_G',114,128)]:
        rows,rms,n=ladder(ref,q,lo,hi)
        print(ref,q,'rms %.2f n %d'%(rms,n))
        for r in rows: print('  ',r)
