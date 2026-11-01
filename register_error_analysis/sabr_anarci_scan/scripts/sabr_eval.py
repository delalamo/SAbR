import json, sys, numpy as np, pandas as pd
FR=[(1,26),(39,55),(66,104),(118,128)]
def region(n):
    if n is None: return None
    for a,b in FR:
        if a<=n<=b: return 'FR'
    if 105<=n<=117: return 'CDR3'
    return 'CDR'
def load(fn):
    recs=[json.loads(l) for l in open(fn)]
    return recs
def segments(res,minlen=3,fr_only=False,skip_cdr3=False):
    """res: list of [auth,icode,resname,seqidx,a_num,a_ins,s_num,s_ins]; return deviation segments over V-region residues."""
    v=[r for r in res if r[4] is not None]
    segs=[]; cur=[]
    for r in v:
        reg=region(r[4])
        dev=(r[4],r[5] or '')!=(r[6],r[7] or '')
        if skip_cdr3 and reg=='CDR3': dev=False
        if fr_only and reg!='FR': dev=False
        if dev: cur.append(r)
        else:
            if len(cur)>=minlen: segs.append(cur)
            cur=[]
    if len(cur)>=minlen: segs.append(cur)
    out=[]
    for s in segs:
        d=[r[6]-r[4] for r in s if region(r[4])=='FR' and not (r[5] or r[7])]
        out.append(dict(start=s[0][0],end=s[-1][0],n=len(s),delta=int(np.median(d)) if d else None,regions=sorted({region(r[4]) for r in s})))
    return out
if __name__=='__main__':
    recs=load(sys.argv[1])
    st=pd.Series([r['status'] for r in recs]).value_counts(); print(st.to_string())
