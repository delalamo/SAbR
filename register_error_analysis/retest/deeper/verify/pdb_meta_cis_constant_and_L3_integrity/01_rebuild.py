"""Independent rebuild of constant-domain and Pro115 cis status from raw caches (not the investigator's chain table).
V-region boundaries come from chains.csv vseq located in SEQRES (not from SAbR seq_index); canonical constant-domain
cis sites are located by sequence motif (Ck 'FYP', CH1 'FPEP', Cl 'FYP'); amyloid excluded by the investigator's list."""
import json, re, collections, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_constant_and_L3_integrity'
T='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz')
amy=set(pd.read_csv(f'{T}/amyloid_fibril_chains.csv').pdb)
rx=set(ch[ch.title.fillna('').str.contains(r'amyloid(?!-beta)',case=False)].pdb)
print('amyloid list',len(amy),'regex',len(rx),'regex-only',sorted(rx-amy)[:20],'list-only',sorted(amy-rx))
feat={}
dup=0
for l in open(f'{SP}/deeper/cache/pdb_meta/cis_sheet.jsonl'):
    x=json.loads(l); k=(x['rcsb_id'].split('.')[0].lower(),x['auth'])
    if k in feat: dup+=1
    feat[k]=x['feats']
print('feature records',len(feat),'dups',dup)
# IMGT numbering (ANARCI, sequence-based) from SAbR output for Pro115 and V end
imgt={}
for fn in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        r=json.loads(l); k=(r['pdb'],r['chain'])
        if k in imgt: continue
        imgt[k]=[x for x in (r.get('residues') or []) if x[3] is not None]
rows=[]; nomatch=0; nofeat=0
for c in ch.itertuples():
    k=(c.pdb,c.chain)
    fs=feat.get(k)
    if fs is None: nofeat+=1; continue
    seq=c.seq if isinstance(c.seq,str) else ''; v=c.vseq if isinstance(c.vseq,str) else ''
    vs=seq.find(v) if v else -1
    if vs<0: nomatch+=1; continue
    ve=vs+len(v)-1
    cis=[]; unobs=set(); cis_raw=[]
    for f in fs:
        for p in f['feature_positions']:
            b=p['beg_seq_id']; e=p.get('end_seq_id') or b
            if f['type']=='CIS-PEPTIDE': cis.append(e-1); cis_raw.append((b,e))
            elif f['type']=='UNOBSERVED_RESIDUE_XYZ': unobs.update(range(b-1,e))
    modeled=[i for i in range(len(seq)) if i not in unobs]
    const_mod=[i for i in modeled if i>ve+3]
    # canonical constant-domain motif sites (Pro index)
    cseq=seq[ve+1:]
    sites=[]
    if c.ctype in 'KL':
        for m in re.finditer(r'(?=FYP)',cseq): sites.append(ve+1+m.start()+2)
    else:
        for m in re.finditer(r'(?=F[PA]EP)',cseq): sites+= [ve+1+m.start()+1, ve+1+m.start()+3]
    sites=sites[:2] if c.ctype=='H' else sites[:1]
    site_mod=[s for s in sites if s in modeled and s-1 in modeled]
    # imgt map
    rec=imgt.get(k,[])
    im={x[3]:(x[4],x[5],x[2],x[6],x[7]) for x in rec}
    p115=[i for i,(a,ins,rn,sa,si) in im.items() if a==115 and not ins and rn=='PRO']
    vmax_sabr=max([i for i,(a,*_) in im.items() if a is not None],default=None)
    rows.append(dict(pdb=c.pdb,chain=c.chain,ctype=c.ctype,method=c.method,res=c.res,deposit=c.deposit,paper=c.paper,unit=c.unit,
        last_author=c.last_author,err_first=c.err_first,err_any=c.err_any,modes=c.err_modes,sars2=c.sars2,irm=c.irm,
        amyloid=c.pdb in amy, amy_rx=c.pdb in rx, vs=vs,ve=ve,vmax_sabr=vmax_sabr,seqlen=len(seq),
        n_cis_chain=len(cis), n_cis_v=sum(vs<=i<=ve for i in cis), n_cis_const=sum(i>ve+3 for i in cis),
        cis_nonpro=sum(1 for i in cis if i<len(seq) and seq[i]!='P'),
        n_const_mod=len(const_mod), const_pro_mod=sum(seq[i]=='P' for i in const_mod),
        site_n=len(sites), site_mod_n=len(site_mod), site_cis_n=sum(s in cis for s in site_mod),
        site_idx=';'.join(map(str,sites)),
        pro115=len(p115)>0, pro115_mod=len(p115)>0 and (p115[0] not in unobs), cis115=len(p115)>0 and p115[0] in cis,
        const_cis_pos=';'.join(str(i-ve) for i in cis if i>ve+3)))
D=pd.DataFrame(rows)
print('chains',len(ch),'built',len(D),'no features',nofeat,'vseq not in seq',nomatch)
print('vmax_sabr==ve',(D.vmax_sabr==D.ve).mean(),' |diff|<=2',((D.vmax_sabr-D.ve).abs()<=2).mean())
D.to_csv(f'{OUT}/frame.csv.gz',index=False)
