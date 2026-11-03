# Per-entry antigen, binding status, antigen class, and competition metrics; per-antibody (group) antigen role
import json, pickle, re, collections, bisect
import numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
C=f'{SP}/deeper/cache/antigen_competition'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
E=pickle.load(open(f'{C}/ent_info.pkl','rb'))
uni=json.load(open(f'{C}/uniprot_info.json'))
# ---- families: union-find over antigen keys <-> 30% sequence-identity groups (single-accession entities only)
par={}
def f(x):
    par.setdefault(x,x)
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
def u(a,b): par[f(a)]=f(b)
for p,v in E.items():
    for x in v['ents'].values():
        if (x['isab'] and not x['isfusion']) or x['isig'] or x['cls']=='fiducial target': 
            for k in x['keys']: f(k)
            continue
        for k in x['keys']: f(k)
        if len(x['keys'])==1 and x['g30']: u(x['keys'][0],'g30_'+x['g30'])
fam=lambda k: f(k)
# ---- per entry: contacted antigens, target, binding status
def dt(s): return pd.Timestamp(s[:10]) if s else pd.NaT
present=collections.defaultdict(set); contact=collections.defaultdict(set)
erow={}
for p,v in E.items():
    ents=v['ents']; groups=v['groups']
    cont_area=collections.Counter(); grp_role={}
    for g,gi in groups.items():
        cl=[ents[x]['cls'] for x in gi['contacts']]
        grp_role[g]='no antigen contact' if not cl else ('fiducial' if all(c=='fiducial target' for c in cl) else 'subject')
        for x,a in gi['contacts'].items(): cont_area[x]+=a
    nonab=[k for k,x in ents.items() if (not (x['isab'] or x['isig'])) or x['isfusion']]
    nonfid_cont=[x for x in cont_area if ents[x]['cls']!='fiducial target']
    if nonfid_cont:
        tgt=max(nonfid_cont,key=lambda x:cont_area[x]); status='bound'
    elif cont_area:
        cand=[k for k in nonab if ents[k]['cls'] not in ('fiducial target',) and ents[k]['L']>=30]
        tgt=max(cand,key=lambda k:ents[k]['L']) if cand else max(cont_area,key=lambda x:cont_area[x]); status='fiducial-only'
    elif nonab:
        cand=[k for k in nonab if ents[k]['L']>=30 and ents[k]['cls']!='fiducial target'] or nonab
        tgt=max(cand,key=lambda k:ents[k]['L']); status='no contact' if v['has_itf'] else 'no interface data'
    else:
        tgt=None; status='unbound'
    keys=ents[tgt]['keys'] if tgt else []
    erow[p]=dict(pdb=p,lab=v['lab'],date=dt(v['dep']),meth=v['meth'],status=status,tgt_keys=keys,tgt_fams=sorted({fam(k) for k in keys}),
                 tgt_cls=ents[tgt]['cls'] if tgt else 'none',tgt_org=ents[tgt]['oc'] if tgt else 'none',tgt_desc=ents[tgt]['desc'] if tgt else '',
                 tgt_name=('; '.join((uni.get(k) or {}).get('name') or '' for k in keys if not k.startswith('g95_'))) if keys else '',
                 n_groups=len(groups),grp_role=grp_role,
                 roles=collections.Counter(grp_role.values()))
    for k in {k for x in nonab for k in ents[x]['keys']}: present[k].add(p)
    for k in {k for x in cont_area for k in ents[x]['keys']}: contact[k].add(p)
famp=collections.defaultdict(set); famc=collections.defaultdict(set)
for k,s in present.items(): famp[fam(k)]|=s
for k,s in contact.items(): famc[fam(k)]|=s
fsz=collections.Counter({k:len(s) for k,s in famp.items()})
print('largest families (n entries):'); 
for k,n in fsz.most_common(12):
    names=collections.Counter(erow[p]['tgt_name'][:40] for p in famp[k]).most_common(3); print(' ',k,n,names)
# ---- SIFTS: all PDB entries per accession with deposition dates
idx={}
for l in open('/tmp/supp/pdbwide/entries.idx').read().splitlines()[2:]:
    q=l.split('\t')
    if len(q)<3: continue
    try: idx[q[0].lower()]=pd.Timestamp(pd.to_datetime(q[2],format='%m/%d/%y'))
    except Exception: pass
up=pd.read_csv('/tmp/supp/pdbwide/pdb_chain_uniprot.tsv.gz',sep='\t',comment='#',usecols=['PDB','SP_PRIMARY']).drop_duplicates()
acc_all=collections.defaultdict(list)
for p,a in zip(up.PDB,up.SP_PRIMARY):
    if p in idx: acc_all[a].append(idx[p])
for a in acc_all: acc_all[a]=np.array(sorted(acc_all[a]),dtype='datetime64[D]')
def count_labs(pset, p, d, lab, days):
    labs=set(); n=0
    for q in pset:
        if q==p: continue
        r=erow[q]
        if pd.isna(r['date']) or abs((r['date']-d).days)>days: continue
        n+=1
        if r['lab']!=lab: labs.add(r['lab'])
    return len(labs), n
def metrics(p):
    r=erow[p]; d=r['date']; lab=r['lab']; keys=r['tgt_keys']; fams=r['tgt_fams']
    if not keys or pd.isna(d): return {}
    P=set().union(*[present[k] for k in keys]); Cc=set().union(*[contact[k] for k in keys])
    FP=set().union(*[famp[k] for k in fams]); FC=set().union(*[famc[k] for k in fams])
    out={}
    for nm,S in [('acc',P),('acc_contact',Cc),('fam',FP),('fam_contact',FC)]:
        for days,lbl in [(30,'1m'),(183,'6m'),(365,'12m')]:
            out[f'labs_{nm}_{lbl}'],out[f'ents_{nm}_{lbl}']=count_labs(S,p,d,lab,days)
        prior=[erow[q]['date'] for q in S if q!=p and not pd.isna(erow[q]['date']) and erow[q]['date']<d]
        out[f'prev12_{nm}']=sum((d-x).days<=365 for x in prior)
        out[f'first_{nm}']=int(not any((d-x).days>30 for x in prior))
        out[f'age_days_{nm}']=(d-min(prior)).days if prior else 0
        # other labs released ab structure before this deposit within past 12 months
        out[f'prev12_otherlab_{nm}']=len({erow[q]['lab'] for q in S if q!=p and not pd.isna(erow[q]['date']) and 0<(d-erow[q]['date']).days<=365 and erow[q]['lab']!=lab})
    dd=np.datetime64(d.date(),'D')
    accs=[k for k in keys if not k.startswith('g95_')]
    if accs:
        arr=np.concatenate([acc_all.get(a,np.array([],dtype='datetime64[D]')) for a in accs])
        out['prev12_allpdb']=int(((arr<dd)&(arr>=dd-365)).sum()); out['first_allpdb']=int(not ((arr<dd-30)).any())
    return out
rows=[]
for p,r in erow.items():
    m=metrics(p)
    rows.append(dict(pdb=p,lab=r['lab'],date=r['date'],ab_status=r['status'],tgt_keys='|'.join(r['tgt_keys']),tgt_fam='|'.join(r['tgt_fams']),
        tgt_cls=r['tgt_cls'],tgt_org=r['tgt_org'],tgt_name=r['tgt_name'],tgt_desc=r['tgt_desc'],n_ab_groups=r['n_groups'],
        n_subject_groups=r['roles'].get('subject',0),n_fiducial_groups=r['roles'].get('fiducial',0),n_nocontact_groups=r['roles'].get('no antigen contact',0),**m))
A=pd.DataFrame(rows)
A.to_csv(f'{OUT}/tables/pdbwide_ab_entries_antigen.csv.gz',index=False)
print(A.ab_status.value_counts().to_string()); print(A.tgt_cls.value_counts().to_string())
# ---- per antibody group (for focal V chains)
G=[]
for p,v in E.items():
    for g,gi in v['groups'].items():
        role=erow[p]['grp_role'][g]
        cl=sorted({v['ents'][x]['cls'] for x in gi['contacts']})
        main=max(gi['contacts'],key=lambda x:gi['contacts'][x]) if gi['contacts'] else None
        for m in gi['members']:
            for c in v['ents'][m]['auth']:
                G.append(dict(pdb=p,chain=c,ab_group=g,ab_role_contact=role,ab_ag_cls=v['ents'][main]['cls'] if main else 'none',ab_ag_keys='|'.join(v['ents'][main]['keys']) if main else '',
                              ab_ag_area=gi['contacts'].get(main,0) if main else 0))
pd.DataFrame(G).to_csv(f'{OUT}/tables/pdbwide_ab_chain_roles.csv.gz',index=False)
print('chains',len(G))
