# Build per-entry antigen table for all antibody-containing PDB entries, then competition metrics for the focal 3-5 A set
import json, pickle, re, collections, unicodedata, gzip
import numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
C=f'{SP}/deeper/cache/antigen_competition'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
lab=pickle.load(open(f'{C}/chain_iglabel.pkl','rb'))
seq=pickle.load(open(f'{C}/seqres.pkl','rb'))['chains']
an=json.load(open(f'{C}/anarci_pdbwide.json'))
uni=json.load(open(f'{C}/uniprot_info.json'))
ABDESC=re.compile(r'\b(fab|fv|scfv|nanobody|vhh|sybody|antibody|heavy chain|light chain|igg\d?|immunoglobulin|megabody|legobody|darpin)\b',re.I)
def norm_name(n):
    if not n: return None
    n=unicodedata.normalize('NFKD',n).encode('ascii','ignore').decode().lower()
    sur,_,ini=n.partition(',')
    ini=re.sub(r'[^a-z]','',ini)[:1]
    return re.sub(r'[^a-z]','',sur)+'_'+ini
def is_ig_acc(a):
    u=uni.get(a)
    if not u: return False
    nm=(u.get('name') or ''); g=' '.join(u.get('gene') or []).upper()
    if 'Immunoglobulin' in (u.get('kw') or []): return True
    return bool(re.search(IGNAME,nm) or re.match(r'IG[HKL]([VCGJAMDE]|@)',g))
IGNAME=re.compile(r'\bIG[HKL]@|^Ig[ hkl]|^Ig-like domain-containing|immunoglobulin|immunoblobulin|antibody|\bmab\b|\bfab\b|heavy chain|light chain|kappa|lambda|\bV[HKL]\b|VH coding|^Vk\b|VHCH1|^IG |Epididymis luminal protein 21|^V\d-\d+ protein|uncharacterized|C_region',re.I)
FID_NAME=re.compile(r'guanine nucleotide-binding protein g\(|cytochrome b562|maltose[- ]|maltodextrin|endolysin|green fluorescent protein|immunoglobulin g-binding protein|^protein l$|^protein a$|immunoglobulin-binding protein|streptavidin',re.I)
def is_fid_acc(a):
    u=uni.get(a); nm=(u or {}).get('name') or ''
    return bool(FID_NAME.search(nm)) or a in {'P0ABE7','P0AEX9','P00720','P42212','P63092','P63096','P62873','P59768','B5Z8H1','P42592','Q13043'}
def org_class(pe, accs):
    orgs=pe.get('rcsb_entity_source_organism') or []
    fid_tax={(uni.get(a) or {}).get('taxid') for a in accs if is_fid_acc(a)}-{None}
    if len(orgs)>1 and fid_tax: orgs=[o for o in orgs if o.get('ncbi_taxonomy_id') not in fid_tax] or orgs
    accs=[a for a in accs if not is_fid_acc(a)] or accs
    tax=[o.get('ncbi_taxonomy_id') for o in orgs]; lin=' '.join(' '.join(o.get('taxonomy_lineage') or []) for o in orgs)
    for a in accs:
        u=uni.get(a) or {}
        if u.get('taxid'): tax.append(u['taxid'])
    if 'Viruses' in lin or 'Vira' in lin: return 'viral'
    if 9606 in tax: return 'human'
    if any(t in (10090,10116,9913,9823,9031,9544,9541,10029,9986,9615,9685) for t in tax): return 'other vertebrate'
    if 'Bacteria' in lin: return 'bacterial'
    if 'Eukaryota' in lin or 'Eucarya' in lin: return 'other eukaryote'
    if 32630 in tax: return 'synthetic'
    return 'unknown'
def ag_class(accs, oc, desc, annot):
    if accs and all(is_fid_acc(a) for a in accs): return 'fiducial target'
    accs=[a for a in accs if not is_fid_acc(a)]
    kws=set(); names=[]; ec=False
    for a in accs:
        u=uni.get(a) or {}
        kws|=set(u.get('kw') or []); names.append(u.get('name') or ''); ec|=bool(u.get('ec'))
    txt=(' '.join(names)+' '+(desc or '')+' '+' '.join(annot)).lower()
    if accs and all(is_fid_acc(a) for a in accs): return 'fiducial target'
    if oc=='viral':
        if kws & {'Viral envelope protein','Fusion of virus membrane with host membrane','Viral attachment to host cell','Virion'} and not (kws & {'Capsid protein'}) \
           or re.search(r'spike|glycoprotein|envelope|hemagglutinin|haemagglutinin|\bgp ?\d+|sosip|\bha\d?\b|\bha[12]\b|\brbd\b|receptor[- ]binding domain|pre-?f\b|\bf protein|fusion|attachment|neuraminidase|\bna\b|\benv\b|\bg[hlbcdn]\b|ectodomain|ul\d+|\be protein|\be2?\b|surface|sars-cov-2 [a-z]{1,3}\.?\d',txt):
            return 'viral surface glycoprotein'
        if 'Capsid protein' in kws or re.search(r'capsid|vp1|vp2|vp3',txt): return 'viral capsid'
        return 'viral other'
    if 'G protein-coupled receptor' in kws or re.search(r'g[- ]protein[- ]coupled receptor|7tm',txt): return 'GPCR'
    if kws & {'Ion channel','Ligand-gated ion channel','Voltage-gated channel'} or (('Transport' in kws or 'Symport' in kws or 'Antiport' in kws) and 'Transmembrane' in kws) or re.search(r'transporter|channel|exchanger|pump|atpase.*transport',txt):
        return 'channel/transporter'
    if 'Toxin' in kws or re.search(r'toxin',txt): return 'toxin'
    if kws & {'Cytokine','Growth factor','Hormone','Chemotaxis'} : return 'cytokine/growth factor/hormone'
    if 'Receptor' in kws or (('Transmembrane' in kws or 'Cell membrane' in kws) and 'Glycoprotein' in kws): return 'receptor/membrane protein'
    if ec or kws & {'Hydrolase','Transferase','Oxidoreductase','Lyase','Isomerase','Ligase','Protease','Kinase'}: return 'enzyme'
    if not accs: return 'unmapped/peptide'
    return 'other protein'
rows=[]; ent_info={}
for l in open(f'{C}/gql_entries.jsonl'):
    e=json.loads(l); p=e['rcsb_id'].lower()
    if p in ent_info: continue
    aa=sorted(e.get('audit_author') or [],key=lambda x:x.get('pdbx_ordinal') or 0)
    labn=norm_name(aa[-1]['name']) if aa else None
    dep=(e.get('rcsb_accession_info') or {}).get('deposit_date')
    meth=';'.join(x['method'] for x in (e.get('exptl') or []))
    ents={}
    for pe in e.get('polymer_entities') or []:
        ci=pe['rcsb_polymer_entity_container_identifiers']; eid=ci['entity_id']
        auth=ci.get('auth_asym_ids') or []
        if (pe.get('entity_poly') or {}).get('rcsb_entity_polymer_type')!='Protein': continue
        labs={lab.get(f'{p}_{c}') for c in auth}
        s=seq.get(f'{p}_{auth[0]}') if auth else None
        ct=sorted({x['chain_type'] for x in (an.get(s) or [])}) if s else []
        accs=ci.get('uniprot_ids') or []
        desc=(pe.get('rcsb_polymer_entity') or {}).get('pdbx_description') or ''
        L=(pe.get('entity_poly') or {}).get('rcsb_sample_sequence_length') or 0
        isab=('Ab' in labs) or bool((not labs-{None}) and ABDESC.search(desc) and (not accs or all(is_ig_acc(a) for a in accs)) and 90<=L<=300 and not re.search(r'binding|antigen|receptor|protease|constant|\bfc\b|ase\b',desc,re.I))
        istcr='TCR' in labs and not isab
        isig=(not isab) and ((bool(accs) and all(is_ig_acc(a) for a in accs)) or bool(not accs and re.search(r'immunoglobulin.*constant|\bfc\b|fc fragment|heavy constant',desc,re.I)))
        # antibody fused to a non-Ig protein (e.g. nanobody-receptor fusions): antibody AND antigen
        fus_accs=[a for a in accs if not is_ig_acc(a) and not is_fid_acc(a) and not re.search(r'immunoglobulin|antibody|\bfab\b|heavy chain|light chain|uncharacterized|\big\b',(uni.get(a) or {}).get('name') or '',re.I)]
        isfusion=bool(isab and L>=280 and fus_accs)
        g30=[g['group_id'] for g in (pe.get('rcsb_polymer_entity_group_membership') or []) if g['aggregation_method']=='sequence_identity' and g['similarity_cutoff']==30.0]
        g95=[g['group_id'] for g in (pe.get('rcsb_polymer_entity_group_membership') or []) if g['aggregation_method']=='sequence_identity' and g['similarity_cutoff']==95.0]
        annot=[a['name'] for a in (pe.get('rcsb_polymer_entity_annotation') or []) if a['type']=='Pfam']
        oc=org_class(pe,accs)
        nonig_accs=[a for a in accs if not is_ig_acc(a)]
        keys=nonig_accs if nonig_accs else (['g95_'+g95[0]] if g95 else [])
        if isfusion: keys=fus_accs
        cls=None
        if isfusion: cls=ag_class(fus_accs,oc,desc,annot)
        elif istcr: cls='TCR'
        elif not (isab or isig): cls=ag_class(nonig_accs,oc,desc,annot)
        ents[eid]=dict(eid=eid,auth=auth,isab=bool(isab),istcr=istcr,isig=isig,isfusion=isfusion,ct=ct,accs=nonig_accs,keys=keys,g30=g30[0] if g30 else None,desc=desc,L=L,oc=oc,cls=cls)
    # interfaces (assembly 1)
    area=collections.Counter()
    for a in (e.get('assemblies') or [])[:1]:
        for itf in a.get('interfaces') or []:
            pr=itf.get('rcsb_interface_partner') or []
            if len(pr)!=2: continue
            e1=pr[0]['interface_partner_identifier']['entity_id']; e2=pr[1]['interface_partner_identifier']['entity_id']
            ar=(itf.get('rcsb_interface_info') or {}).get('interface_area') or 0
            if e1!=e2: area[(e1,e2)]+=ar; area[(e2,e1)]+=ar
    has_itf=bool(area)
    abe=[k for k,v in ents.items() if v['isab']]
    # pair light chains to best heavy partner -> antibody groups
    Hs=[k for k in abe if 'H' in ents[k]['ct'] or not ents[k]['ct']]; Ls=[k for k in abe if k not in Hs]
    grp={k:k for k in abe}
    for k in Ls:
        best=max(Hs,key=lambda h:area.get((k,h),0),default=None)
        if best is not None and area.get((k,best),0)>=300: grp[k]=best
    groups=collections.defaultdict(list)
    for k,g in grp.items(): groups[g].append(k)
    agents=[k for k,v in ents.items() if (not (v['isab'] or v['isig'])) or v['isfusion']]
    gres={}
    for g,mem in groups.items():
        cont={x:sum(area.get((m,x),0) for m in mem) for x in agents}
        cont={x:a for x,a in cont.items() if a>=150}
        for m in mem:
            if ents[m]['isfusion']: cont[m]=cont.get(m,0)+1000.0  # pseudo-contact: antigen fused to the antibody
        gres[g]=dict(members=mem,contacts=cont)
    ent_info[p]=dict(lab=labn,dep=dep,meth=meth,ents=ents,groups=gres,has_itf=has_itf)
seen_non=collections.Counter(a for v in ent_info.values() for x in v['ents'].values() if not x['isab'] for a in x['accs'])
nfix=0
for p,v in ent_info.items():
    for x in v['ents'].values():
        if x['isfusion'] and not any(seen_non[a]>0 or 'Transmembrane' in ((uni.get(a) or {}).get('kw') or []) for a in x['keys']):
            x['isfusion']=False; x['cls']=None; nfix+=1
            for g in v['groups'].values():
                g['contacts'].pop(x['eid'],None)
print('fusions demoted',nfix,'fusions kept',sum(x['isfusion'] for v in ent_info.values() for x in v['ents'].values()))
pickle.dump(ent_info,open(f'{C}/ent_info.pkl','wb'))
print('entries',len(ent_info))
