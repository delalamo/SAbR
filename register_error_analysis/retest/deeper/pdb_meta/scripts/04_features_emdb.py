"""Extract EMDB-entry features (maps, half maps, masks, focused/consensus/composite maps, post-processing, pixel/box size,
processing software, startup model, specimen) from the cached EMDB REST JSON, then aggregate to PDB entries."""
import json, gzip, glob, re, os, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
def walk(x,p=''):
    if isinstance(x,dict):
        for k,v in x.items(): yield from walk(v,p+'.'+k)
    elif isinstance(x,list):
        for v in x: yield from walk(v,p+'[]')
    else: yield p,x
def fl(x):
    try: return float(x)
    except: return np.nan
SKIP=re.compile(r'sequence|citation|author|\.file$|\.label$|grant|funding|statistics|\.cell\.|axis_order|data_type|instance_type|\.format$')
KW={'kw_local':r'local refine|locally refined|local[- ]refinement|focus(?:ed)? refine|focused map|focused (?:3d )?class|masked refine|signal subtract|particle subtract|local map',
    'kw_composite':r'composite','kw_consensus':r'consensus','kw_deepemhancer':r'deep ?emhancer','kw_densmod':r'density[- ]modif|resolve_cryo_em|phenix\.resolve',
    'kw_locscale':r'locscale|loc_scale|local[- ]?sharpen|localdeblur|local deblur|emready|em-ready','kw_sharpen':r'sharpen','kw_unsharp':r'unsharpen|non-?sharpen|unfiltered|raw map',
    'kw_nu':r'non[- ]?uniform','kw_symexp':r'symmetry[- ]expan|expanded particles','kw_3dflex':r'3d ?flex|3dva|variability|cryodrgn|multi-?body'}
rows=[]
for f in sorted(glob.glob(f'{SP}/cache/pdb_meta/emdb/entry_EMD-*.json.gz')):
    d=json.load(gzip.open(f)); eid=d.get('emdb_id'); r={'emdb':eid}
    pairs=list(walk(d))
    txt=' || '.join(str(v) for p,v in pairs if isinstance(v,str) and not SKIP.search(p)).lower()
    for k,pat in KW.items(): r[k]=bool(re.search(pat,txt))
    adm=d.get('admin',{})
    r['emdb_deposit_site']=(adm.get('sites') or {}).get('deposition'); r['emdb_proc_site']=(adm.get('sites') or {}).get('last_processing')
    r['composite_flag']=str(adm.get('composite_map'))=='True'
    kd=adm.get('key_dates') or {}; r['map_release']=kd.get('map_release'); r['emdb_deposition']=kd.get('deposition')
    mp=d.get('map') or {}
    r['map_pix']=fl(((mp.get('pixel_spacing') or {}).get('x') or {}).get('valueOf_'))
    r['map_box']=fl((mp.get('dimensions') or {}).get('col'))
    r['map_box_A']=r['map_pix']*r['map_box'] if r['map_box']==r['map_box'] else np.nan
    r['map_annot']=(mp.get('annotation_details') or '')[:300]
    r['map_contour_src']=';'.join(sorted({str(c.get('source')) for c in ((mp.get('contour_list') or {}).get('contour') or [])}))
    it=d.get('interpretation') or {}
    hm=(it.get('half_map_list') or {}).get('half_map') or []; am=(it.get('additional_map_list') or {}).get('additional_map') or []
    sg=(it.get('segmentation_list') or {}).get('segmentation') or []
    r['n_half']=len(hm); r['n_add']=len(am); r['n_mask']=len(sg)
    r['add_annot']=' | '.join((a.get('annotation_details') or '') for a in am)[:600]
    xr=d.get('crossreferences') or {}
    el=(xr.get('emdb_list') or {}).get('emdb_reference') or []
    rels=[((e.get('relationship') or {}).get('other') or (e.get('relationship') or {}).get('in_frame') or 'NA') for e in el]
    r['n_emdb_ref']=len(el); r['n_focused_ref']=sum('focused' in x for x in rels); r['n_consensus_ref']=sum('consensus' in x for x in rels)
    r['n_pdb_ref']=len((xr.get('pdb_list') or {}).get('pdb_reference') or [])
    sd=((d.get('structure_determination_list') or {}).get('structure_determination') or [{}])[0]
    ip=(sd.get('image_processing') or [{}])[0]
    fr=ip.get('final_reconstruction') or {}
    r['n_images_used']=fl(fr.get('number_images_used'))
    r['final_sw']=';'.join(sorted({s.get('name') or '' for s in ((fr.get('software_list') or {}).get('software') or [])}))
    r['final_sw_ver']=';'.join(sorted({f"{s.get('name')}={s.get('version')}" for s in ((fr.get('software_list') or {}).get('software') or [])}))
    r['applied_sym']=((fr.get('applied_symmetry') or {}).get('point_group'))
    r['final_details']=(fr.get('details') or '')[:300]
    sm=ip.get('startup_model') or [{}]
    r['startup']=';'.join(sorted({(s.get('type_of_model') or 'NA') for s in sm}))
    r['fa_type']=((ip.get('final_angle_assignment') or {}).get('type_'))
    ps=ip.get('particle_selection') or []
    r['n_picked']=sum(fl(p.get('number_selected')) for p in ps if p.get('number_selected') is not None) if ps else np.nan
    r['n_proc_steps']=len(sd.get('image_processing') or [])
    mic=((sd.get('microscopy_list') or {}).get('microscopy') or [{}])[0]
    r['energy_filter']=bool(((mic.get('specialist_optics') or {}).get('energy_filter')))
    spl=((sd.get('specimen_preparation_list') or {}).get('specimen_preparation') or [{}])[0]
    r['grid_model']=((spl.get('grid') or {}).get('model'))
    sf=((spl.get('grid') or {}).get('support_film') or [{}])
    r['film']=';'.join(sorted({str(s.get('film_material')) for s in sf}))
    r['buffer_ph']=fl((spl.get('buffer') or {}).get('ph'))
    r['n_buffer_comp']=len(((spl.get('buffer') or {}).get('component') or []))
    smp=d.get('sample') or {}
    r['n_macromol']=len(((smp.get('macromolecule_list') or {}).get('macromolecule') or []))
    r['n_supramol']=len(((smp.get('supramolecule_list') or {}).get('supramolecule') or []))
    r['n_fields']=len(pairs)
    rows.append(r)
E=pd.DataFrame(rows); E.to_csv(f'{OUT}/emdb_features.csv.gz',index=False)
# aggregate to PDB entries: primary EMDB id from RCSB
F=pd.read_csv(f'{OUT}/entry_features_rcsb.csv.gz',usecols=['pdb','emdb_ids'])
F=F[F.emdb_ids.notna()]; F['emdb']=F.emdb_ids.str.split(';').str[0]
M=F[['pdb','emdb']].merge(E,on='emdb',how='left')
M.to_csv(f'{OUT}/entry_features_emdb.csv.gz',index=False)
print(E.shape, M.shape, M.n_half.notna().sum())
print(E[[c for c in E.columns if c.startswith('kw_')]+['composite_flag','energy_filter']].mean())
print(E[['n_half','n_add','n_mask','n_emdb_ref','n_focused_ref','n_consensus_ref','n_pdb_ref','map_pix','map_box','n_images_used']].describe().T)
