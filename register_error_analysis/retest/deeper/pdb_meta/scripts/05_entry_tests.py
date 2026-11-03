"""Entry-level tests within cryo-EM: first-occurrence error entries vs clean entries (no run of any kind).
Logistic regression per feature, adjusted for deposition-year bin, resolution and log2(number of V chains),
cluster-robust SE by paper (DOI). BH q across all tests."""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature, bh, ybin
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
ent=pd.read_csv(f'{SP}/entries.csv')
F=pd.read_csv(f'{OUT}/entry_features_rcsb.csv.gz')
E=pd.read_csv(f'{OUT}/entry_features_emdb.csv.gz')
ch=pd.read_csv(f'{SP}/chains.csv.gz',usecols=['pdb','role','aux_reagent','vhh','ctype','irm','fit'])
d=ent.merge(F,on='pdb',how='left').merge(E,on='pdb',how='left')
d=d[d.method=='EM'].copy()
d['amyloid']=d.title.fillna('').str.contains(r'amyloid(?!-beta)',case=False)
d=d[(d.err_first)|(~d.err_any)].copy()
if '--with-amyloid' not in sys.argv: d=d[~d.amyloid].copy()
d['y']=d.err_first.astype(int)
d['cluster']=d.doi.fillna(d.pdb)
d['dep_year']=pd.to_datetime(d.deposit).dt.year
d['ybin']=ybin(d.dep_year.values)
d['l2nV']=np.log2(d.n_V_chains)
# ---------- derived features
L=np.log2; L10=np.log10
d['log2_tot_mw']=L(d.tot_mw); d['log2_n_inst']=L(d.n_inst); d['log2_nonab_inst']=L(1+d.n_nonab_protein_inst)
d['ab_alone']=(d.n_nonab_protein_inst==0).astype(int)
d['huge_gt1MDa']=(d.tot_mw>1000).astype(int); d['big_gt500kDa']=(d.tot_mw>500).astype(int)
d['ab_small_part_lt15pct']=(d.ab_frac_mw<0.15).astype(int)
kw=d.pdbx_keywords.fillna('').str.upper()
d['kw_membrane']=kw.str.contains('MEMBRANE|TRANSPORT|CHANNEL|RECEPTOR|SIGNALING').astype(int)
d['kw_viral']=kw.str.contains('VIR').astype(int)
d['kw_immune_only']=(kw=='IMMUNE SYSTEM').astype(int)
d['h_atoms_present']=(d.deposited_hydrogen_atom_count>0).astype(int)
d['cis_per_1000']=1000*d.cis_peptide_count/d.deposited_modeled_polymer_monomer_count
d['fsc_gap_calc_minus_author']=d.vem_calculated_fsc_resolution_by_cutoff_pt_143-d.res
d['log10_particles']=L10(d.rec_num_particles)
symord=d.point_symmetry.fillna(d.applied_sym).fillna('').str.extract(r'^[CD](\d+)')[0].astype(float)
symord=symord*np.where(d.point_symmetry.fillna(d.applied_sym).fillna('').str.startswith('D'),2,1)
d['sym_order']=symord; d['sym_C1']=(symord==1).astype(float).where(symord.notna())
d['log10_asu_particles']=L10(d.rec_num_particles*d.sym_order.fillna(1))
d['log10_particles_per_kDa']=L10(d.rec_num_particles/d.tot_mw)
d['particle_retention']=d.rec_num_particles/d.n_picked.where(d.n_picked>0)
d['log10_picked']=L10(d.n_picked.where(d.n_picked>0))
d['nyquist_ratio']=d.res/d.map_pix
d['pix_lt_1A']=(d.map_pix<1.0).astype(float).where(d.map_pix.notna())
d['box_A']=d.map_box_A
d['krios']=d.im_microscope_model.fillna('').str.contains('KRIOS').astype(int)
det=d.rec2_film_or_detector_model.fillna('')
d['det_K2']=det.str.contains('K2').astype(int); d['det_K3']=det.str.contains('K3').astype(int); d['det_Falcon']=det.str.contains('FALCON').astype(int)
d['kv300']=(d.im_accelerating_voltage==300).astype(int)
d['dose']=d.rec2_avg_electron_dose_per_image
d['energy_filter']=d.energy_filter.astype(float)
for c in ['kw_local','kw_composite','kw_consensus','kw_deepemhancer','kw_densmod','kw_locscale','kw_sharpen','kw_unsharp','kw_nu','kw_symexp','kw_3dflex','composite_flag']:
    d[c]=d[c].astype(float)
ttl=(d.title.fillna('')+' '+d.rec_details.fillna('')+' '+d.related_details.fillna('')).str.lower()
d['title_local']=ttl.str.contains('local refine|focused|locally refined|local map|interface').astype(int)
d['any_local_or_focused']=((d.kw_local==1)|(d.n_focused_ref>0)|(d.title_local==1)).astype(float).where(d.kw_local.notna())
d['has_focused_ref']=(d.n_focused_ref>0).astype(float).where(d.n_focused_ref.notna())
d['has_emdb_ref']=(d.n_emdb_ref>0).astype(float).where(d.n_emdb_ref.notna())
d['has_additional_map']=(d.n_add>0).astype(float).where(d.n_add.notna())
d['n_add_maps']=d.n_add
d['has_half_maps']=(d.n_half>=2).astype(float).where(d.n_half.notna())
d['has_mask']=(d.n_mask>0).astype(float).where(d.n_mask.notna())
d['multi_pdb_per_map']=(d.n_pdb_ref>1).astype(float).where(d.n_pdb_ref.notna())
d['multi_emdb_per_pdb']=(d.n_emdb>1).astype(int)
d['has_related_emdb']=(d.n_related_emdb>0).astype(int)
fs=d.final_sw.fillna('')
d['sw_cryosparc_final']=fs.str.contains('cryoSPARC',case=False).astype(float).where(d.final_sw.notna()&(fs!=''))
d['sw_relion_final']=fs.str.contains('RELION',case=False).astype(float).where(d.final_sw.notna()&(fs!=''))
st=d.startup.fillna('')
for k in ['PDB ENTRY','INSILICO MODEL','NONE','EMDB MAP','OTHER']:
    d['startup_'+k.split()[0].lower()]=st.str.contains(k).astype(float).where(d.startup.notna())
for s in ['RCSB','PDBE','PDBJ','PDBC']:
    d['proc_'+s]=(d.process_site==s).astype(int); d['dep_'+s]=(d.deposit_site==s).astype(int)
d['proc_ne_dep']=(d.process_site!=d.deposit_site).astype(int)
d['revised_major']=(d.n_major_rev_events>0).astype(int)
d['log2_authors']=L(d.n_authors)
fc=d.fund_countries.fillna('')
d['fund_US']=fc.str.contains('United States').astype(int); d['fund_China']=fc.str.contains('China').astype(int)
d['fund_UK']=fc.str.contains('United Kingdom').astype(int); d['fund_none']=(d.n_funding==0).astype(int)
d['fund_NIH']=d.fund_orgs.fillna('').str.contains('National Institutes of Health|NIH|NIAID|NIGMS',case=False).astype(int)
d['fund_Gates']=d.fund_orgs.fillna('').str.contains('Gates',case=False).astype(int)
d['ctf_none']=(d.ctf.fillna('')=='NONE').astype(int)
# metadata completeness
opt=['im_c2_aperture_diameter','im_nominal_defocus_min','im_nominal_magnification','spec_conc','vit_instrument','vit_humidity','rec2_num_real_images','rec2_num_grids_imaged','rec2_average_exposure_time','n_particles_selected','point_symmetry','im_nominal_cs','im_date','fit_b']
d['meta_complete']=d[opt].notna().mean(axis=1)+0.0
d['fit_recorded']=(d.fit_protocols.fillna('NA')!='NA').astype(int)
d['sw_versions_frac']=d.n_em_sw_version/d.n_em_sw_named.replace(0,np.nan)
d['emdb_n_fields']=d.n_fields
d['log_mw_per_V']=L(d.tot_mw/d.n_V_chains)
d['ab_cis_zero']=(d.ab_cis_per_1000==0).astype(int)
d['non_cis_zero']=(d.non_cis_per_1000==0).astype(float).where(d.n_nonab_protein_inst>0)
TAG='_with_amyloid' if '--with-amyloid' in sys.argv else ''
d.to_csv(f'{OUT}/em_entry_analysis_frame{TAG}.csv.gz',index=False)
FEATS=[ # (feature, kind, group, note)
 ('log2_tot_mw','cont','composition',''),('ab_frac_mw','cont','composition',''),('log2_n_inst','cont','composition',''),('log2_nonab_inst','cont','composition',''),
 ('ab_alone','bin','composition',''),('big_gt500kDa','bin','composition',''),('huge_gt1MDa','bin','composition',''),('ab_small_part_lt15pct','bin','composition',''),
 ('polymer_entity_count_protein','cont','composition',''),('polymer_entity_taxonomy_count','cont','composition',''),('kw_membrane','bin','composition',''),('kw_viral','bin','composition',''),('kw_immune_only','bin','composition',''),
 ('unmod_frac','cont','modelled fraction','entry-wide'),('non_unobs','cont','modelled fraction','non-antibody chains only'),('ab_unobs','cont','modelled fraction','OVERLAPS gap mechanism'),
 ('h_atoms_present','bin','modelling',''),('cis_per_1000','cont','modelling',''),('ab_cis_per_1000','cont','modelling','antibody chains'),('ab_cis_zero','bin','modelling','antibody chains: no cis peptide modelled'),('non_cis_per_1000','cont','modelling','non-antibody chains'),('fit_recorded','bin','metadata',''),
 ('vem_Q_score','cont','map quality','entry'),('vem_atom_inclusion_backbone','cont','map quality','entry'),('vem_calculated_fsc_resolution_by_cutoff_pt_143','cont','map quality',''),('fsc_gap_calc_minus_author','cont','map quality',''),
 ('non_q','cont','map quality','non-antibody chains'),('ab_q','cont','map quality','antibody chains; partly OVERLAPS error'),('dq_ab_minus_non','cont','map quality','antibody vs rest of complex'),
 ('non_inc','cont','map quality',''),('ab_inc','cont','map quality','partly OVERLAPS'),('dinc_ab_minus_non','cont','map quality',''),
 ('log10_particles','cont','particles',''),('log10_asu_particles','cont','particles',''),('log10_particles_per_kDa','cont','particles',''),('log10_picked','cont','particles',''),('particle_retention','cont','particles',''),
 ('sym_C1','bin','particles',''),('nyquist_ratio','cont','imaging',''),('map_pix','cont','imaging',''),('box_A','cont','imaging',''),
 ('krios','bin','imaging',''),('kv300','bin','imaging',''),('det_K2','bin','imaging',''),('det_K3','bin','imaging',''),('det_Falcon','bin','imaging',''),('energy_filter','bin','imaging',''),('dose','cont','imaging',''),('ctf_none','bin','imaging',''),
 ('any_local_or_focused','bin','local/focused/composite',''),('kw_local','bin','local/focused/composite',''),('has_focused_ref','bin','local/focused/composite',''),('kw_consensus','bin','local/focused/composite',''),('kw_composite','bin','local/focused/composite',''),('composite_flag','bin','local/focused/composite',''),
 ('has_emdb_ref','bin','local/focused/composite',''),('has_additional_map','bin','local/focused/composite',''),('n_add_maps','cont','local/focused/composite',''),('has_half_maps','bin','deposited files',''),('has_mask','bin','deposited files',''),
 ('multi_pdb_per_map','bin','local/focused/composite',''),('multi_emdb_per_pdb','bin','local/focused/composite',''),('has_related_emdb','bin','local/focused/composite',''),
 ('kw_deepemhancer','bin','post-processing',''),('kw_densmod','bin','post-processing',''),('kw_locscale','bin','post-processing',''),('kw_sharpen','bin','post-processing',''),('kw_unsharp','bin','post-processing',''),
 ('kw_nu','bin','processing',''),('kw_symexp','bin','processing',''),('kw_3dflex','bin','processing',''),
 ('sw_cryosparc_final','bin','processing',''),('sw_relion_final','bin','processing',''),('startup_pdb','bin','processing','EM startup model'),('startup_insilico','bin','processing',''),('startup_none','bin','processing',''),
 ('proc_RCSB','bin','site',''),('proc_PDBE','bin','site',''),('proc_PDBJ','bin','site',''),('proc_PDBC','bin','site',''),('dep_RCSB','bin','site',''),('dep_PDBE','bin','site',''),('dep_PDBJ','bin','site',''),('proc_ne_dep','bin','site',''),('deposit_group','bin','site',''),
 ('vg_clashscore','cont','validation',''),('vg_percent_ramachandran_outliers','cont','validation',''),('vg_percent_rotamer_outliers','cont','validation',''),('vg_angles_RMSZ','cont','validation',''),('vg_bonds_RMSZ','cont','validation',''),
 ('ab_clash','cont','validation','antibody chains; may OVERLAP'),('non_clash','cont','validation','non-antibody chains'),
 ('revised_major','bin','revisions',''),('n_rev','cont','revisions',''),
 ('log2_authors','cont','authorship',''),('n_funding','cont','authorship',''),('fund_US','bin','authorship',''),('fund_China','bin','authorship',''),('fund_UK','bin','authorship',''),('fund_NIH','bin','authorship',''),('fund_Gates','bin','authorship',''),('fund_none','bin','authorship',''),
 ('meta_complete','cont','metadata',''),('sw_versions_frac','cont','metadata',''),('emdb_n_fields','cont','metadata',''),('spec_conc','cont','specimen',''),('buffer_ph','cont','specimen',''),
]
res=[]
for feat,kind,grp,note in FEATS:
    if feat not in d: continue
    dd=d.copy()
    if kind=='bin': dd[feat]=dd[feat].astype(float)
    r=fit_feature(dd,feat,kind,['C(ybin)','res','l2nV']); r.update(group=grp,note2=note); res.append(r)
R=bh(pd.DataFrame(res))
R=R[['group','feature','kind','note2','n_err','n_ctrl','err_val','ctrl_val','unit','OR','lo','hi','p','q','p_unadj']+[c for c in R.columns if c=='note']]
R.to_csv(f'{OUT}/entry_tests_em{TAG}.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',45)
print(R.sort_values('p').to_string(index=False,float_format=lambda x:f'{x:.3g}'))
