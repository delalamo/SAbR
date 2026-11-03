"""Build entry-level and V-chain-level feature tables from the RCSB GraphQL dump."""
import json, re, numpy as np, pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
import os; os.makedirs(OUT,exist_ok=True)
ent=pd.read_csv(f'{SP}/entries.csv'); ch=pd.read_csv(f'{SP}/chains.csv.gz')
abchains=ch.groupby('pdb').chain.apply(set).to_dict()
def g(d,*ks):
    for k in ks:
        if d is None: return None
        d=d.get(k) if isinstance(d,dict) else (d[0].get(k) if isinstance(d,list) and d else None)
    return d
def first(l): return l[0] if isinstance(l,list) and l else (l if isinstance(l,dict) else {})
rows=[]; crows=[]
for l in open(f'{SP}/cache/pdb_meta/rcsb_meta.jsonl'):
    e=json.loads(l); pdb=e['rcsb_id'].lower(); r={'pdb':pdb}
    info=e.get('rcsb_entry_info') or {}
    for k in ['molecular_weight','deposited_atom_count','deposited_polymer_monomer_count','deposited_modeled_polymer_monomer_count','deposited_unmodeled_polymer_monomer_count','polymer_entity_count','polymer_entity_count_protein','deposited_polymer_entity_instance_count','nonpolymer_entity_count','deposited_nonpolymer_entity_instance_count','cis_peptide_count','disulfide_bond_count','polymer_entity_taxonomy_count','deposited_hydrogen_atom_count','branched_entity_count','deposited_model_count']:
        r[k]=info.get(k)
    r['polymer_composition']=info.get('polymer_composition')
    r['unmod_frac']=(info.get('deposited_unmodeled_polymer_monomer_count') or 0)/max(1,info.get('deposited_polymer_monomer_count') or 1)
    rec=first(e.get('em_3d_reconstruction'))
    r['n_recon']=len(e.get('em_3d_reconstruction') or [])
    for k in ['resolution_method','num_particles','symmetry_type','algorithm','actual_pixel_size','nominal_pixel_size','num_class_averages']:
        r['rec_'+k]=rec.get(k)
    r['rec_details']=' | '.join([(x.get('details') or '') for x in (e.get('em_3d_reconstruction') or [])])
    r['n_particles_selected']=g(e,'em_particle_selection','num_particles_selected')
    r['point_symmetry']=g(e,'em_single_particle_entity','point_symmetry')
    im=first(e.get('em_imaging'))
    for k in ['microscope_model','accelerating_voltage','electron_source','nominal_magnification','nominal_defocus_min','nominal_defocus_max','c2_aperture_diameter','nominal_cs','specimen_holder_model']:
        r['im_'+k]=im.get(k)
    r['im_date']=im.get('date')
    ir=first(e.get('em_image_recording'))
    for k in ['avg_electron_dose_per_image','film_or_detector_model','detector_mode','num_real_images','num_grids_imaged','average_exposure_time']:
        r['rec2_'+k]=ir.get(k)
    sp=first(e.get('em_specimen')); r['spec_conc']=sp.get('concentration')
    vi=first(e.get('em_vitrification')); r['vit_instrument']=vi.get('instrument'); r['vit_humidity']=vi.get('humidity'); r['vit_temp']=vi.get('chamber_temperature')
    fi=e.get('em_3d_fitting') or []
    r['fit_protocols']=';'.join(sorted({(x.get('ref_protocol') or 'NA') for x in fi}))
    r['fit_space']=';'.join(sorted({(x.get('ref_space') or 'NA') for x in fi}))
    r['fit_target']=';'.join(sorted({(x.get('target_criteria') or 'NA') for x in fi}))
    r['fit_b']=g(e,'em_3d_fitting','overall_b_value')
    r['fit_details']=' | '.join([(x.get('details') or '') for x in fi])
    fl=e.get('em_3d_fitting_list') or []
    r['n_fit_list']=len(fl); r['fit_list_ids']=';'.join(sorted({(x.get('pdb_entry_id') or '').upper() for x in fl}))
    r['ctf']=';'.join(sorted({(x.get('type') or 'NA') for x in (e.get('em_ctf_correction') or [])}))
    sw=e.get('em_software') or []
    r['em_sw']=';'.join(sorted({f"{x.get('category')}={x.get('name')}" for x in sw if x.get('name')}))
    r['n_em_sw_named']=sum(1 for x in sw if x.get('name'))
    r['n_em_sw_version']=sum(1 for x in sw if x.get('version'))
    st=e.get('pdbx_database_status') or {}
    r['deposit_site']=st.get('deposit_site'); r['process_site']=st.get('process_site')
    acc=e.get('rcsb_accession_info') or {}
    r['major_rev']=acc.get('major_revision'); r['minor_rev']=acc.get('minor_revision')
    rh=e.get('pdbx_audit_revision_history') or []
    r['n_rev']=len(rh); r['n_major_rev_events']=len({x.get('major_revision') for x in rh})-1
    vem=first(e.get('pdbx_vrpt_summary_em'))
    for k in ['Q_score','atom_inclusion_all_atoms','atom_inclusion_backbone','author_provided_fsc_resolution_by_cutoff_pt_143','calculated_fsc_resolution_by_cutoff_pt_143','calculated_fsc_resolution_by_cutoff_halfbit','contour_level_primary_map']:
        r['vem_'+k]=vem.get(k)
    vg=first(e.get('pdbx_vrpt_summary_geometry'))
    for k in ['clashscore','percent_ramachandran_outliers','percent_rotamer_outliers','angles_RMSZ','bonds_RMSZ']:
        r['vg_'+k]=vg.get(k)
    vd=first(e.get('pdbx_vrpt_summary_diffraction'))
    for k in ['DCC_Rfree','percent_RSRZ_outliers','Wilson_B_estimate','data_completeness']:
        r['vd_'+k]=vd.get(k)
    rf=first(e.get('refine')); r['R_free']=rf.get('ls_R_factor_R_free'); r['R_work']=rf.get('ls_R_factor_R_work'); r['B_mean']=rf.get('B_iso_mean')
    sk=e.get('struct_keywords') or {}; r['pdbx_keywords']=sk.get('pdbx_keywords'); r['kw_text']=sk.get('text')
    ci=e.get('rcsb_entry_container_identifiers') or {}
    r['emdb_ids']=';'.join(ci.get('emdb_ids') or []); r['n_emdb']=len(ci.get('emdb_ids') or [])
    r['n_related_emdb']=len(ci.get('related_emdb_ids') or [])
    r['n_authors']=len(e.get('audit_author') or [])
    r['deposit_group']=bool(e.get('pdbx_deposit_group'))
    rel=e.get('pdbx_database_related') or []
    r['n_related']=len(rel); r['n_related_emdb_db']=sum(1 for x in rel if (x.get('db_name') or '').upper()=='EMDB')
    r['related_types']=';'.join(sorted({(x.get('content_type') or 'NA') for x in rel}))
    r['related_details']=' | '.join([(x.get('details') or '') for x in rel])[:2000]
    fs=e.get('pdbx_audit_support') or []
    r['n_funding']=len(fs); r['fund_countries']=';'.join(sorted({(x.get('country') or 'NA') for x in fs}))
    r['fund_orgs']=';'.join(sorted({(x.get('funding_organization') or 'NA') for x in fs}))
    # polymer composition / antibody share / per-chain quality
    abset=abchains.get(pdb,set())
    ab_mw=0; tot_mw=0; n_inst=0; abq=[]; nonq=[]; abinc=[]; noninc=[]; abun=[]; nonun=[]; abclash=[]; nonclash=[]; abcis=[]; noncis=[]; nonmod=[]
    for pe in e.get('polymer_entities') or []:
        fw=g(pe,'rcsb_polymer_entity','formula_weight') or 0
        L=g(pe,'entity_poly','rcsb_sample_sequence_length') or 0
        ptype=g(pe,'entity_poly','rcsb_entity_polymer_type')
        desc=g(pe,'rcsb_polymer_entity','pdbx_description')
        for inst in pe.get('polymer_entity_instances') or []:
            a=g(inst,'rcsb_polymer_entity_instance_container_identifiers','auth_asym_id')
            n_inst+=1; tot_mw+=fw
            q=g(inst,'pdbx_vrpt_summary_entity_fit_to_map','Q_score'); inc=g(inst,'pdbx_vrpt_summary_entity_fit_to_map','average_residue_inclusion')
            fsum={f['type']:f for f in (inst.get('rcsb_polymer_instance_feature_summary') or [])}
            un=(fsum.get('UNOBSERVED_RESIDUE_XYZ') or {}).get('coverage') or 0.0
            cl=(fsum.get('CLASHES') or {}).get('count') or 0
            mod=g(inst,'rcsb_polymer_instance_info','modeled_residue_count')
            isab=a in abset
            if isab:
                ab_mw+=fw; abq.append(q); abinc.append(inc); abun.append(un); abclash.append(cl/max(1,mod or 1)); abcis.append(1000*((fsum.get('CIS-PEPTIDE') or {}).get('count') or 0)/max(1,mod or 1))
                crows.append({'pdb':pdb,'chain':a,'chain_q':q,'chain_inc':inc,'chain_unobs_cov':un,'chain_clash_per_res':cl/max(1,mod or 1),
                    'chain_sheet_cov':(fsum.get('SHEET') or {}).get('coverage'),'chain_unassigned_ss_cov':(fsum.get('UNASSIGNED_SEC_STRUCT') or {}).get('coverage'),
                    'chain_angle_outl':(fsum.get('ANGLE_OUTLIERS') or {}).get('count') or 0,'chain_cis':(fsum.get('CIS-PEPTIDE') or {}).get('count') or 0,'chain_len':L,'chain_modeled':mod,
                    'chain_rmsz_ang':g(inst,'pdbx_vrpt_summary_entity_geometry','angles_RMSZ'),'chain_fw':fw,'chain_desc_rcsb':desc})
            elif ptype=='Protein':
                nonq.append(q); noninc.append(inc); nonun.append(un); nonclash.append(cl/max(1,mod or 1)); noncis.append(1000*((fsum.get('CIS-PEPTIDE') or {}).get('count') or 0)/max(1,mod or 1)); nonmod.append(mod or 0)
    f=lambda x: np.nanmean([v for v in x if v is not None]) if any(v is not None for v in x) else np.nan
    r.update(n_inst=n_inst, ab_mw=ab_mw, tot_mw=tot_mw, ab_frac_mw=ab_mw/tot_mw if tot_mw else np.nan, n_ab_inst=len(abq),
             ab_q=f(abq), non_q=f(nonq), ab_inc=f(abinc), non_inc=f(noninc), ab_unobs=f(abun), non_unobs=f(nonun),
             ab_clash=f(abclash), non_clash=f(nonclash), ab_cis_per_1000=f(abcis), non_cis_per_1000=f(noncis), nonab_modeled=sum(nonmod), n_nonab_protein_inst=len(nonq))
    rows.append(r)
F=pd.DataFrame(rows)
F['dq_ab_minus_non']=F.ab_q-F.non_q
F['dinc_ab_minus_non']=F.ab_inc-F.non_inc
F.to_csv(f'{OUT}/entry_features_rcsb.csv.gz',index=False)
C=pd.DataFrame(crows); C.to_csv(f'{OUT}/chain_features_rcsb.csv.gz',index=False)
m=ch[['pdb','chain']].merge(C,on=['pdb','chain'],how='left',indicator=True)
print(F.shape, C.shape, m._merge.value_counts().to_dict())
print(F.describe().T.to_string())
