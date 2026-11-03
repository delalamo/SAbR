"""Fetch entry-level PDB metadata (EM + validation + composition) via RCSB GraphQL for all entries in entries.csv."""
import json, urllib.request, sys, time, os
import pandas as pd
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUT=f'{SP}/cache/pdb_meta/rcsb_meta.jsonl'
Q='''query($ids:[String!]!){ entries(entry_ids:$ids){
 rcsb_id
 em_experiment{aggregation_state reconstruction_method}
 em_3d_reconstruction{resolution resolution_method num_particles symmetry_type refinement_type algorithm actual_pixel_size nominal_pixel_size num_class_averages details}
 em_imaging{microscope_model accelerating_voltage electron_source illumination_mode mode nominal_magnification calibrated_magnification nominal_defocus_min nominal_defocus_max specimen_holder_model nominal_cs c2_aperture_diameter alignment_procedure date}
 em_image_recording{avg_electron_dose_per_image film_or_detector_model detector_mode num_real_images num_grids_imaged average_exposure_time}
 em_specimen{concentration vitrification_applied staining_applied}
 em_vitrification{cryogen_name instrument humidity chamber_temperature}
 em_3d_fitting{ref_protocol ref_space target_criteria overall_b_value method details}
 em_3d_fitting_list{pdb_entry_id pdb_chain_id details}
 em_ctf_correction{type details}
 em_software{category name version}
 em_entity_assembly{name type source oligomeric_details}
 em_single_particle_entity{point_symmetry}
 em_particle_selection{num_particles_selected}
 pdbx_database_status{deposit_site process_site status_code SG_entry methods_development_category recvd_initial_deposition_date}
 rcsb_accession_info{deposit_date initial_release_date revision_date major_revision minor_revision}
 rcsb_entry_info{molecular_weight deposited_atom_count deposited_model_count deposited_polymer_monomer_count deposited_modeled_polymer_monomer_count deposited_unmodeled_polymer_monomer_count polymer_entity_count polymer_entity_count_protein deposited_polymer_entity_instance_count entity_count assembly_count nonpolymer_entity_count deposited_nonpolymer_entity_instance_count software_programs_combined resolution_combined polymer_composition cis_peptide_count disulfide_bond_count polymer_entity_taxonomy_count deposited_hydrogen_atom_count branched_entity_count}
 pdbx_vrpt_summary{report_creation_date attempted_validation_steps}
 pdbx_vrpt_summary_em{Q_score atom_inclusion_all_atoms atom_inclusion_backbone author_provided_fsc_resolution_by_cutoff_pt_143 calculated_fsc_resolution_by_cutoff_pt_143 calculated_fsc_resolution_by_cutoff_halfbit contour_level_primary_map}
 pdbx_vrpt_summary_geometry{clashscore percent_ramachandran_outliers percent_rotamer_outliers angles_RMSZ bonds_RMSZ}
 pdbx_vrpt_summary_diffraction{DCC_R DCC_Rfree percent_RSRZ_outliers Wilson_B_estimate data_completeness Q_score}
 struct_keywords{pdbx_keywords text}
 rcsb_entry_container_identifiers{emdb_ids related_emdb_ids}
 audit_author{name}
 pdbx_audit_revision_history{ordinal revision_date major_revision minor_revision data_content_type}
 pdbx_deposit_group{group_id group_type}
 pdbx_database_related{db_name db_id content_type details}
 pdbx_audit_support{funding_organization country}
 refine{ls_R_factor_R_free ls_R_factor_R_work B_iso_mean pdbx_refine_id ls_d_res_high}
 polymer_entities{ rcsb_polymer_entity{pdbx_description formula_weight pdbx_number_of_molecules} entity_poly{rcsb_sample_sequence_length rcsb_entity_polymer_type}
   polymer_entity_instances{ rcsb_polymer_entity_instance_container_identifiers{auth_asym_id asym_id} rcsb_polymer_instance_info{modeled_residue_count} pdbx_vrpt_summary_entity_fit_to_map{Q_score average_residue_inclusion} pdbx_vrpt_summary_entity_geometry{angles_RMSZ bonds_RMSZ} rcsb_polymer_instance_feature_summary{type count coverage} } }
}}'''
def post(ids):
    req=urllib.request.Request('https://data.rcsb.org/graphql', data=json.dumps({'query':Q,'variables':{'ids':ids}}).encode(), headers={'Content-Type':'application/json'})
    for a in range(5):
        try:
            return json.load(urllib.request.urlopen(req, timeout=300))
        except Exception as e:
            print('retry',a,e,file=sys.stderr); time.sleep(5*(a+1))
    raise SystemExit('failed')
if __name__=='__main__':
    ids=[p.upper() for p in pd.read_csv(f'{SP}/entries.csv').pdb]
    if len(sys.argv)>1: ids=sys.argv[1:]; OUT=OUT+'.test'
    done=set()
    if os.path.exists(OUT):
        done={json.loads(l)['rcsb_id'] for l in open(OUT)}
    todo=[i for i in ids if i not in done]
    B=50
    with open(OUT,'a') as f:
        for k in range(0,len(todo),B):
            r=post(todo[k:k+B])
            if r.get('errors'): print(json.dumps(r['errors'])[:2000],file=sys.stderr)
            for e in (r.get('data') or {}).get('entries') or []:
                if e: f.write(json.dumps(e)+'\n')
            f.flush(); print(k+B,len(todo),file=sys.stderr)
