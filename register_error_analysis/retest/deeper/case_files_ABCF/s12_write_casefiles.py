"""Write the case files (case_files.md) and the case summary table (tables/case_summary.csv).
Facts come from tables/case_table.csv; the 'clue' line for each case is the investigator's reading of the methods text,
template comparison, local structure and sequence (s03-s11)."""
import pandas as pd, numpy as np
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF'
D = pd.read_csv(f'{OUT}/tables/case_table.csv')
CLUE = {
 '6ad0': 'Homology model of the Fab from an earlier CVA6 immune complex (5XS7; Discovery Studio), docked into a segmented capsid map; only the V domains were built. The mouse IGKV1-117 light chain has an 11-residue CDR-L1, the template light chain a 6-residue one. The run (IMGT 1-32, +1) covers FR1 and stops in CDR1, where the lengths differ.',
 '7czu': 'Chainsaw model from 7C2L (a different antibody), MDFF into the whole-spike map and the focused RBD-Fab map. The N-terminus is built from IMGT 1 but the strand is one residue off (1-9, -1), with poor site fit (24th percentile). The same slip is in the sibling entries 7czr/7czs/7czv/7d00 of the same paper.',
 '7jhg': 'Recorded starting models include 6CMO, a Fab from the Xu H.E. lab at the same institute. Its light chain is 98% identical and carries the same N-terminal slip (-2). This is a near-identical template copy, missed by the identical-sequence copy rule. The same applies to 7jhh and 7m74.',
 '7tz5': 'No starting model recorded and no open-access text. SEQRES starts at IMGT 2. The whole FR1 is shifted -1 (2-31). The period control 7whz (another lab) has an identical IGLV1-40 window sequence and was built correctly.',
 '8e8l': 'ABodyBuilder model, built in ISOLDE then Phenix in an icosahedral poliovirus map. The construct starts at IMGT 4 (SEQRES truncated), so the strand has no N-terminal anchor (4-16, -1).',
 '8rvn': 'Starting model 6T3F (a different Fab, 81% identity, same lengths at the site). Three N-terminal residues are unbuilt, so the built strand starts at IMGT 4 (4-17, -1). The V region is far from germline (67%).',
 '9mnb': 'An AlphaFold2 model of the rabbit Fab E82.AS was the starting model, then ISOLDE and Phenix. The rabbit VH sequence starts at IMGT 3 (QSVEESRG), but the strand was built one residue off (3-9, -1; 20th-percentile fit). An AlphaFold start did not prevent an N-terminal slip.',
 '6bpe': 'X-ray molecular replacement with 5W53/5EN2. Two N-terminal residues are unbuilt. Run 3-8 (-1) with RSCC at the 14th percentile and a site B-factor z of +1.9.',
 '7e39': 'No starting model recorded; symmetry expansion plus focused classification. The run spans CDR2 to the end of FR3 (62-93, +1) next to 3 unbuilt residues. The light chain of the same paper (7e3c) also slipped.',
 '12iu': 'Unpublished (2026). Kappa CDR2/FR3 run 58-72 (+3) in a chain whose ANARCI gap pattern at IMGT 59-62 disagrees with SAbR in all 3 scanned chains. This is possibly a gap-placement artifact (no high-resolution example).',
 '8cr9': 'No open-access text. FR2-CDR2 run 50-63 (-3) at the antigen contact. The same entry has FR3 (C) and kappa runs. Map at 4.2 A.',
 '7oh0': 'No open-access text. CDR2 run 63-69 (-1) at the antigen interface, with a clash excess at the site.',
 '7mkl': 'Templates chosen by BLAST: 1KIQ for VH and 5XJM for VL. They were docked into a locally refined Fv/RBD map and refined in Coot, ISOLDE and Phenix. The CDR2 slip (54-59, +1) sits at the antigen interface. The template has the same CDR2 length, so this slip happened without a length mismatch. The entry has an "Atomic model" revision but is still flagged.',
 '8dg8': 'No starting model stated in the text. Lambda CDR2 run 57-70 (+1) next to unbuilt residues; the first built residue is IMGT 3.',
 '7wvl': 'X-ray. The model was started from the STEC90-C11 Fab (7B3O) and "the sequence ... was slowly changed to that of P4A2". The template CDR-H2 is 7 residues, P4A2 has 8, and the run 62-82 is shifted +1, so the extra residue was absorbed by sliding FR3. This is a textbook template-threading error.',
 '9dm7': '"Most of the Fab ... was built based on a preexisting model at high resolution (5UCB)". The Fab E6 FR3 is two residues shorter than 5UCB (ANARCI gaps 94-95; FR3 36 vs 38), and FR3 72-92 is shifted +1 (25th-percentile fit). The length mismatch is exactly at the slipped strand.',
 '7e23': 'No starting model stated. The kappa FR3 FSGSGSGTDF slipped by -2 (76-93), exactly the period of the SG repeat. The same -2 kappa FR3 slip is in 7stf (Miller M.). The germline IGKV3-11 light chain had been solved at high resolution 14 times before.',
 '8w5e': 'Mouse IGHV5-15 (mouse IGHV5 is enriched in mode C; see ../deepdive). Six residues are unbuilt near the run (71-89, +1). No open-access text.',
 '7kra': 'The yeast EMC was built de novo by hand, with landmark residues used to assign sequence. FR3 run 80-93 (-2) next to 9 unbuilt residues. The same germline V region had two earlier high-resolution structures, which were not used.',
 '7mhz': 'Mouse Fab, 4 unbuilt residues next to FR3 run 87-95 (-2). An "Atomic model" revision was later deposited, but the run is still present.',
 '9yfu': 'One-residue FR3 deletion (ANARCI gap at 85 instead of the usual 81). The 81-84 (+1) run lies between the two possible gap positions. Probably a gap-placement artifact (pattern seen in 2 chains, no high-resolution example).',
 '2x7l': 'X-ray. FR3 deletion with ANARCI gaps at 80 and 84. SAbR shows the identical disagreement (D81/80, T82/83, S83/84) on the 2.3 A structure 5DHV, so this is a numbering artifact, not a register error.',
 '3jbf': 'VHH modelled from the llama VHH 1I3U. The template CDR1 is 8 residues, PVSP19B has 5, and FR2 is shifted -3 (37-48), the exact length difference. 4.8 A map, 2015.',
 '6lht': 'Localized reconstruction of a CVA16 virion with a mouse Fab. The IGKV8-30 light chain has a 12-residue CDR-L1, and CDR1-FR2-CDR2 is shifted +5 (31-57). This is consistent with a short-L1 starting model whose missing residues were absorbed downstream. No open-access text.',
 '7t6x': 'Short FR2 run 44-47 (-1, QKPG) in an HCV E1E2-Fab map. No open-access text.',
 '8gse': 'Starting models 5HDQ and 1NMC. The mouse 6D10 VH has an FR1 deletion (SCQA-FYTF; ANARCI gap at 26). SAbR shows the same 27-35 disagreement on the 2.49 A structure 5AAM, so this is a numbering artifact. A different lab\'s anti-echovirus-3 Fab 5G3 (7eaj/7eak) has the same rare FR1 deletion and the same artifact.',
 '9ml5': 'Rabbit Fab M8b-B8. The model was started by docking human Fab-RBD structures (7UZD, 7UZ4) into a locally refined map. The rabbit CDR-H1 (SSSYWICW, 9 residues, with an extra Cys) differs in length from the templates (8 or 10), and CDR1 is shifted +1 (31-41). The V-domain local resolution is 6.7 A against 3.4 A global. The same chain also has an FR3 run on an artifact-prone FR3 gap (84).',
 '9yc6': 'The recorded starting model 3U74 is uPAR alone, so there was no Fab template. The mouse IGKV1-117 light chain has an 11-residue CDR-L1, and the whole chain from CDR1 to the end is shifted +5 (31-127). The CDR-L1 insertion was absorbed downstream. 4.8 A map; ISOLDE was used.',
 '7e5y': 'X-ray molecular replacement with CB6 (7C01, whose kappa light chain has CDR-L1 of 6) for a lambda chain with CDR-L1 of 8. FR2-CDR2-FR3 is shifted +1 (45-80). The upstream length difference was absorbed downstream.',
}
cases = D[D.group == 'case'].copy()
cases['clue'] = cases.pdb.map(CLUE)
keep = ['mode', 'pdb', 'chain', 'last_author', 'year', 'method', 'res', 'ag_name', 'lo', 'hi', 'win_seq', 'fit_pct', 'unbuilt_win5',
        'first_built_imgt', 'seqres_first_imgt', 'win_contact_nonab', 'v_gene', 'v_id', 'species', 'site_indel', 'artifact',
        'best_ab_template', 'ident', 'tmpl_mismatch', 'irm_records', 'fulltext', 'same_ab_before', 'same_ab_after',
        'ag_6m_otherlab_inscan', 'ag_6m_otherlab_inscan_err', 'ag_first_ab_structure', 'clue']
cases[keep].to_csv(f'{OUT}/tables/case_summary.csv', index=False)
ctrl = D[D.group == 'control']
f = lambda x, d=2: '' if pd.isna(x) else (f'{x:.{d}f}' if isinstance(x, float) else str(x))
lines = ['# Case files: modes A (FR1 N-terminus), B (CDR2 to FR3), C (FR3) and F (CDR1)', '',
         'There are 29 first-occurrence error entries (mostly cryo-EM; one per lab) and 45 matched, correctly modelled controls.',
         'The controls are of three kinds: same antigen, same period and target class, or a "near miss" that shares the local hazard.',
         '',
         '**Column key:**',
         '* **fit**: site map fit, as a percentile within the V domain.',
         '* **unbuilt**: unbuilt residues within ±5 of the site.',
         '* **contact**: the site contacts a non-antibody chain.',
         '* **indel**: an atypical length or indel at the site, read from SEQRES.',
         '* **template**: the closest antibody template named in the paper or PDB record. *mismatch* means it differs in length at the site.',
         '',
         'Scripts are `s01`–`s12`; the tables are in `tables/`. The method-text snippets are in `methods_snippets.txt`.', '']
for mode, g in cases.groupby('mode', sort=False):
    lines += [f'## {mode}', '']
    for r in g.itertuples():
        lines += [f'### {r.pdb} chain {r.chain}: {r.last_author}, {r.year}, {r.method} {f(r.res)} Å',
                  f'* **Antigen:** {r.ag_name}.',
                  f'* **Run:** IMGT {r.lo}–{r.hi}, sequence `{r.win_seq}`.',
                  f'* **Site:** fit {f(r.fit_pct)}; {int(r.unbuilt_win5)} unbuilt residues; first built residue IMGT {r.first_built_imgt} (SEQRES starts at {r.seqres_first_imgt}); antigen contact {int(r.win_contact_nonab) if r.win_contact_nonab == r.win_contact_nonab else "?"}.',
                  f'* **Antibody:** {r.v_gene} (identity {f(r.v_id)}, {r.species}); indel at site {int(r.site_indel) if r.site_indel == r.site_indel else "?"}; gap-placement artifact {int(r.artifact)}.',
                  f'* **Template:** {r.best_ab_template if isinstance(r.best_ab_template, str) else "none named"}; mismatch {int(r.tmpl_mismatch)}. Open-access text: {bool(r.fulltext)}.',
                  f'* **Same V region elsewhere in the PDB:** {f(r.same_ab_before,0) or 0} before, {f(r.same_ab_after,0) or 0} after.',
                  f'* **Same antigen, other labs, ±6 months (in the scan):** {f(r.ag_6m_otherlab_inscan,0)} entries, of which {f(r.ag_6m_otherlab_inscan_err,0)} have errors.',
                  f'* **Clue:** {r.clue}']
        cc = ctrl[ctrl.case_pdb == r.pdb]
        for c in cc.itertuples():
            lines.append(f'* **Control ({c.why}): {c.pdb}/{c.chain}, {c.last_author}, {c.year}.** Window `{c.win_seq}`; fit {f(c.fit_pct)}; '
                         f'{int(c.unbuilt_win5)} unbuilt; first built residue {c.first_built_imgt}; indel {int(c.site_indel) if c.site_indel == c.site_indel else "?"}; '
                         f'template {c.best_ab_template if isinstance(c.best_ab_template, str) else "none named"} (mismatch {int(c.tmpl_mismatch)}).')
        lines.append('')
open(f'{OUT}/case_files.md', 'w').write('\n'.join(lines))
print('written', len(cases), 'cases')
