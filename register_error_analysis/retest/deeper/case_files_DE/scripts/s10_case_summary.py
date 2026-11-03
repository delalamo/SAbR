"""Per-mode commonality table for the chosen case files vs their matched correct controls (descriptive; n=9 per mode)."""
import sys, re
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/scripts')
from common import *
T = pd.read_csv(f'{OUT}/tables/cases_DE.csv')
txt = open(f'{OUT}/tables/case_files_DE.txt').read()
blocks = {b.split()[0]: b for b in txt.split('=== ')[1:]}
T['is_case'] = T.role == 'case'
def methods_flag(cid, pat):
    b = blocks.get(cid, '')
    m = b.split('methods excerpts:')[1] if 'methods excerpts:' in b else ''
    m = m.split('  control:')[0]
    return bool(re.search(pat, m, re.I)) if m else np.nan
T['oa_methods'] = [('methods excerpts:' in blocks.get(c, '')) for c in T.case_id]
T['m_other_fab_template'] = [methods_flag(c, r'PDB ID|PDB code|template|homolog|swiss-?model|from the PDB') if r else np.nan for c, r in zip(T.case_id, T.is_case)]
T['m_local_refine'] = [methods_flag(c, r'local(ly)? refine|focused refine|local mask|focused 3D') if r else np.nan for c, r in zip(T.case_id, T.is_case)]
T['pdb_template'] = T.irm.eq('experimental (PDB) model')
T['tpl_cdr3_mismatch'] = T.template_cdr3_diff.abs().ge(2)
T['rigid'] = T.fit.eq('rigid body only')
T['fr4_model_trunc'] = T.last_built_imgt.lt(125)
T['cdr3_contacts_ag'] = T.cdr3_ag_contacts.gt(0)
CH = chains(); R = runs()
ER = pd.read_pickle(f'{CACHE}/err_struct.pkl'); bad = set(zip(ER[ER.confirm == 'not confirmed'].pdb, ER[ER.confirm == 'not confirmed'].chain))
errc = {(r.pdb, r.chain) for r in R.itertuples() if not (r.mode in MODES and (r.pdb, r.chain) in bad)}
ct = dict(zip(zip(CH.pdb, CH.chain), CH.ctype))
def partner(p, c):
    me = 'H' if ct.get((p, c)) == 'H' else 'L'
    return any(((p, cc) in errc) for cc in CH[CH.pdb == p].chain if ('H' if ct[(p, cc)] == 'H' else 'L') != me)
T['partner_err'] = [partner(p, c) for p, c in zip(T.pdb, T.chain)]
T['crowded_antigen'] = T.sameAg_otherlab_labs.fillna(0).ge(10)
T['sars2'] = T.sars2.astype(bool)
T['pdbj'] = T.process_site.eq('PDBJ')
T['local_kw'] = T.local_refine_kw.astype(str).eq('True')
agg = {'n': ('pdb', 'size'), 'labs': ('last_author', 'nunique'), 'years': ('deposit', lambda s: f"{s.str[:4].min()}-{s.str[:4].max()}"),
       'EM': ('method', lambda s: (s == 'EM').mean()), 'res_med': ('res', 'median'), 'sars2': ('sars2', 'mean'),
       'pdb_template': ('pdb_template', 'mean'), 'tpl_cdr3_mismatch': ('tpl_cdr3_mismatch', 'mean'), 'rigid_only': ('rigid', 'mean'),
       'EMDB_local_refine_kw': ('local_kw', 'mean'), 'fr4_model_trunc': ('fr4_model_trunc', 'mean'), 'cdr3_contacts_ag': ('cdr3_contacts_ag', 'mean'),
       'partner_chain_err': ('partner_err', 'mean'), 'crowded_antigen(>=10 labs/6mo)': ('crowded_antigen', 'mean'),
       'same_ab_elsewhere': ('same_ab_other_entries', lambda s: (s > 0).mean()), 'clashscore_med': ('clashscore', 'median'),
       'chainQ_med': ('chain_q', 'median'), 'fw_rmsd_med': ('fw_rmsd', 'median'), 'w118_ca10_med': ('w118_ca10', 'median'),
       'PDBj_processed': ('pdbj', 'mean'), 'oa_methods': ('oa_methods', 'mean')}
T['grp'] = np.where(T.is_case, 'case', 'control')
S = T.groupby(['mode', 'grp']).agg(**agg).round(2)
S.to_csv(f'{OUT}/tables/case_commonalities.csv')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(S.T.to_string())
c = T[T.is_case]
print(c.groupby('mode')[['m_other_fab_template', 'm_local_refine']].agg(lambda s: f'{int(s.sum())}/{s.notna().sum()}'))
# compact case table
cols = ['case_id', 'pdb', 'chain', 'ctype', 'deposit', 'res', 'last_author', 'journal', 'antigen', 'sars2', 'species', 'v_gene', 'j_gene', 'cdr3_seq', 'cdr3_n', 'fr4_seq',
        'runs', 'indep_confirm', 'last_built_imgt', 'w118_ca10', 'slot118_aa', 'cdr3_ag_contacts', 'irm', 'start_models', 'template_cdr3', 'fit', 'local_refine_kw',
        'clashscore', 'chain_q', 'fw_rmsd', 'sameAg_otherlab_labs', 'sameAg_otherlab_err', 'same_ab_other_entries', 'partner_err_modes']
c[cols].to_csv(f'{OUT}/tables/case_table_DE.csv', index=False)
print(c[cols].to_string()[:6000])
