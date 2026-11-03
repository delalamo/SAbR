"""Assemble case files for modes D+, D-, E (chosen cases + matched correctly modelled controls).

Sources: chains.csv / runs (scan); struct_features (s02: coordinates, independent superposition check);
modelling.jsonl (starting models, fitting, revisions); pdb_meta caches (RCSB entry + EMDB features, validation);
junction/tables/template_cdr3.csv (template CDR3 length); antigen_competition pdbwide antigen table (same-antigen
structures +/-6 months); Europe PMC full text (s03; model-building sentences).
Outputs: tables/cases_DE.csv (cases + controls, one row each), tables/case_files_DE.txt (narrative fields + methods excerpts)
"""
import sys, os, re, json, glob, xml.etree.ElementTree as ET
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/scripts')
from common import *
import s02_struct as S2

CASES = {
    'Dplus_CDR3short': [('5ken', 'C'), ('6qx3', 'O'), ('7k8x', 'D'), ('7czv', 'H'), ('7v2a', 'E'), ('7we8', 'L'), ('8gnh', 'H'), ('8xsd', 'G'), ('12ty', 'C')],
    'Dminus_CDR3long': [('6jmr', 'F'), ('6lht', 'L'), ('6orp', 'H'), ('6xja', 'H'), ('7d3k', 'L'), ('7mej', 'B'), ('8gnh', 'L'), ('8jag', 'H'), ('9ma0', 'B')],
    'E_FR4bulge': [('7czr', 'H'), ('7kc1', 'H'), ('7mkl', 'H'), ('7o85', 'B'), ('7t0w', 'H'), ('8rmk', 'L'), ('9dm7', 'L'), ('9pv5', 'L'), ('9zxd', 'A')],
}
PM = '/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
AC = '/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition/tables'

C = chains(); R = runs(); RA = runs_all()
ent = pd.read_csv(f'{D}/entries.csv').set_index('pdb')
F = pd.read_csv(f'{OUT}/tables/struct_features.csv')
ERS = pd.read_pickle(f'{CACHE}/err_struct.pkl')[['pdb', 'chain', 'confirm']]
MOD = modelling()
RC = pd.read_csv(f'{PM}/entry_features_rcsb.csv.gz').set_index('pdb')
EMF = pd.read_csv(f'{PM}/entry_features_emdb.csv.gz').drop_duplicates('pdb').set_index('pdb')
CHF = pd.read_csv(f'{PM}/chain_features_rcsb.csv.gz').set_index(['pdb', 'chain'])
TPL = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/junction/tables/template_cdr3.csv').set_index(['pdb', 'chain'])
AG = pd.read_csv(f'{AC}/pdbwide_ab_entries_antigen.csv.gz')
AG['date'] = pd.to_datetime(AG.date)
agk = AG.set_index('pdb')
L = pd.read_pickle(f'{CACHE}/chains_local.pkl').set_index(['pdb', 'chain'])
DD = pd.concat([pd.read_csv(f) for f in glob.glob('/home/user/SAbR/register_error_analysis/retest/deepdive/tables/deepdive_*_members.csv')])
DD = DD.drop_duplicates(['pdb', 'chain']).set_index(['pdb', 'chain'])
Cx = C.set_index(['pdb', 'chain'])
errk = set(zip(R.pdb, R.chain)); anyk = set(zip(RA.pdb, RA.chain))

KW = re.compile(r'(model build|built|rebuil|initial model|starting model|template|homology|swiss-?model|alphafold|colabfold|abodybuilder|rosetta|coot|phenix|isolde|chimera|fitted|docked|dock|rigid[- ]body|mutat\w+ to|CDR|local refine|focused refine|mask|sequence)', re.I)
def methods_excerpt(doi, maxn=8):
    fn = f'{CACHE}/epmc/{re.sub(r"[^A-Za-z0-9]", "_", str(doi))}'
    if not os.path.exists(fn + '.xml'): return None
    try: root = ET.fromstring(open(fn + '.xml', 'rb').read())
    except Exception: return None
    txt = lambda el: re.sub(r'\s+', ' ', ''.join(el.itertext()))
    secs = [s for s in root.iter('sec') if re.search(r'model|refine|structure determination|cryo|data processing|method',
            (txt(s.find('title')) if s.find('title') is not None else ''), re.I)]
    body = ' '.join(txt(s) for s in secs) if secs else (txt(root.find('.//body')) if root.find('.//body') is not None else '')
    sents = re.split(r'(?<=[.;])\s+(?=[A-Z])', body)
    keep = []
    for s in sents:
        if re.search(r'model build|initial model|starting model|template|homology model|swiss-?model|alphafold|colabfold|abodybuilder|fitted into|docked into|rigid[- ]body|mutated|CDR|manually (re)?built|rebuilt|local refine|focused refine', s, re.I) \
           and re.search(r'fab|antibod|nanobod|vhh|heavy|light|cdr|fv|ig', s, re.I):
            s = s.strip()[:420]
            if s not in keep: keep.append(s)
    return keep[:maxn]

def same_antigen(p, dep, lab, months=6):
    if p not in agk.index: return {}
    k = agk.loc[p]
    keys = set(str(k.tgt_keys).split(';')) if isinstance(k.tgt_keys, str) else set()
    if not keys: return dict(antigen=k.tgt_name)
    d = pd.Timestamp(dep)
    win = AG[(AG.date - d).abs().dt.days.le(30.5 * months) & (AG.pdb != p)]
    win = win[win.tgt_keys.fillna('').apply(lambda s: bool(keys & set(s.split(';'))))]
    other = win[win.lab != k.lab]; same = win[win.lab == k.lab]
    o_err = [x for x in other.pdb if x in ent.index and ent.loc[x, 'err_first']]
    o_known = [x for x in other.pdb if x in ent.index]
    return dict(antigen=k.tgt_name, antigen_class=k.tgt_cls, sameAg_otherlab_entries=len(other), sameAg_otherlab_labs=other.lab.nunique(),
                sameAg_otherlab_scanned=len(o_known), sameAg_otherlab_err=len(o_err), sameAg_samelab_entries=len(same),
                sameAg_otherlab_examples=','.join(other.sort_values('date').pdb.head(8)))

def struct_row(p, c):
    f = F[(F.pdb == p) & (F.chain == c)]
    if len(f): return f.iloc[0].to_dict()
    rec = sabr_records()
    if (p, c) not in rec: return {}
    # compute on the fly (control chains not in the s02 sample)
    from Bio.PDB import MMCIFParser
    REF = getattr(struct_row, 'REF', None)
    if REF is None:
        tp = {}
        for l in open(f'{SP}/retest/templates.jsonl'):
            r = json.loads(l)
            if r.get('status') == 'ok' and (r['pdb'], r['chain']) in {('1n8z', 'A'), ('5dt1', 'H'), ('5dt1', 'L')}: tp[(r['pdb'], r['chain'])] = r['residues']
        REF = {}
        for (pp, cc), ct in {('5dt1', 'H'): 'H', ('1n8z', 'A'): 'K', ('5dt1', 'L'): 'L'}.items():
            parser = MMCIFParser(QUIET=True); st = parser.get_structure(pp, S2.open_cif(S2.cif_path(pp))); ch = next(iter(st))[cc]
            num = {(x[0], x[1]): (x[6], x[7]) for x in tp[(pp, cc)] if x[6] is not None}
            REF[ct] = {num[(r.id[1], r.id[2].strip())]: r['CA'].coord for r in ch if (r.id[1], r.id[2].strip()) in num and 'CA' in r}
        struct_row.REF = REF
    refs = {'_ct': {c: Cx.loc[(p, c), 'ctype']}}; refs.update(REF)
    o = S2.analyse_entry((p, [(c, rec[(p, c)])], list(C[C.pdb == p].chain), refs, {}))
    return o[0] if o else {}

def describe(p, c, role, mode, case_id):
    r = Cx.loc[(p, c)]
    row = dict(case_id=case_id, role=role, mode=mode, pdb=p, chain=c, ctype=r.ctype, method=r.method, res=r.res, deposit=r.deposit,
               release=r.release, hold_days=r.hold_days, last_author=r.last_author, journal=r.journal_abbrev, doi=r.doi, sars2=r.sars2,
               chain_desc=r.chain_desc, species=r.species, vhh=r.vhh, v_gene=r.v_gene, j_gene=r.j_gene, v_id=r.v_id,
               L1=r.L1, L2=r.L2, L3=r.L3, ab_role=r.role, novel_target=r.novel_target, competitors=r.competitors,
               entries_per_paper=r.entries_per_paper, irm=r.irm, fit=r.fit, lq_z=r.lq_z)
    if (p, c) in L.index:
        l = L.loc[(p, c)]; row.update(cdr3_seq=l.cdr3_seq, cdr3_n=l.cdr3_n, fr4_seq=l.fr4_seq, aa117=l.aa117)
    rr = R[(R.pdb == p) & (R.chain == c)]
    row['runs'] = ';'.join(f'{a}-{b}:{d:+.0f}:{m}' for a, b, d, m in zip(rr.imgt0, rr.imgt1, rr.delta, rr['mode']))
    row['err_first'] = (p, c) in errk; row['err_any'] = (p, c) in anyk
    s = struct_row(p, c)
    for k in ['last_built_imgt', 'first_built_imgt', 'unbuilt_100_128', 'fr4_seqres_after118', 'fr4_unbuilt_after118_10', 'w118_ca10', 'slot118_aa',
              'cdr3_ag_contacts', 'v_ag_contacts', 'cdr3_partner_contacts', 'sc_trunc_site', 'sc_trunc_rest', 'b_site_rel', 'fw_rmsd'] + \
             [f'slotshift_{n}' for n in range(116, 125)]:
        row[k] = s.get(k)
    cf = ERS[(ERS.pdb == p) & (ERS.chain == c)]
    row['indep_confirm'] = cf.confirm.iloc[0] if len(cf) else ('n/a (control)' if role != 'case' else None)
    if (p, c) in DD.index:
        row['site_fit_pct'] = DD.loc[(p, c), 'win_fit_pct']; row['unmod_near'] = DD.loc[(p, c), 'unmod_near']
    if (p, c) in CHF.index:
        h = CHF.loc[(p, c)]; row.update(chain_q=h.chain_q, chain_sheet_cov=h.chain_sheet_cov, chain_clash=h.chain_clash_per_res)
    m = MOD.get(p, {})
    irm = m.get('pdbx_initial_refinement_model') or []
    row['start_models'] = ';'.join(sorted({f"{(x.get('accession_code') or '')}{'(' + (x.get('type') or '') + ')' if x.get('type') else ''}" for x in irm if isinstance(x, dict)}))[:200]
    fits = m.get('em_3d_fitting') or []
    row['fit_protocol'] = ';'.join(sorted({str(x.get('ref_protocol')) for x in fits if x.get('ref_protocol')}))
    sw = (m.get('software') or []) + (m.get('em_software') or [])
    row['software'] = ';'.join(sorted({f"{x.get('name')}{' ' + x['version'] if x.get('version') else ''}" for x in sw if x.get('name')}))[:250]
    hist = m.get('pdbx_audit_revision_history') or []
    row['n_revisions'] = len(hist); row['n_major_rev'] = len({x.get('major_revision') for x in hist}) - 1 if hist else 0
    if (p, c) in TPL.index:
        t = TPL.loc[(p, c)]
        if isinstance(t, pd.DataFrame): t = t.iloc[0]
        row.update(template=t.template, template_cdr3=t.t_cdr3_len, template_cdr3_diff=t.dl, template_rel=t.rel)
    if p in RC.index:
        e = RC.loc[p]
        row.update(clashscore=e.vg_clashscore, rama_out=e.vg_percent_ramachandran_outliers, entry_Q=e.vem_Q_score, process_site=e.process_site,
                   microscope=e.im_microscope_model, particles=e.rec_num_particles, symmetry=e.point_symmetry, em_sw=str(e.em_sw)[:150],
                   n_authors=e.n_authors, fund_countries=e.fund_countries, ab_frac_mw=e.ab_frac_mw, tot_mw=e.tot_mw)
    if p in EMF.index:
        e = EMF.loc[p]
        row.update(emdb=e.emdb, local_refine_kw=e.kw_local, composite=e.composite_flag, n_add_maps=e.n_add, n_half=e.n_half, final_sw=e.final_sw,
                   final_sw_ver=e.final_sw_ver, applied_sym=e.applied_sym, map_annot=str(e.map_annot)[:150])
    row.update(same_antigen(p, r.deposit, None))
    # same antibody elsewhere (same V-region sequence) and its status
    same = C[(C.vseq == r.vseq) & (C.pdb != p)]
    row['same_ab_other_entries'] = len(same.pdb.unique())
    row['same_ab_other_entries_err'] = sum(1 for k in zip(same.pdb, same.chain) if k in errk)
    row['same_ab_other_papers'] = same.paper.nunique()
    # partner chains in the entry
    pc = C[(C.pdb == p) & (C.chain != c)]
    row['partner_err_modes'] = ';'.join(sorted({m for q in pc.itertuples() for m in R[(R.pdb == p) & (R.chain == q.chain)]['mode']}))
    return row

def controls_for(p, c, mode):
    r = Cx.loc[(p, c)]
    out = []
    # (a) within-paper: a clean unit (entry with no run at all) of the same chain type from the same paper; else any clean chain in the same paper
    wp = C[(C.paper == r.paper) & (C.unit != r.unit)]
    wp = wp.loc[np.array([k not in anyk for k in zip(wp.pdb, wp.chain)], dtype=bool)]
    wp = wp.assign(entry_clean=[not ent.loc[x, 'err_any'] if x in ent.index else False for x in wp.pdb])
    wp = wp.sort_values('entry_clean', ascending=False)
    if len(wp):
        w = wp[wp.ctype == r.ctype] if (wp.ctype == r.ctype).any() else wp
        w = w.assign(dd=(pd.to_datetime(w.deposit) - pd.Timestamp(r.deposit)).abs()).sort_values(['entry_clean', 'dd'], ascending=[False, True]).iloc[0]
        out.append((w.pdb, w.chain, 'control: same paper, other antibody, correct'))
    # (b) same antigen, other lab, cryo-EM, clean entry, nearest deposition date (within 12 months)
    if p in agk.index and isinstance(agk.loc[p, 'tgt_keys'], str):
        keys = set(agk.loc[p, 'tgt_keys'].split(';'))
        d = pd.Timestamp(r.deposit)
        cand = AG[(AG.pdb != p) & (AG.lab != agk.loc[p, 'lab']) & AG.tgt_keys.fillna('').apply(lambda s: bool(keys & set(s.split(';'))))]
        cand = cand[cand.pdb.isin(ent.index)]
        cand = cand.loc[np.array([(ent.loc[x, 'method'] == r.method) and not ent.loc[x, 'err_any'] for x in cand.pdb], dtype=bool)]
        cand = cand.assign(dd=(cand.date - d).abs().dt.days).sort_values('dd')
        cand = cand[cand.dd <= 366]
        for x in cand.pdb:
            cc = C[(C.pdb == x) & (C.ctype == r.ctype) & (C.vhh == r.vhh)]
            if len(cc):
                out.append((x, cc.chain.iloc[0], 'control: same antigen, other lab, correct')); break
    # (c) fall-back: same year, same chain type, cryo-EM, clean, similar resolution
    if len(out) < 2:
        cc = C[(C.year == r.year) & (C.ctype == r.ctype) & (C.vhh == r.vhh) & (C.method == r.method) & C.pdb.isin([x for x in ent.index if not ent.loc[x, 'err_any']])]
        cc = cc.assign(dr=(cc.res - r.res).abs()).sort_values('dr')
        if len(cc): out.append((cc.pdb.iloc[0], cc.chain.iloc[0], 'control: same year/chain type/resolution, correct'))
    return out

rows, texts = [], []
for mode, lst in CASES.items():
    for i, (p, c) in enumerate(lst, 1):
        cid = f'{mode.split("_")[0]}-{i}'
        row = describe(p, c, 'case', mode, cid); rows.append(row)
        ex = methods_excerpt(row['doi'])
        txt = [f'=== {cid}  {p.upper()} chain {c}  ({mode})  {row["last_author"]}  {row["journal"]}  {row["doi"]}',
               f'deposit {row["deposit"]}  release {row["release"]}  {row["method"]} {row["res"]} A  antigen: {row.get("antigen")}  SARS-CoV-2: {row["sars2"]}',
               f'antibody: {row["chain_desc"]} | {row["species"]} {row["v_gene"]} {row["j_gene"]} V-identity {row["v_id"]} | VHH {row["vhh"]}',
               f'CDR3 {row.get("cdr3_seq")} ({row.get("cdr3_n")})  FR4 {row.get("fr4_seq")}  runs {row["runs"]}  independent check: {row["indep_confirm"]}',
               f'model: built IMGT {row.get("first_built_imgt")}-{row.get("last_built_imgt")}, unbuilt 100-128: {row.get("unbuilt_100_128")}, FR4 SEQRES after 118: {row.get("fr4_seqres_after118")}, unbuilt in next 10: {row.get("fr4_unbuilt_after118_10")}',
               f'W/F118 CA-within-10A {row.get("w118_ca10")} (slot-118 residue {row.get("slot118_aa")}); CDR3 residues contacting antigen {row.get("cdr3_ag_contacts")}, partner {row.get("cdr3_partner_contacts")}; side chains truncated at site {row.get("sc_trunc_site")}',
               f'starting model: {row["irm"]} {row.get("start_models")} | template CDR3 {row.get("template_cdr3")} vs target {row.get("cdr3_n")} ({row.get("template_rel")}) | fit: {row["fit"]} {row.get("fit_protocol")}',
               f'software: {row.get("software")}',
               f'EMDB {row.get("emdb")}: local-refine keyword {row.get("local_refine_kw")}, composite {row.get("composite")}, add. maps {row.get("n_add_maps")}, final {row.get("final_sw")} {row.get("final_sw_ver")}, particles {row.get("particles")}, sym {row.get("applied_sym")}',
               f'validation: clashscore {row.get("clashscore")}, Rama out {row.get("rama_out")}%, entry Q {row.get("entry_Q")}, chain Q {row.get("chain_q")}, site fit pct {row.get("site_fit_pct")}; processed {row.get("process_site")}; revisions {row["n_revisions"]} (major {row["n_major_rev"]})',
               f'same antigen +/-6 mo: other labs {row.get("sameAg_otherlab_labs")} ({row.get("sameAg_otherlab_entries")} entries; scanned {row.get("sameAg_otherlab_scanned")}, with first-occurrence errors {row.get("sameAg_otherlab_err")}); same lab {row.get("sameAg_samelab_entries")}; e.g. {row.get("sameAg_otherlab_examples")}',
               f'same antibody in {row["same_ab_other_entries"]} other entries ({row["same_ab_other_entries_err"]} chains erroneous); partner-chain errors in entry: {row["partner_err_modes"] or "none"}']
        if ex is None: txt.append('methods: no open-access full text in Europe PMC')
        else:
            txt.append('methods excerpts:'); txt += [f'  - {s}' for s in ex]
        for (q, qc, role) in controls_for(p, c, mode):
            cr = describe(q, qc, role, mode, cid); rows.append(cr)
            txt.append(f'  {role}: {q.upper()} {qc} {cr["last_author"]} dep {cr["deposit"]} {cr["res"]} A | CDR3 {cr.get("cdr3_seq")} ({cr.get("cdr3_n")}) FR4 {cr.get("fr4_seq")} | built to {cr.get("last_built_imgt")}, W118 CA10 {cr.get("w118_ca10")}, CDR3-Ag contacts {cr.get("cdr3_ag_contacts")} | start {cr["irm"]} {cr.get("start_models")} tpl CDR3 {cr.get("template_cdr3")} | fit {cr["fit"]} | clash {cr.get("clashscore")} | chainQ {cr.get("chain_q")} | local-refine {cr.get("local_refine_kw")}')
        texts.append('\n'.join(txt))
        print(texts[-1][:300], flush=True)
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/tables/cases_DE.csv', index=False)
open(f'{OUT}/tables/case_files_DE.txt', 'w').write('\n\n'.join(texts) + '\n')
print(T.groupby(['mode', 'role']).size())
