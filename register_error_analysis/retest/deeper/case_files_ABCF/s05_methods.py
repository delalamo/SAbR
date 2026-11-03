"""What the papers say about how each case / control antibody was built, and what its stated template looked like.

  * Europe PMC open-access full text (s02): methods sections -> model-building sentences; workflow flags
  * stated starting models: PDB IDs named in model-building sentences + pdbx_initial_refinement_model (RCSB REST)
  * for every antibody template: best-matching V chain of the case chain type (pdb_seqres + ANARCI numbering),
    identity to the case V region, and its length at the case's error location
      A: first SEQRES IMGT position and N-terminal insertion; F: CDR1 length; B: CDR2 length and FR3 length;
      C: FR3 length; plus CDR3 length
    -> "template length mismatch at the error site"
Outputs: tables/methods_flags.csv, tables/templates.csv, methods_snippets.txt
"""
import json, gzip, os, re, urllib.parse, numpy as np, pandas as pd, xml.etree.ElementTree as ET, pickle
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CA = f'{SP}/deeper/cache/case_files_ABCF'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF'
L = pd.read_csv(f'{OUT}/tables/local_sites.csv')
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
R_all = pd.read_csv(f'{SP}/deeper/runs_all.csv')
ROLES = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition/tables/pdbwide_ab_chain_roles.csv.gz')
ab_entries = set(ROLES.pdb)
AN = pickle.load(open(f'{SP}/deeper/cache/seq_novelty/anarci_num.pkl', 'rb'))
dois = json.load(open(f'{CA}/paper_dois.json'))
pdb2doi = {p: d for d, ps in dois.items() for p in ps}
valid = set()
for line in open('/tmp/supp/pdbwide/entries.idx', encoding='latin-1'):
    if len(line) > 4 and line[4] == '\t': valid.add(line[:4].lower())

txt = lambda el: re.sub(r'\s+', ' ', ''.join(el.itertext()))
def methods_text(d):
    f = f'{CA}/epmc/{urllib.parse.quote(d, safe="")}.xml'
    if not os.path.exists(f): return None
    root = ET.fromstring(open(f, 'rb').read())
    secs = [s for s in root.iter('sec') if re.search(r'method|procedure|model building|structure determination|cryo-?em|data processing|crystallograph',
                                                       (txt(s.find('title')) if s.find('title') is not None else '') + ' ' + (s.get('sec-type') or ''), re.I)]
    body = root.find('.//body')
    m = ' '.join(txt(s) for s in secs) if secs else (txt(body) if body is not None else '')
    # supplementary methods are often in <supplementary-material> captions or <app>
    for a in root.iter('app'): m += ' ' + txt(a)
    return m
BUILD = re.compile(r'model(?:led|ed|ling|ing)?\b|built|build|fitt?ed|fitting|dock|template|initial model|starting model|homology|refine|coot|phenix|'
                   r'isolde|rosetta|alphafold|colabfold|swiss-?model|abodybuilder|immunebuilder|mutated|rigid|chimera|namdinator|modell?er|'
                   r'crystal structure|\bPDB\b|variable domain|\bFv\b|local(?:ly)? refine|focused', re.I)
FLAGS = {
    'coot': r'\bcoot\b', 'phenix': r'\bphenix\b', 'isolde': r'\bisolde\b', 'rosetta': r'\brosetta', 'namdinator_mdff': r'namdinator|\bmdff\b',
    'alphafold': r'alphafold|colabfold', 'homology_tool': r'swiss-?model|modell?er\b|abodybuilder|immunebuilder|rosettaantibody|sabpred|bioluminate|discovery studio|abysis|phyre|i-?tasser|pigs\b|igfold|deepab',
    'homology_model': r'homology model', 'other_fab_template': r'(?:fab|antibody|nanobody|vhh)[^.]{0,150}(?:pdb|template)[^.]{0,60}\b[1-9][a-z0-9]{3}\b',
    'own_crystal': r'crystal structure[^.]{0,80}(?:determined (?:here|in this study)|this study|we determined|solved)|we (?:also )?(?:determined|solved) the crystal structure',
    'rigid_body': r'rigid[- ]body', 'chimera_fit': r'fit in map|fitmap|fit_in_map|chimera', 'mutated': r'mutat\w+ (?:to|into) (?:the|match)|were mutated|was mutated|in silico mutat|sequence was (?:changed|mutated)|replaced (?:with|by) the',
    'cdr_rebuilt': r'cdr\w*[^.]{0,80}(?:built|rebuilt|manually|adjusted|remodel)|(?:built|rebuilt|manually|adjusted|remodel)\w*[^.]{0,80}cdr',
    'local_refine': r'local(?:ly)? refine|focused refine|local refinement|focus(?:ed)? classification|signal subtraction|particle subtraction|symmetry expansion|block-?based',
    'composite_map': r'composite map', 'fv_only': r'variable domain|\bfv\b|constant domains? (?:were|was) (?:removed|deleted|not modell?ed|truncated)',
    'poor_density': r'poor(?:ly)? (?:resolved|defined)|weak density|(?:flexib|disorder)\w+[^.]{0,60}(?:fab|antibody|constant|elbow)|not (?:well )?resolved|lower local resolution|low local resolution',
    'not_modelled': r'not (?:been )?modell?ed|were not built|omitted|truncated|stubbed|removed from the (?:final )?model',
    'molprobity': r'molprobity', 'denovo_tool': r'modelangelo|deeptracer|buccaneer|map[_ ]to[_ ]model|trace[_ ]and[_ ]build',
    'icosahedral': r'icosahedral|block-?based|localized reconstruction|capsid',
}
rows, snippets, tmpl_ids = [], [], {}
entries = L[['pdb', 'group', 'mode', 'case_pdb']].drop_duplicates('pdb')
for r in entries.itertuples():
    d = pdb2doi.get(r.pdb)
    row = dict(pdb=r.pdb, group=r.group, mode=r.mode, case_pdb=r.case_pdb, doi=d)
    try: e = json.load(open(f'{CA}/entry/{r.pdb}.json'))
    except Exception: e = {}
    irm = e.get('pdbx_initial_refinement_model') or []
    row['irm_records'] = ';'.join(f"{x.get('type','')}:{x.get('accession_code','')}:{x.get('source_name','')}" for x in irm)
    ids = {str(x.get('accession_code') or '').lower() for x in irm if x.get('accession_code')}
    ids = {i for i in ids if i in valid}
    fit = e.get('em3d_fitting') or e.get('em_3d_fitting') or []
    row['em_fit_text'] = ' | '.join(str(x.get('details') or '') + ' ' + str(x.get('ref_protocol') or '') for x in fit)[:300]
    sw = e.get('software') or []; emsw = e.get('em_software') or []
    row['software'] = ';'.join(sorted({f"{x.get('name')} {x.get('version') or ''}".strip() for x in sw + emsw if x.get('name')}))[:300]
    m = methods_text(d) if d else None
    row['fulltext'] = m is not None
    if m:
        sents = re.split(r'(?<=[.;])\s+(?=[A-Z(])', m)
        b = [s for s in sents if BUILD.search(s)]
        mb = ' '.join(b)
        for k, pat in FLAGS.items(): row[k] = bool(re.search(pat, mb if k not in ('icosahedral', 'local_refine', 'composite_map') else m, re.I))
        for s in b:
            for x in re.findall(r'(?:PDB|pdb|ID|code|entry|entries|accession)[^.]{0,40}?\b([1-9][A-Za-z0-9]{3})\b', s) + re.findall(r'\(([1-9][A-Za-z0-9]{3})\)', s):
                x = x.lower()
                if x in valid and not x.isdigit() and x != r.pdb: ids.add(x)
        snippets.append(f'### {r.group} {r.mode} {r.pdb} (case {r.case_pdb}) doi {d}\n' + '\n'.join('- ' + s[:500] for s in b if re.search(
            r'model|built|build|fit|dock|template|initial|homology|coot|phenix|isolde|rosetta|alphafold|mutat|rigid|chimera|refine', s, re.I))[:6000] + '\n')
    else:
        snippets.append(f'### {r.group} {r.mode} {r.pdb} (case {r.case_pdb}) doi {d}\n(no open-access full text)\n')
    row['template_ids'] = ';'.join(sorted(ids)); tmpl_ids[r.pdb] = ids
    rows.append(row)
MF = pd.DataFrame(rows); MF.to_csv(f'{OUT}/tables/methods_flags.csv', index=False)
open(f'{OUT}/methods_snippets.txt', 'w').write('\n'.join(snippets))

# ---------- templates: compare with the case V region at the error location
seqres = {}
need = {t for s in tmpl_ids.values() for t in s if t in ab_entries}
cur = None
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz', 'rt') as fh:
    for line in fh:
        if line.startswith('>'): cur = line[1:7] if line[1:5] in need and 'mol:protein' in line else None; continue
        if cur: seqres[cur] = line.strip()
def num_of(seq, light):
    doms = AN.get(seq)
    if doms is None:
        from anarci import run_anarci
        res = run_anarci([('q', seq)], scheme='imgt')
        doms = []
        for (nm, st, en), info in zip(res[1][0] or [], res[2][0] or []):
            doms.append((info['chain_type'], st, en, 0, [(p, ins, aa) for (p, ins), aa in nm if aa != '-']))
        AN[seq] = doms
    out = []
    for d in doms:
        if isinstance(d, (tuple, list)): ct, num = d[0], [(p, str(i).strip(), a) for p, i, a in d[4]]
        else: ct, num = d.get('chain_type'), d.get('num')
        if ct and num and ((ct in 'KL') == light): out.append(num)
    return out
def feats(num):
    pos = [(int(p), i) for p, i, a in num]
    reg = lambda lo, hi: sum(lo <= p <= hi for p, _ in pos)
    seq = ''.join(a for p, i, a in num)
    return dict(first=min(p for p, _ in pos), nins_nterm=sum(p <= 10 and i != '' for p, i in pos), L1=reg(27, 38), L2=reg(56, 65), FR3=reg(66, 104), L3=reg(105, 117), seq=seq)
case_num = {}
for r in L[L.group == 'case'].itertuples():
    v = C[(C.pdb == r.pdb) & (C.chain == r.chain)].iloc[0]
    nn = num_of(v.vseq, v.ctype in 'KL')
    if nn: case_num[(r.pdb, r.mode)] = (feats(nn[0]), v.ctype)
from difflib import SequenceMatcher
T = []
first_runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
for r in L.drop_duplicates(['pdb', 'mode']).itertuples():
    key = (r.case_pdb, r.mode)
    if key not in case_num: continue
    cf, ct = case_num[key]
    if r.group == 'control':
        v = C[(C.pdb == r.pdb) & (C.chain == r.chain)]
        if not len(v): continue
        nn = num_of(v.vseq.iloc[0], ct in 'KL')
        if not nn: continue
        cf = feats(nn[0])
    for t in sorted(tmpl_ids.get(r.pdb, [])):
        if t not in ab_entries: T.append(dict(pdb=r.pdb, group=r.group, mode=r.mode, template=t, is_antibody=False)); continue
        best = None
        for k, s in seqres.items():
            if not k.startswith(t): continue
            for nm in num_of(s, ct in 'KL'):
                f = feats(nm); idn = SequenceMatcher(None, f['seq'], cf['seq']).ratio()
                if best is None or idn > best[0]: best = (idn, k, f)
        if best is None: T.append(dict(pdb=r.pdb, group=r.group, mode=r.mode, template=t, is_antibody=True, note='no V of this chain type')); continue
        idn, k, f = best
        mm = {'A_FR1Nterm': f['first'] != cf['first'] or f['nins_nterm'] != cf['nins_nterm'], 'F_CDR1': f['L1'] != cf['L1'],
              'B_CDR2FR3': f['L2'] != cf['L2'] or f['FR3'] != cf['FR3'], 'C_FR3': f['FR3'] != cf['FR3']}[r.mode]
        terr = R_all[(R_all.pdb == t)]
        T.append(dict(pdb=r.pdb, group=r.group, mode=r.mode, template=t, is_antibody=True, tchain=k, ident=round(idn, 3), same_ab=idn > 0.97,
                      t_first=f['first'], c_first=cf['first'], t_nins=f['nins_nterm'], c_nins=cf['nins_nterm'], t_L1=f['L1'], c_L1=cf['L1'],
                      t_L2=f['L2'], c_L2=cf['L2'], t_FR3=f['FR3'], c_FR3=cf['FR3'], t_L3=f['L3'], c_L3=cf['L3'], length_mismatch_at_site=mm,
                      template_has_run=len(terr) > 0))
TT = pd.DataFrame(T); TT.to_csv(f'{OUT}/tables/templates.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print(MF.groupby('group')[['fulltext']].mean())
print(MF[['group', 'mode', 'pdb', 'fulltext', 'irm_records', 'template_ids']].to_string())
print(TT.to_string())
