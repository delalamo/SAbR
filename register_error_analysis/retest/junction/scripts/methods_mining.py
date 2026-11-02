"""What do the papers behind CDR3/FR4-error structures say about how the antibody was built?

Cases: papers of entries with a CDR3-FR4 junction or FR4-bulge run (AlphaFold-independent scan; ../../unexplained).
Controls: papers of EM antibody entries from the same years in which no antibody chain has any disagreement run.
Full texts: Europe PMC REST API (open-access PMC XML only). The methods section(s) are extracted; tool and workflow
keywords are flagged; the model-building sentences are saved for reading.
Outputs: ../tables/methods_papers.csv (flags per paper), methods_snippets_cases.txt, methods_keywords.csv
"""
import json, re, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import fisher_exact
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
O = '/home/user/SAbR/register_error_analysis/retest/junction/tables'
R = pd.read_csv(f'{SP}/unexpl/runs_origin_v2.csv'); R['cls'] = R.origin.str.split(' ').str[0]
M = pd.read_pickle(f'{SP}/unexpl/master4.pkl')
doi, dep = {}, {}
for line in open('/tmp/supp/pdbwide/cite_all.jsonl'):
    e = json.loads(line); i = e['rcsb_id'].lower()
    doi[i] = ((e.get('rcsb_primary_citation') or {}).get('pdbx_database_id_DOI') or '').lower() or None
case_e = set(R[R.cls.isin(['5', '6'])].pdb)
flag_e = set(R.pdb)
yrs = M[M.pdb.isin(case_e)].year
ctrl_pool = M[(M.method == 'EM') & M.year.between(yrs.min(), yrs.max()) & ~M.pdb.isin(flag_e)].drop_duplicates('pdb')
ctrl_pool = ctrl_pool[ctrl_pool.pdb.map(doi).notna()]
cases = sorted({doi[p] for p in case_e if doi.get(p)})
ctrl = [d for d in ctrl_pool.pdb.map(doi).drop_duplicates().sample(frac=1, random_state=0) if d not in cases][:300]
print('case papers with DOI', len(cases), '(entries', len(case_e), ') | control papers', len(ctrl), flush=True)

def get(url, tries=4):
    for k in range(tries):
        try: return urllib.request.urlopen(url, timeout=60).read()
        except Exception as e:
            if '404' in str(e): return None
            time.sleep(1 + 2 ** k)
    return None
def fetch(d):
    q = urllib.parse.quote(f'DOI:"{d}"')
    r = get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&format=json&resultType=lite')
    if not r: return dict(doi=d, status='search failed')
    res = json.loads(r)['resultList']['result']
    pmcid = next((x.get('pmcid') for x in res if x.get('pmcid')), None)
    if not pmcid: return dict(doi=d, status='no PMC full text')
    x = get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML')
    if not x: return dict(doi=d, pmcid=pmcid, status='full text unavailable')
    try: root = ET.fromstring(x)
    except Exception: return dict(doi=d, pmcid=pmcid, status='xml parse failed')
    txt = lambda el: re.sub(r'\s+', ' ', ''.join(el.itertext()))
    secs = [s for s in root.iter('sec') if re.search(r'method|procedure|model building|structure determination|cryo-?em|data processing',
                                                       (txt(s.find('title')) if s.find('title') is not None else '') + ' ' + (s.get('sec-type') or ''), re.I)]
    body = root.find('.//body')
    mtxt = ' '.join(txt(s) for s in secs) if secs else ''
    full = txt(body) if body is not None else ''
    return dict(doi=d, pmcid=pmcid, status='ok', has_methods_sec=bool(secs), methods=mtxt or full, full_len=len(full))
with ThreadPoolExecutor(4) as ex:
    papers = list(ex.map(fetch, [(d) for d in cases + ctrl]))
P = pd.DataFrame(papers); P['group'] = np.where(P.doi.isin(cases), 'case', 'control')
print(P.groupby(['group', 'status']).size().to_string(), flush=True)
K = {  # tool / workflow keywords
    'Coot': r'\bcoot\b', 'Phenix real-space refine': r'real[_ -]?space[_ -]?refine', 'Phenix (any)': r'\bphenix\b', 'ISOLDE': r'\bisolde\b',
    'Rosetta': r'\brosetta', 'Namdinator': r'namdinator', 'MDFF': r'\bmdff\b|molecular dynamics flexible fitting', 'ModelAngelo': r'modelangelo',
    'DeepTracer': r'deeptracer', 'Buccaneer': r'buccaneer', 'map_to_model/trace_and_build': r'map[_ ]to[_ ]model|trace[_ ]and[_ ]build',
    'dock_and_rebuild': r'dock[_ ]and[_ ]rebuild', 'SWISS-MODEL': r'swiss[- ]?model', 'MODELLER': r'\bmodell?er\b', 'Phyre2': r'phyre',
    'I-TASSER': r'i-?tasser', 'ABodyBuilder/SAbPred': r'abodybuilder|sabpred|immunebuilder', 'RosettaAntibody': r'rosettaantibody|rosetta antibody',
    'PIGS': r'\bpigs\b', 'AlphaFold/ColabFold': r'alphafold|colabfold', 'Sculptor/Chainsaw': r'sculptor|chainsaw',
    'Chimera fit in map': r'fit in map|fitmap|fit_in_map', 'homology model': r'homology model', 'rigid body': r'rigid[- ]body',
    'mutated (to sequence)': r'mutat\w+ (?:to|into) (?:the|match)|were mutated|was mutated|sequence was (?:changed|mutated)',
    'CDR loops (re)built': r'cdr\w*[^.]{0,80}(?:built|rebuilt|manually|adjusted|remodel)|(?:built|rebuilt|manually|adjusted|remodel)\w*[^.]{0,80}cdr',
    'template PDB of another Fab': r'(?:fab|antibody|nanobody)[^.]{0,120}(?:pdb|template)[^.]{0,40}\b\d[a-z0-9]{3}\b',
    'only Fv / variable domains modelled': r'variable domain|\bfv\b',
}
def flags(t):
    t = (t or '').lower(); return {k: bool(re.search(v, t)) for k, v in K.items()}
ok = P[P.status == 'ok'].copy()
F = pd.DataFrame([flags(t) for t in ok.methods]); F.index = ok.index; ok = pd.concat([ok, F], axis=1)
rows = []
for k in K:
    a = ok[ok.group == 'case'][k]; b = ok[ok.group == 'control'][k]
    p = fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1]
    rows.append(dict(keyword=k, case_pct=round(100 * a.mean(), 1), control_pct=round(100 * b.mean(), 1), case_n=int(a.sum()), control_n=int(b.sum()), p=round(p, 4)))
T = pd.DataFrame(rows).sort_values('p'); pd.set_option('display.width', 200)
print(f"\npapers with OA full text: case {int((ok.group=='case').sum())}, control {int((ok.group=='control').sum())}")
print(T.to_string(index=False)); T.to_csv(f'{O}/methods_keywords.csv', index=False)
ok.drop(columns=['methods']).to_csv(f'{O}/methods_papers.csv', index=False)
# snippets: model-building sentences of case papers
pat = re.compile(r'[^.]*(?:fab|antibod|nanobod|fv\b|cdr|heavy chain|light chain)[^.]*(?:model|built|build|dock|fit|template|refine|mutat|homolog)[^.]*\.|'
                 r'[^.]*(?:model|built|build|dock|fit|template|refine|mutat|homolog)[^.]*(?:fab|antibod|nanobod|fv\b|cdr|heavy chain|light chain)[^.]*\.', re.I)
ent = {}
for p in case_e:
    if doi.get(p): ent.setdefault(doi[p], []).append(p)
with open(f'{O}/methods_snippets_cases.txt', 'w') as f:
    for r in ok[ok.group == 'case'].itertuples():
        sn = [s.strip() for s in pat.findall(r.methods)][:12]
        f.write(f"### {r.doi}  {r.pmcid}  entries: {', '.join(sorted(ent.get(r.doi, [])))}\n" + '\n'.join('- ' + s[:500] for s in sn) + '\n\n')
print('snippets written')
