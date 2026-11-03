"""Fetch everything needed for the case files (cases + controls), official APIs only, cached.

Per entry:
  * RCSB REST core entry JSON (software, refinement, initial model, EM fitting text, revision history, citation)
  * mmCIF (files.rcsb.org) - reused from earlier caches when present
  * wwPDB validation XML (files.rcsb.org/pub/pdb/validation_reports) - per-residue outliers, clashes, Q-score/RSRZ
  * EMDB entry JSON and EMDB validation-analysis JSON (per-residue local resolution, Q-score)
Per paper:
  * Europe PMC open-access full-text XML (PMC id from ../paper_meta, else a DOI search; preprint PPR ids also tried)
Cache: scratchpad/deeper/cache/case_files_ABCF/
"""
import json, os, gzip, shutil, time, urllib.request, urllib.parse, urllib.error, sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CA = f'{SP}/deeper/cache/case_files_ABCF'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF'
for d in ['entry', 'cif', 'vrpt', 'emdb', 'emdb_va', 'epmc']: os.makedirs(f'{CA}/{d}', exist_ok=True)
cases = pd.read_csv(f'{OUT}/tables/cases.csv'); ctrls = pd.read_csv(f'{OUT}/tables/controls.csv')
ids = sorted(set(cases.pdb) | set(ctrls.ctrl_pdb) | set(sys.argv[1:]))
print('entries', len(ids), flush=True)

def get(url, tries=4, timeout=120):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'SAbR-register-analysis'}), timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 410): return None
            time.sleep(2 + 2 ** k)
        except Exception:
            time.sleep(2 + 2 ** k)
    return None

def save(path, data, gz=False):
    if data is None: return False
    with open(path, 'wb') as f: f.write(data)
    return True

def fetch_entry(p):
    P = p.upper(); out = {}
    f = f'{CA}/entry/{p}.json'
    if not os.path.exists(f): save(f, get(f'https://data.rcsb.org/rest/v1/core/entry/{P}'))
    # mmCIF: reuse caches
    f = f'{CA}/cif/{p}.cif.gz'
    if not os.path.exists(f):
        src = [f'{SP}/deeper/cache/struct_unusual/cif/{p}.cif.gz', f'{SP}/sabr/dl/{p}.cif']
        if os.path.exists(src[0]): shutil.copy(src[0], f)
        elif os.path.exists(src[1]):
            with open(src[1], 'rb') as a, gzip.open(f, 'wb') as b: shutil.copyfileobj(a, b)
        else: save(f, get(f'https://files.rcsb.org/download/{P}.cif.gz', timeout=300))
    f = f'{CA}/vrpt/{p}_validation.xml.gz'
    if not os.path.exists(f):
        src = f'{SP}/deeper/cache/struct_unusual/vrpt/{p}_validation.xml.gz'
        if os.path.exists(src): shutil.copy(src, f)
        else: save(f, get(f'https://files.rcsb.org/pub/pdb/validation_reports/{p[1:3]}/{p}/{p}_validation.xml.gz', timeout=300))
    # EMDB
    try: e = json.load(open(f'{CA}/entry/{p}.json'))
    except Exception: return p, 'no entry json'
    emdb = ((e.get('rcsb_entry_container_identifiers') or {}).get('emdb_ids') or [])
    for em in emdb[:1]:
        f = f'{CA}/emdb/{em}.json'
        if not os.path.exists(f):
            src = f'{SP}/deeper/cache/pdb_meta/emdb/entry_{em}.json.gz'
            if os.path.exists(src):
                with gzip.open(src) as a: save(f, a.read())
            else: save(f, get(f'https://www.ebi.ac.uk/emdb/api/entry/{em}'))
        f = f'{CA}/emdb_va/{em}.json'
        if not os.path.exists(f): save(f, get(f'https://www.ebi.ac.uk/emdb/api/analysis/{em}', timeout=300))
    return p, 'ok'

with ThreadPoolExecutor(3) as ex:
    for p, s in ex.map(fetch_entry, ids): print(p, s, flush=True)

# ---------- papers
PM = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta/paper_meta.csv')
pm = PM.set_index(PM.doi.str.lower())
dois = {}
for p in ids:
    try: e = json.load(open(f'{CA}/entry/{p}.json'))
    except Exception: continue
    c = [x for x in (e.get('citation') or []) if x.get('id') == 'primary']
    d = (c[0].get('pdbx_database_id_DOI') if c else None)
    if d: dois.setdefault(d.lower(), []).append(p)
print('papers', len(dois), flush=True)

def fetch_paper(d):
    key = urllib.parse.quote(d, safe='')
    f = f'{CA}/epmc/{key}.xml'
    if os.path.exists(f) or os.path.exists(f + '.none'): return d, 'cached'
    pmcid = pm.pmcid.get(d) if d in pm.index else None
    cand = []
    if isinstance(pmcid, str): cand.append(pmcid)
    else:
        r = get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={urllib.parse.quote(f"DOI:{chr(34)}{d}{chr(34)}")}&format=json&resultType=lite')
        if r:
            for x in json.loads(r)['resultList']['result']:
                if x.get('pmcid'): cand.append(x['pmcid'])
    if d in pm.index and isinstance(pm.ppr_ids.get(d), str):
        cand += [x for x in pm.ppr_ids.get(d).split(';') if x]
    for c in cand:
        x = get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{c}/fullTextXML')
        if x and b'<body' in x:
            save(f, x); return d, f'ok {c}'
    open(f + '.none', 'w').write(';'.join(cand)); return d, 'no OA full text'

with ThreadPoolExecutor(3) as ex:
    for d, s in ex.map(fetch_paper, list(dois)): print(d, s, flush=True)
json.dump(dois, open(f'{CA}/paper_dois.json', 'w'))
