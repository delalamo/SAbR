"""Europe PMC full texts (open-access XML) for every paper behind a D+/D-/E error, plus EMDB entry JSON for the error entries.
Cached under scratchpad/deeper/cache/case_files_DE/{epmc,emdb}. Official REST APIs only."""
import sys, os, json, re, time, urllib.request, urllib.parse
sys.path.insert(0, '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE/scripts')
from common import *
from concurrent.futures import ThreadPoolExecutor
EP = f'{CACHE}/epmc'; os.makedirs(EP, exist_ok=True)

def get(url, tries=4):
    for k in range(tries):
        try: return urllib.request.urlopen(url, timeout=60).read()
        except Exception as e:
            if '404' in str(e): return None
            time.sleep(1 + 2 ** k)
    return None

def fetch(doi):
    fn = f'{EP}/{re.sub(r"[^A-Za-z0-9]", "_", doi)}'
    if os.path.exists(fn + '.json'): return json.load(open(fn + '.json'))
    q = urllib.parse.quote(f'DOI:"{doi}"')
    r = get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&format=json&resultType=core')
    meta = {'doi': doi}
    if r:
        res = json.loads(r)['resultList']['result']
        if res:
            x = res[0]
            meta.update(pmcid=x.get('pmcid'), pmid=x.get('pmid'), title=x.get('title'), journal=(x.get('journalInfo') or {}).get('journal', {}).get('title'),
                        pubdate=x.get('firstPublicationDate'), oa=x.get('isOpenAccess'), authors=x.get('authorString'),
                        affil=[a.get('affiliation') for a in (x.get('authorList') or {}).get('author', [])[-3:] if isinstance(a, dict)],
                        grants=[g.get('agency') for g in (x.get('grantsList') or {}).get('grant', [])])
            if x.get('pmcid'):
                t = get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{x["pmcid"]}/fullTextXML')
                if t:
                    open(fn + '.xml', 'wb').write(t); meta['fulltext'] = True
    json.dump(meta, open(fn + '.json', 'w'))
    return meta

if __name__ == '__main__':
    R = runs(); E = R[R['mode'].isin(MODES)]
    C = chains()
    papers = sorted({d for d in C[C.pdb.isin(E.pdb)].doi.dropna() if d.startswith('10.')})
    print('papers', len(papers))
    with ThreadPoolExecutor(3) as ex: out = list(ex.map(fetch, papers))
    print(sum(1 for o in out if o.get('fulltext')), 'full texts')
