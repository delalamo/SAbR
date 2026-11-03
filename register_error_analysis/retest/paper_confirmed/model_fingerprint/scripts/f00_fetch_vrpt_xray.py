"""Fetch wwPDB validation XML for every X-ray entry in the scan that is not cached yet (EM entries are all cached by
verify/struct_unusual_antibody_vs_rest_of_model_confounding). Cache: scratchpad/paperfp/vrpt."""
import os, subprocess, pandas as pd
from concurrent.futures import ThreadPoolExecutor
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CACHES = [f'{SP}/deeper/cache/{d}/vrpt' for d in ('struct_unusual', 'verify_su_gap', 'verify_bsb', 'case_files_ABCF')] + [f'{SP}/paperfp/vrpt']
NEW = f'{SP}/paperfp/vrpt'
def have(p):
    return any(os.path.exists(f'{d}/{p}_validation.xml.gz') and os.path.getsize(f'{d}/{p}_validation.xml.gz') > 0 for d in CACHES)
def get(p):
    if have(p): return 'cached'
    fn = f'{NEW}/{p}_validation.xml.gz'
    url = f'https://files.rcsb.org/pub/pdb/validation_reports/{p[1:3]}/{p}/{p}_validation.xml.gz'
    r = subprocess.run(['curl', '-sS', '-f', '--retry', '3', '-o', fn + '.part', url], capture_output=True)
    if r.returncode == 0:
        os.rename(fn + '.part', fn); return 'ok'
    return 'fail'
if __name__ == '__main__':
    C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'method'])
    ps = sorted(set(C[C.method == 'X-ray'].pdb))
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(get, ps))
    print(pd.Series(res).value_counts().to_dict())
