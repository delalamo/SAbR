"""Fetch wwPDB validation XML for every cryo-EM entry in entries.csv (population-wide re-analysis).
Reuses the struct_unusual cache when present; new files go to cache/verify_su_gap/vrpt."""
import os, subprocess, pandas as pd
from concurrent.futures import ThreadPoolExecutor
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OLD = f'{SP}/deeper/cache/struct_unusual/vrpt'
NEW = f'{SP}/deeper/cache/verify_su_gap/vrpt'
os.makedirs(NEW, exist_ok=True)
e = pd.read_csv(f'{SP}/deeper/entries.csv')
pdbs = sorted(e.loc[e.method == 'EM', 'pdb'])


def job(p):
    if os.path.exists(f'{OLD}/{p}_validation.xml.gz') or os.path.exists(f'{NEW}/{p}_validation.xml.gz'):
        return p, 'cached'
    fn = f'{NEW}/{p}_validation.xml.gz'
    url = f'https://files.rcsb.org/pub/pdb/validation_reports/{p[1:3]}/{p}/{p}_validation.xml.gz'
    r = subprocess.run(['curl', '-sS', '-f', '--retry', '3', '--max-time', '120', '-o', fn + '.part', url],
                       capture_output=True)
    if r.returncode == 0:
        os.rename(fn + '.part', fn); return p, 'ok'
    return p, 'fail ' + r.stderr.decode()[:80]


with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(job, pdbs))
bad = [r for r in res if r[1].startswith('fail')]
print(len(res), 'entries;', sum(r[1] == 'ok' for r in res), 'downloaded;', len(bad), 'failed', bad[:10])
