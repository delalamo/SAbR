"""Fetch mmCIFs (if not cached in sabr/dl) and wwPDB validation XML reports for every sample entry."""
import os, subprocess, pandas as pd
from concurrent.futures import ThreadPoolExecutor
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
C = f'{SP}/deeper/cache/struct_unusual'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
os.makedirs(f'{C}/cif', exist_ok=True); os.makedirs(f'{C}/vrpt', exist_ok=True)
s = pd.read_csv(f'{OUT}/sample.csv')
pdbs = sorted(set(s.pdb))


def get(url, fn):
    if os.path.exists(fn) and os.path.getsize(fn) > 0:
        return 'cached'
    r = subprocess.run(['curl', '-sS', '-f', '--retry', '3', '-o', fn + '.part', url], capture_output=True)
    if r.returncode == 0:
        os.rename(fn + '.part', fn); return 'ok'
    return 'fail ' + r.stderr.decode()[:80]


def job(p):
    out = []
    if not os.path.exists(f'{SP}/sabr/dl/{p}.cif'):
        out.append(get(f'https://files.rcsb.org/download/{p}.cif.gz', f'{C}/cif/{p}.cif.gz'))
    out.append(get(f'https://files.rcsb.org/pub/pdb/validation_reports/{p[1:3]}/{p}/{p}_validation.xml.gz',
                   f'{C}/vrpt/{p}_validation.xml.gz'))
    return p, out


with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(job, pdbs))
bad = [r for r in res if any(x.startswith('fail') for x in r[1])]
print(len(res), 'entries;', len(bad), 'with failures', bad[:10])
