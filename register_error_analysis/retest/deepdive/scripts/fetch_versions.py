"""Fetch EM software names + versions (all categories) for every EM entry in the scan (RCSB GraphQL)."""
import json, urllib.request, time, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
M = pd.read_pickle(f'{SP}/unexpl/master4.pkl'); ids = sorted(set(M[M.method == 'EM'].pdb.str.upper()))
Q = '{ entries(entry_ids: %s) { rcsb_id em_software { category name version } em_3d_fitting { ref_protocol target_criteria } } }'
out = {}
for i in range(0, len(ids), 100):
    q = Q % json.dumps(ids[i:i + 100])
    for k in range(4):
        try:
            r = urllib.request.Request('https://data.rcsb.org/graphql', json.dumps({'query': q}).encode(), {'Content-Type': 'application/json'})
            d = json.load(urllib.request.urlopen(r, timeout=120))['data']['entries']; break
        except Exception as e:
            time.sleep(2 + 2 ** k); d = []
    for e in d or []: out[e['rcsb_id'].lower()] = e
    print(i, len(out), flush=True)
json.dump(out, open(f'{SP}/unexpl/em_software_versions.json', 'w'))
