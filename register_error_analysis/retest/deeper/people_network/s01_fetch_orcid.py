#!/usr/bin/env python3
"""Fetch PDB audit authors with ORCID (RCSB GraphQL) for the 4,085 antibody entries."""
import json, os, time, urllib.request
import pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper'
OUTF = f'{SP}/cache/people_network/audit_author_orcid.jsonl'
E = pd.read_csv(f'{SP}/entries.csv')
ids = [p.upper() for p in E.pdb]
done = set()
if os.path.exists(OUTF):
    done = {json.loads(l)['rcsb_id'] for l in open(OUTF)}
todo = [i for i in ids if i not in done]
Q = 'query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id audit_author{name pdbx_ordinal identifier_ORCID}}}'
with open(OUTF, 'a') as fo:
    for k in range(0, len(todo), 200):
        b = todo[k:k + 200]
        req = urllib.request.Request('https://data.rcsb.org/graphql', data=json.dumps({'query': Q, 'variables': {'ids': b}}).encode(),
                                     headers={'Content-Type': 'application/json'})
        for t in range(4):
            try:
                r = json.load(urllib.request.urlopen(req, timeout=120)); break
            except Exception as ex:
                print('retry', ex); time.sleep(5)
        for x in r['data']['entries'] or []:
            fo.write(json.dumps(x) + '\n')
        print(k, len(todo), flush=True)
