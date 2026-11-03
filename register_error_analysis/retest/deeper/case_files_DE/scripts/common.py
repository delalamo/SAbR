import json, os, gzip, pandas as pd, numpy as np
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D=f'{SP}/deeper'
CACHE=f'{D}/cache/case_files_DE'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/case_files_DE'
MODES=['Dplus_CDR3short','Dminus_CDR3long','E_FR4bulge']
def chains(): return pd.read_csv(f'{D}/chains.csv.gz')
def runs(): return pd.read_csv(f'{D}/runs_first_occurrence.csv')
def runs_all(): return pd.read_csv(f'{D}/runs_all.csv')
_rec=None
def sabr_records(keys=None):
    global _rec
    if _rec is None:
        _rec={}
        for f in ['validation','snapshot_rest','new']:
            for l in open(f'{SP}/sabr/out/{f}.jsonl'):
                r=json.loads(l)
                if r.get('status')!='ok': continue
                k=(r['pdb'],r['chain'])
                if keys is not None and k not in keys: continue
                _rec[k]=r['residues']
    return _rec
def modelling():
    m={}
    for l in open(f'{SP}/retest/modelling.jsonl'):
        r=json.loads(l); m[r['rcsb_id'].lower()]=r
    return m
