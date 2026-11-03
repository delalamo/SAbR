"""Verifier (confounding lens) for SN2_trunc_FR4_modesDE.
Step 1: per-unit features needed to challenge the finding.
 - reproduce the investigator's unit table / case-control sets (cryo-EM, first-occurrence error units vs clean units)
 - model-level C-terminal end (last modelled residue, ANARCI IMGT of SEQRES mapped through seq_index; and SAbR number)
 - whether the SEQRES really ends after the ANARCI V-domain parse (tail length) -> detects spurious 'truncation'
 - for error units: the D/E run(s) and whether they reach the last modelled residue (chain end)
Writes tables/unit_features.csv.gz
"""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
SN = '/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/seq_novelty_SN2_trunc_FR4_modesDE_confounding/tables'
import os; os.makedirs(OUT, exist_ok=True)

U = pd.read_csv(f'{SN}/units.csv.gz', low_memory=False)
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False)
runs = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
runs_all = pd.read_csv(f'{SP}/deeper/runs_all.csv')

REC = {}
for fn in ['new.jsonl', 'snapshot_rest.jsonl', 'validation.jsonl']:
    for line in open(f'{SP}/sabr/out/{fn}'):
        r = json.loads(line)
        if r.get('status') == 'ok' and r.get('residues'):
            REC[(r['pdb'], r['chain'])] = r['residues']

def chain_end(pdb, ch):
    res = REC.get((pdb, ch))
    if not res: return dict(n_mod=np.nan, mod_last_anarci=np.nan, mod_last_sabr=np.nan, last_seqidx=np.nan, mod_first_anarci=np.nan)
    an = [x[4] for x in res if x[4] is not None]
    sb = [x[6] for x in res if x[6] is not None]
    return dict(n_mod=len(res), mod_last_anarci=max(an) if an else np.nan, mod_last_sabr=max(sb) if sb else np.nan,
                mod_first_anarci=min(an) if an else np.nan, last_seqidx=max(x[3] for x in res if x[3] is not None))

seqmap = C.set_index(['pdb', 'chain'])[['seq', 'vseq', 'unit']]
rows = []
for r in U.itertuples():
    f = dict(unit=r.unit)
    f.update(chain_end(r.pdb, r.chain))
    try:
        s, v = seqmap.loc[(r.pdb, r.chain), ['seq', 'vseq']]
        i = s.find(v) if isinstance(v, str) else -1
        f['seq_len'] = len(s); f['v_end_idx'] = i + len(v) if i >= 0 else np.nan
        f['seq_tail_after_v'] = len(s) - (i + len(v)) if i >= 0 else np.nan
    except KeyError:
        pass
    rows.append(f)
F = pd.DataFrame(rows)
U = U.merge(F, on='unit', how='left')

# runs of each unit (first occurrence) with chain-end check
runs = runs.merge(C[['pdb', 'chain', 'unit']], on=['pdb', 'chain'], how='left')
def run_end_info(r):
    e = chain_end(r.pdb, r.chain)
    # chain end = last modelled residue (ANARCI IMGT number of SEQRES mapped to the model); the run 'touches' the C-terminus
    # if its last ANARCI IMGT position is within 1 position of the last modelled one
    ml = e['mod_last_anarci']
    return pd.Series(dict(run_last_i=r.i1, chain_mod_last_anarci=ml,
                          run_touches_cterm=(r.imgt1 >= ml - 1) if ml == ml else np.nan))
runs = pd.concat([runs, runs.apply(run_end_info, axis=1)], axis=1)
runs.to_csv(f'{OUT}/runs_first_with_cterm.csv.gz', index=False)
isD = runs['mode'].isin(['Dplus_CDR3short', 'Dminus_CDR3long', 'D0']); isE = runs['mode'].eq('E_FR4bulge')
U['D_run_touches_cterm'] = U.unit.map(runs[isD].groupby('unit').run_touches_cterm.any())
U['E_run_touches_cterm'] = U.unit.map(runs[isE].groupby('unit').run_touches_cterm.any())
U['DE_err_chain_mod_last'] = U.unit.map(runs[isD | isE].groupby('unit').chain_mod_last_anarci.min())

# all-run information (incl. copied/artifact) for the 'neither error nor clean' bucket
runs_all = runs_all.merge(C[['pdb', 'chain', 'unit']], on=['pdb', 'chain'], how='left')
U['any_run_modes_all'] = U.unit.map(runs_all.groupby('unit')['mode'].apply(lambda x: ';'.join(sorted(set(x)))))
U.to_csv(f'{OUT}/unit_features.csv.gz', index=False)
E = U[(U.method_u == 'EM')]
print('EM units', len(E), 'err_first', E.err_first_u.sum(), 'clean', E.clean.sum())
t = E[E.seq_fr4_trunc]
print(t[['pdb', 'chain', 'ctype', 'dep_year', 'err_first_u', 'clean', 'modes_u', 'seq_last_imgt', 'mod_last_anarci', 'mod_last_sabr', 'seq_len',
         'seq_tail_after_v', 'D_run_touches_cterm', 'E_run_touches_cterm', 'last_author']].to_string())
