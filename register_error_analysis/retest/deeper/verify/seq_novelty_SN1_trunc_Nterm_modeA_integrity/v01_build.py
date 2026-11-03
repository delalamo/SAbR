"""Independent per-chain table for verifying SN1 (SEQRES N-terminal truncation vs mode A errors).

Sources (all independent of the seq_novelty investigator's ANARCI re-run and abdb.pkl):
  * detector's own ANARCI numbering of the deposited entity sequence (jobs_*.json: start index, IMGT labels)
  * detector's per-residue records (out/*.jsonl): built residues with ANARCI and SAbR labels
  * shared chains.csv.gz / runs_first_occurrence.csv / runs_all.csv
Also the same for the 1,039 high-resolution template chains (retest/jobs_templates.json, templates.jsonl).
Output: cache/verify_SN1/chain_nterm.pkl, templ_nterm.pkl
"""
import json, gzip, pickle, os, numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
C = f'{SP}/deeper/cache/verify_SN1'; os.makedirs(C, exist_ok=True)


def load_jobs(fns):
    out = {}
    for fn in fns:
        for pdb, chains, anum in json.load(open(fn)):
            for c, doms in anum.items():
                out[(pdb, c)] = doms
    return out


def runs_of(res, minlen=3):
    """mismatch runs over consecutive built residues with an ANARCI label (detector definition)."""
    lab = []
    for r in res:
        a, ai, s, si = r[4], r[5], r[6], r[7]
        if a is None:
            lab.append(None); continue
        lab.append((a, ai or '') != (s, si or ''))
    runs = []; i = 0
    while i < len(lab):
        if lab[i]:
            j = i
            while j + 1 < len(lab) and lab[j + 1]:
                j += 1
            if j - i + 1 >= minlen:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def per_chain(rec, doms):
    d = dict(pdb=rec['pdb'], chain=rec['chain'])
    if doms and len(doms) == 1:
        dm = doms[0]
        d['seq_start_idx'] = dm['start']           # SEQRES residues before the ANARCI V domain
        d['seq_first_imgt'] = dm['pos'][0][1]       # IMGT label of first numbered SEQRES residue
        d['seq_last_imgt'] = dm['pos'][-1][1]
        d['seq_first_aa_idx'] = dm['pos'][0][0]
    res = [r for r in rec.get('residues', []) if r[4] is not None]
    if not res:
        return d
    f = res[0]
    d['built_first_anarci'] = f[4]; d['built_first_sabr'] = f[6]; d['built_first_seqidx'] = f[3]
    d['built_first_delta'] = (f[6] - f[4]) if f[6] is not None else np.nan
    # number of SEQRES V residues (numbered by ANARCI) preceding the first built V residue
    if 'seq_first_aa_idx' in d and f[3] is not None:
        d['n_unbuilt_Nterm_V'] = f[3] - d['seq_first_aa_idx']
    rr = runs_of(rec['residues'])
    # N-terminal run: starts within first 3 labelled residues and lies in FR1 (ANARCI imgt0 <= 26)
    allres = rec['residues']
    nt = []
    for i, j in rr:
        lab_i = [k for k in range(len(allres)) if allres[k][4] is not None]
        first_lab = lab_i[0] if lab_i else 0
        a0 = allres[i][4]
        if a0 is not None and a0 <= 26:
            nt.append((i, j, a0, allres[j][4], (allres[i][6] or 0) - a0, i - first_lab))
    d['n_runs_all'] = len(rr)
    d['nt_run'] = len(nt) > 0
    d['nt_run_at_start'] = any(x[5] <= 2 for x in nt)
    if nt:
        d['nt_run_delta'] = nt[0][4]; d['nt_run_imgt0'] = nt[0][2]; d['nt_run_imgt1'] = nt[0][3]
    return d


if __name__ == '__main__':
    J = load_jobs([f'{SP}/sabr/jobs_new.json', f'{SP}/sabr/jobs_snapshot_rest.json', f'{SP}/sabr/jobs_validation.json'])
    rows = []
    for fn in ['new', 'snapshot_rest', 'validation']:
        for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
            rec = json.loads(l)
            if rec.get('status') != 'ok':
                continue
            d = per_chain(rec, J.get((rec['pdb'], rec['chain'])))
            d['src'] = fn
            rows.append(d)
    T = pd.DataFrame(rows).drop_duplicates(['pdb', 'chain'])
    T.to_pickle(f'{C}/chain_nterm.pkl')
    print('scan chains', T.shape, T.seq_first_imgt.notna().sum())
    # high-resolution templates
    JT = load_jobs([f'{SP}/retest/jobs_templates.json'])
    rows = []
    for l in open(f'{SP}/retest/templates.jsonl'):
        rec = json.loads(l)
        if rec.get('status') != 'ok':
            continue
        rows.append(per_chain(rec, JT.get((rec['pdb'], rec['chain']))))
    TT = pd.DataFrame(rows)
    TT.to_pickle(f'{C}/templ_nterm.pkl')
    print('template chains', TT.shape)
