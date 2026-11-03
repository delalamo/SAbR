"""Extra chains for DE2: (1) the other V chains in the same entries as the confirmed EM D/E error chains (partner
chains: same map, same modeller), (2) clean chains elsewhere with an identical V sequence (same antibody, other
model), (3) EM first-occurrence error chains of the other modes (A/B/C/F/other), one chain per unit."""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s00_reproduce import load_sample, sets, D, OUT, MODES
from s01_features import build_refs, load_records, target_feats

if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    X = load_sample(); e, mc, wp = sets(X)
    CH = pd.read_csv(f'{D}/chains.csv.gz')
    R = pd.read_csv(f'{D}/runs_first_occurrence.csv'); RA = pd.read_csv(f'{D}/runs_all.csv')
    ekeys = set(zip(e.pdb, e.chain))
    part = CH[CH.pdb.isin(e.pdb) & ~pd.Series(list(zip(CH.pdb, CH.chain))).isin(ekeys).values].assign(set='partner')
    ev = e.merge(CH[['pdb', 'chain', 'vseq']], on=['pdb', 'chain'])
    same = CH[(CH.err_any == False) & CH.vseq.isin(ev.vseq) & ~CH.pdb.isin(e.pdb)].assign(set='same_vseq_clean')
    oth = R[~R['mode'].isin(MODES) & (R.method == 'EM')][['pdb', 'chain', 'mode']].drop_duplicates(['pdb', 'chain'])
    om = CH.merge(oth, on=['pdb', 'chain']).sort_values(['pdb', 'chain']).drop_duplicates('unit').assign(set='other_mode_error')
    S = pd.concat([part, same, om])
    S.drop(columns=['seq']).to_csv(f'{OUT}/extra_sample.csv', index=False)
    print(S.set.value_counts())
    panel, REF = build_refs()
    runpos = {}
    for r in RA.itertuples():
        runpos.setdefault((r.pdb, r.chain), set()).update(range(int(r.imgt0) - 1, int(r.imgt1) + 2))
    keys = set(zip(S.pdb, S.chain)); rec = load_records(keys)
    ctype = dict(zip(zip(CH.pdb, CH.chain), CH.ctype))
    jobs = [(p, c, rec[(p, c)], runpos.get((p, c), set()), ctype[(p, c)], REF[ctype[(p, c)]], panel)
            for p, c in sorted(keys) if (p, c) in rec]
    print('jobs', len(jobs), flush=True)
    res = []; t0 = time.time()
    with ProcessPoolExecutor(2) as ex:
        for i, o in enumerate(ex.map(target_feats, jobs, chunksize=4)):
            res.append(o)
            if i % 100 == 0: print(i, round(time.time() - t0), flush=True)
    F = pd.DataFrame(res)
    # Q-score
    from s01_features import FW
    q = {}
    for l in open('/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/retest/local_all.jsonl'):
        r = json.loads(l)
        if (r['pdb'], r['chain']) in keys and 'Q_SCORE' in r['feats']:
            am = r['auth_map']; q[(r['pdb'], r['chain'])] = {am[int(k) - 1]: v for k, v in r['feats']['Q_SCORE'].items() if v is not None}
    qr = []
    for (p, c), qq in q.items():
        if (p, c) not in rec: continue
        rp = runpos.get((p, c), set())
        fw = [qq.get(f'{x[0]}{x[1]}') for x in rec[(p, c)] if x[4] in FW and not x[5] and x[4] not in rp]
        fw = [z for z in fw if z is not None]
        qr.append(dict(pdb=p, chain=c, q_fw=np.mean(fw) if fw else np.nan))
    F = F.merge(pd.DataFrame(qr), on=['pdb', 'chain'], how='left')
    F.to_csv(f'{OUT}/features_extra.csv', index=False)
    print(F.status.value_counts())
