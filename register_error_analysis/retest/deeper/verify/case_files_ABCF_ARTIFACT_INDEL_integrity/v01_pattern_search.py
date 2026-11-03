"""V01: PDB-wide search for heavy-chain V domains carrying each ANARCI gap pattern (independent of the investigator's
candidate list). Uses the PDB-wide ANARCI cache (seq_novelty/anarci_num.pkl, keyed by SEQRES sequence) and
entries.idx for method/resolution. Pattern = IMGT positions (no insertion code) absent between the first numbered
position and 104, restricted to FR1 11-26 (excluding 10) and FR3 74-100 (73 excluded as heavy norm).
Output: tables/pdbwide_patterns.csv (one row per unique sequence) and printed counts."""
import json, pickle, pandas as pd, collections
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_ABCF_ARTIFACT_INDEL_integrity'
A = pickle.load(open(f'{SP}/deeper/cache/seq_novelty/anarci_num.pkl', 'rb'))
C = json.load(open(f'{SP}/deeper/cache/seq_novelty/cand_seqs.json'))
E = {}
for i, l in enumerate(open('/tmp/supp/pdbwide/entries.idx')):
    if i < 2: continue
    f = l.rstrip('\n').split('\t')
    try: r = float(f[6])
    except: r = None
    E[f[0].lower()] = (r, f[7], f[2])
rows = []
for s, doms in A.items():
    for d in doms or []:
        if d[0] != 'H': continue
        pos = {int(p) for p, ins, a in d[4] if str(ins).strip() == ''}
        ins = sorted({f'{p}{ins.strip()}' for p, ins, a in d[4] if str(ins).strip() != ''})
        if not pos or min(pos) > 20: continue
        miss = set(range(min(pos), 105)) - pos
        fr1 = tuple(sorted(k for k in miss if 11 <= k <= 26))
        fr3 = tuple(sorted(k for k in miss if 74 <= k <= 100))
        fr3ins = tuple(x for x in ins if 74 <= int(''.join(ch for ch in x if ch.isdigit())) <= 100)
        vseq = ''.join(a for p, i2, a in d[4])
        ents = []
        for pc in C.get(s, []):
            p, c = pc.split('_', 1)
            r, m, dt = E.get(p, (None, '', ''))
            ents.append((p, c, r, m, dt))
        rows.append(dict(seq=s, vseq=vseq, fr1=fr1, fr3=fr3, fr3ins=fr3ins, n_chains=len(ents),
                         entries=ents))
D = pd.DataFrame(rows)
def pat(r):
    out = []
    if r.fr1: out.append('FR1:' + '+'.join(map(str, r.fr1)))
    if r.fr3 and not r.fr3ins: out.append('FR3:' + '+'.join(map(str, r.fr3)))
    return ';'.join(out) or 'none'
D['pattern'] = D.apply(pat, axis=1)
def best_xray(ents):
    x = [e for e in ents if e[3].strip() == 'X-RAY DIFFRACTION' and e[2] is not None]
    return min(x, key=lambda e: e[2]) if x else None
D['best_xray'] = D.entries.apply(best_xray)
D['best_xray_res'] = D.best_xray.apply(lambda e: e[2] if e else None)
D.drop(columns=['entries']).to_csv(f'{OUT}/tables/pdbwide_patterns.csv', index=False)
pickle.dump(D, open(f'{SP}/deeper/cache/verify_artifact_indel/pdbwide_patterns.pkl', 'wb'))
for p in ['FR3:84', 'FR3:80+84', 'FR3:84+85', 'FR1:26', 'FR3:81', 'FR3:96', 'FR3:85', 'FR3:82', 'FR3:83', 'FR3:82+83']:
    s = D[D.pattern == p]
    hi = s[s.best_xray_res.notna() & (s.best_xray_res < 2.6)]
    print(f'{p:12s} unique heavy V seqs={len(s):4d}  with X-ray <2.6A: {len(hi):3d}  distinct V: {hi.vseq.nunique():3d}')
