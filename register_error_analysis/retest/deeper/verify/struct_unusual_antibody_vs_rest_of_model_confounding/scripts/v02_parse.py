"""Parse wwPDB validation XML for every cryo-EM entry into per-(entry, chain, category, distance-bin) tallies.
Categories: V  = antibody V-region residue (residue present in the SAbR/ANARCI record of a chain in chains.csv)
            C  = non-V residue on an antibody V chain (constant domain, linker, tag)
            O  = residue on any other chain (antigen, receptor, other proteins; also unscanned antibody chains)
Distance bin (V only): sequence distance to the nearest residue where SAbR and ANARCI IMGT numbering disagree
            'd0' disagreeing residue, 'd1_3', 'd4_10', 'd11p' (chain with >=1 disagreement), 'clean' (chain none).
Also per V chain: internal unbuilt residues (gaps in seq_index between first and last modelled V residue).
Denominator 'n' = residues carrying a rama or rota attribute (as in struct_unusual s05b).
Output: cache/verify_su_gap/tallies.pkl, vchain_info.pkl"""
import gzip, json, os, sys, numpy as np, pandas as pd, xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OLD = f'{SP}/deeper/cache/struct_unusual/vrpt'
NEW = f'{SP}/deeper/cache/verify_su_gap/vrpt'
OUT = f'{SP}/deeper/cache/verify_su_gap'

d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', usecols=['pdb', 'chain', 'method'], dtype={'chain': str})
d = d[d.method == 'EM']
keys = set(zip(d.pdb, d.chain))


def load_sabr():
    out = {}
    for fn in ['validation', 'snapshot_rest', 'new']:
        with open(f'{SP}/sabr/out/{fn}.jsonl') as f:
            for line in f:
                p = line[9:13]
                i = line.find('"chain": "')
                ch = line[i + 10:line.index('"', i + 10)]
                if (p, ch) in keys:
                    r = json.loads(line)
                    if 'residues' in r:
                        out[(p, ch)] = r['residues']
    return out


SAB = None


def vinfo(p):
    """{chain: {(auth, ic): distbin}} and per-chain info"""
    res, info = {}, []
    for (pp, ch), recs in SAB_BY_PDB.get(p, {}).items():
        rr = [r for r in recs if r[6] is not None]
        if not rr:
            continue
        dis = np.array([(r[4] is not None) and ((r[4], r[5] or '') != (r[6], r[7] or '')) for r in rr])
        idx = np.arange(len(rr))
        di = idx[dis]
        m = {}
        for k, r in enumerate(rr):
            if len(di) == 0:
                b = 'clean'
            else:
                dd = np.abs(di - k).min()
                b = 'd0' if dd == 0 else 'd1_3' if dd <= 3 else 'd4_10' if dd <= 10 else 'd11p'
            m[(int(r[0]), (r[1] or '').strip())] = b
        si = [r[3] for r in rr if r[3] is not None]
        unb = (max(si) - min(si) + 1 - len(set(si))) if si else 0
        # longest consecutive disagreement run
        run = best = 0
        for x in dis:
            run = run + 1 if x else 0; best = max(best, run)
        res[ch] = m
        info.append(dict(pdb=p, chain=ch, n_v=len(rr), n_dis=int(dis.sum()), max_run=best, unbuilt_int=int(unb)))
    return res, info


def work(p):
    fn = f'{OLD}/{p}_validation.xml.gz'
    if not os.path.exists(fn):
        fn = f'{NEW}/{p}_validation.xml.gz'
    if not os.path.exists(fn):
        return [], [], None
    vm, info = vinfo(p)
    T = {}
    hdr = None
    try:
        for ev, el in ET.iterparse(gzip.open(fn), events=('end',)):
            if el.tag == 'Entry':
                hdr = dict(pdb=p)
                for h in ['Q_score', 'clashscore', 'percent-rama-outliers', 'percent-rota-outliers',
                          'atom_inclusion_backbone', 'EMDB-resolution']:
                    try:
                        hdr['e_' + h.replace('-', '_')] = float(el.get(h))
                    except (TypeError, ValueError):
                        hdr['e_' + h.replace('-', '_')] = np.nan
            elif el.tag == 'ModelledSubgroup':
                if el.get('model', '1') == '1' and (el.get('rama') is not None or el.get('rota') is not None):
                    ch = el.get('chain')
                    key = (int(el.get('resnum')), (el.get('icode') or '').strip())
                    if ch in vm:
                        b = vm[ch].get(key)
                        cat, b = ('V', b) if b is not None else ('C', '-')
                    else:
                        cat, b = 'O', '-'
                    t = T.setdefault((ch, cat, b), [0] * 9)
                    rama = el.get('rama'); rota = el.get('rota')
                    t[0] += 1
                    t[1] += rama == 'Allowed'; t[2] += rama == 'OUTLIER'; t[3] += rama is not None
                    t[4] += any(c.tag == 'clash' for c in el)
                    t[5] += rota == 'OUTLIER'; t[6] += rota is not None
                    q = el.get('Q_score')
                    if q:
                        t[7] += float(q); t[8] += 1
                el.clear()
    except Exception as e:
        print('fail', p, e, file=sys.stderr)
        return [], info, hdr
    rows = [dict(pdb=p, chain=ch, cat=cat, dbin=b, n=t[0], allowed=t[1], rama_out=t[2], n_rama=t[3], clash=t[4],
                 rota_out=t[5], n_rota=t[6], q_sum=t[7], n_q=t[8]) for (ch, cat, b), t in T.items()]
    return rows, info, hdr


SAB_BY_PDB = {}
if __name__ == '__main__':
    SAB = load_sabr()
    for (p, ch), r in SAB.items():
        SAB_BY_PDB.setdefault(p, {})[(p, ch)] = r
    print('sabr chains', len(SAB), flush=True)
    pdbs = sorted({p for p, _ in keys})
    rows, infos, hdrs = [], [], []
    with ProcessPoolExecutor(2) as ex:
        for r, i, h in ex.map(work, pdbs, chunksize=8):
            rows += r; infos += i
            if h:
                hdrs.append(h)
    T = pd.DataFrame(rows); I = pd.DataFrame(infos); H = pd.DataFrame(hdrs)
    T.to_pickle(f'{OUT}/tallies.pkl'); I.to_pickle(f'{OUT}/vchain_info.pkl'); H.to_pickle(f'{OUT}/entry_hdr.pkl')
    print('entries parsed', T.pdb.nunique(), 'of', len(pdbs), 'rows', len(T))
