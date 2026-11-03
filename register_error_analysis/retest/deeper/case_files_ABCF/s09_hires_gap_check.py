"""Does SAbR disagree with ANARCI for the suspect indel patterns even in high-resolution (<2.6 A) X-ray structures?
If yes, runs carrying these patterns are numbering-convention artifacts, not register errors.

Patterns (heavy chain, ANARCI IMGT positions absent):  FR3 gap at 84 ({73,84}), at 96 ({73,96}), at 80+84 ({73,80,84});
FR1 gap at 26; and the common FR3 gap at 81 ({73,81}) as a negative control (never disagrees in the scan).
For each pattern, up to 4 high-resolution entries (s08 candidate list) are downloaded and renumbered with SAbR; the
number of residues in IMGT 20-40 / 74-100 where SAbR and ANARCI disagree is counted.
Output: tables/hires_gap_check.csv
"""
import json, os, gzip, urllib.request, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from anarci import run_anarci
from sabr import renumber_structure
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
CA = f'{SP}/deeper/cache/case_files_ABCF'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/case_files_ABCF'
os.makedirs(f'{CA}/hires_cif', exist_ok=True)
cand = json.load(open(f'{CA}/gap_pattern_pdbs.json'))
A3 = dict(ALA='A', ARG='R', ASN='N', ASP='D', CYS='C', GLN='Q', GLU='E', GLY='G', HIS='H', ILE='I', LEU='L', LYS='K', MET='M', PHE='F',
          PRO='P', SER='S', THR='T', TRP='W', TYR='Y', VAL='V')
rows = []
for pat, L in cand.items():
    seen = set(); picks = []
    for r, p, c, e, s in L:
        if r < 2.6 and p not in seen: picks.append((r, p, c)); seen.add(p)
        if len(picks) == 4: break
    for r, p, c in picks:
        f = f'{CA}/hires_cif/{p}.cif'
        if not os.path.exists(f):
            data = urllib.request.urlopen(f'https://files.rcsb.org/download/{p.upper()}.cif', timeout=120).read()
            open(f, 'wb').write(data)
        try:
            st = MMCIFParser(QUIET=True).get_structure(p, f)
            md = MMCIF2Dict(f)
            model = next(iter(st)); ch = model[c]
            res = [x for x in ch if x.id[0] == ' ' and x.get_resname() in A3][:128]
            # trim to the V domain
            tst = Structure(p); tst.header = dict(getattr(st, 'header', {}) or {}); tm = Model(0); tst.add(tm); tc = ch.copy()
            keep = {x.id for x in res}
            for x in list(tc):
                if x.id not in keep: tc.detach_child(x.id)
            tm.add(tc)
            new = renumber_structure(tst, chain=c, chain_type='H')
            sab = [(x.id[1], x.id[2].strip()) for x in next(iter(new))[c]]
            seq = ''.join(A3[x.get_resname()] for x in res)
            an = run_anarci([('q', seq)], scheme='imgt')
            num = [(pp, ii.strip()) for (pp, ii), aa in an[1][0][0][0] if aa != '-']
            st0 = an[1][0][0][1]
            pairs = list(zip(num, sab[st0:st0 + len(num)]))
            dis_fr3 = sum(1 for a, b in pairs if 74 <= a[0] <= 100 and a != b)
            dis_fr1 = sum(1 for a, b in pairs if 20 <= a[0] <= 40 and a != b)
            detail = ' '.join(f'{seq[st0+i]}{a[0]}{a[1]}/{b[0]}{b[1]}' for i, (a, b) in enumerate(pairs) if (74 <= a[0] <= 100 or 20 <= a[0] <= 40) and a != b)
            rows.append(dict(pattern=pat, pdb=p, chain=c, res=r, dis_FR3_74_100=dis_fr3, dis_20_40=dis_fr1, detail=detail[:300]))
        except Exception as ex:
            rows.append(dict(pattern=pat, pdb=p, chain=c, res=r, note=f'{type(ex).__name__}: {str(ex)[:120]}'))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/tables/hires_gap_check.csv', index=False)
