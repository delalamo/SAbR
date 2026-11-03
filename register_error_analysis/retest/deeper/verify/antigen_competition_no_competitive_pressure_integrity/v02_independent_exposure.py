# Independent rebuild of the competition exposure without the investigator's GraphQL/interface pipeline:
#   antibody entries  = PDB entries with >=1 SEQRES protein chain matching a V-domain motif regex (own scan)
#   antigen           = all SIFTS UniProt accessions on non-V chains of the entry, minus Ig accessions and fiducials
#   lab, date         = last author and deposition date from entries.idx
#   exposure          = distinct other last authors depositing antibody entries sharing >=1 antigen accession within +-N days
import gzip, re, json, collections, warnings, unicodedata
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
INV = '/home/user/SAbR/register_error_analysis/retest/deeper/antigen_competition'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/verify/antigen_competition_no_competitive_pressure_integrity'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

VRX = re.compile(r'C.{8,20}W[FVIYLAM][RQKHE][QRKEH].{38,75}?[YFHL].C.{2,45}?[WF]G.G')
vchains = set(); vent = set(); nprot = 0
with gzip.open('/tmp/supp/pdbwide/pdb_seqres.txt.gz', 'rt') as fh:
    hdr = None
    for line in fh:
        if line[0] == '>':
            hdr = line[1:].split()
            continue
        if hdr[1] != 'mol:protein': continue
        nprot += 1
        s = line.strip()
        if len(s) >= 90 and VRX.search(s):
            vchains.add(hdr[0]); vent.add(hdr[0].split('_')[0])
print('protein chains', nprot, 'V-motif chains', len(vchains), 'entries', len(vent))
inv_ab = set(json.load(open(f'{SP}/deeper/cache/antigen_competition/ab_entries_pdbwide.json')))
print('concordance with investigator ANARCI entry list: both', len(vent & inv_ab), 'regex only', len(vent - inv_ab), 'ANARCI only', len(inv_ab - vent))
foc = pd.read_csv(f'{SP}/deeper/entries.csv')
print('focal entries found by regex', foc.pdb.isin(vent).mean().round(4))

# entries.idx: date + last author
def norm(n):
    n = unicodedata.normalize('NFKD', n).encode('ascii', 'ignore').decode().lower().strip()
    sur, _, ini = n.partition(',')
    return re.sub(r'[^a-z]', '', sur) + '_' + re.sub(r'[^a-z]', '', ini)[:1]
date = {}; lab = {}
for l in open('/tmp/supp/pdbwide/entries.idx', encoding='latin-1').read().splitlines()[2:]:
    q = l.split('\t')
    if len(q) < 6: continue
    p = q[0].lower()
    try: date[p] = pd.Timestamp(pd.to_datetime(q[2], format='%m/%d/%y'))
    except Exception: continue
    auth = [a for a in re.split(r'(?<=\.),\s*', q[5].strip()) if a.strip()]
    lab[p] = norm(auth[-1]) if auth else None
# SIFTS
up = pd.read_csv('/tmp/supp/pdbwide/pdb_chain_uniprot.tsv.gz', sep='\t', comment='#', usecols=['PDB', 'CHAIN', 'SP_PRIMARY']).drop_duplicates()
up['key'] = up.PDB + '_' + up.CHAIN
up['isv'] = up.key.isin(vchains)
# Ig accessions: mapped to V-motif chains in >=50% of their chain occurrences across the PDB
fr = up.groupby('SP_PRIMARY').isv.mean()
igacc = set(fr[fr >= 0.5].index)
uni = json.load(open(f'{SP}/deeper/cache/antigen_competition/uniprot_info.json'))
FIDRX = re.compile(r'guanine nucleotide-binding protein g\(|cytochrome b562|maltose[- ]|maltodextrin|endolysin|green fluorescent protein|immunoglobulin g-binding protein|^protein l$|^protein a$|immunoglobulin-binding protein|streptavidin|immunoglobulin|ig gamma|ig kappa|ig lambda|ig heavy', re.I)
fid = {a for a, u in uni.items() if u and FIDRX.search(u.get('name') or '')} | {'P0ABE7', 'P0AEX9', 'P00720', 'P42212', 'P63092', 'P63096', 'P62873', 'P59768', 'P04896', 'P54311', 'P63212', 'P63211', 'P0DOX5', 'P01857', 'P01834', 'P0DOY2', 'P0DOY3'}
upn = up[~up.isv & ~up.SP_PRIMARY.isin(igacc) & ~up.SP_PRIMARY.isin(fid) & up.PDB.isin(vent)]
ag = upn.groupby('PDB').SP_PRIMARY.agg(lambda s: frozenset(s)).to_dict()
print('antibody entries with >=1 antigen accession', len(ag), 'of', len(vent))
# index: accession -> list of (date, lab, pdb) for antibody entries
byacc = collections.defaultdict(list)
for p, accs in ag.items():
    if p not in date: continue
    for a in accs: byacc[a].append((date[p], lab.get(p), p))
def comp(p, days):
    if p not in ag or p not in date: return np.nan
    d = date[p]; L = lab.get(p); others = set()
    for a in ag[p]:
        for dq, lq, q in byacc[a]:
            if q != p and abs((dq - d).days) <= days and lq != L: others.add(lq)
    return len(others)
foc = foc.copy()
for days, nm in [(30, '1m'), (183, '6m'), (365, '12m')]:
    foc[f'v_labs_{nm}'] = [comp(p, days) for p in foc.pdb]
foc['v_lab'] = foc.pdb.map(lab); foc['v_date'] = foc.pdb.map(date)
foc[['pdb', 'v_lab', 'v_date', 'v_labs_1m', 'v_labs_6m', 'v_labs_12m']].to_csv(f'{OUT}/v02_independent_exposure.csv', index=False)

# compare with investigator
F = pd.read_csv(f'{INV}/tables/focal_entries_antigen_features.csv.gz')
M = F.merge(foc[['pdb', 'v_lab', 'v_date', 'v_labs_1m', 'v_labs_6m', 'v_labs_12m']], on='pdb', how='left')
M['v_date'] = pd.to_datetime(M.v_date); M['dep'] = pd.to_datetime(M.deposit)
print('deposit date agreement entries.idx vs entries.csv:', (M.v_date == M.dep).mean().round(4))
from scipy.stats import spearmanr
ok = M.labs_acc_12m.notna() & M.v_labs_12m.notna()
print('Spearman independent vs investigator labs_acc_12m:', round(spearmanr(M.labs_acc_12m[ok], M.v_labs_12m[ok]).correlation, 3), 'n', ok.sum(),
      '| exact agreement', round((M.labs_acc_12m[ok] == M.v_labs_12m[ok]).mean(), 3), '| zero vs nonzero agreement', round(((M.labs_acc_12m[ok] > 0) == (M.v_labs_12m[ok] > 0)).mean(), 3))
M['sars2'] = M.sars2.astype(bool)
for c in ['v_labs_1m', 'v_labs_6m', 'v_labs_12m']: M['l_' + c] = np.log2(1 + M[c])
M['v_any12'] = (M.v_labs_12m > 0).astype(float).where(M.v_labs_12m.notna())
M['v_race'] = (M.v_labs_1m > 0).astype(float).where(M.v_labs_1m.notna())
def fit(D, term, extra='C(ycat) + res'):
    D = D.dropna(subset=[term])
    mod = smf.glm(f'y ~ {term} + {extra}', D, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.paper)[0]})
    k = [x for x in mod.params.index if x.startswith(term)][0]; ci = mod.conf_int().loc[k]
    return dict(term=term, OR=np.exp(mod.params[k]), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=mod.pvalues[k], n=len(D), n_err=int(D.y.sum()))
R = []
EM = M[M.method == 'EM']; XR = M[M.method == 'X-ray']
for lbl, D, extra in [('EM all (incl. unbound/other)', EM, 'C(ycat) + res'), ('EM non-SARS2', EM[~EM.sars2], 'C(ycat) + res'), ('EM SARS2', EM[EM.sars2], 'C(ycat) + res'),
                      ('EM 2020-21', EM[EM.period == '2020-21'], 'C(ycat) + res'), ('EM 2022+', EM[EM.period == '2022+'], 'C(ycat) + res'), ('X-ray', XR, 'C(period) + res')]:
    for t in ['l_v_labs_12m', 'l_v_labs_6m', 'v_race', 'v_any12']:
        try: r = fit(D, t, extra); r['stratum'] = lbl; R.append(r)
        except Exception as ex: print('fail', lbl, t, ex)
Rt = pd.DataFrame(R)[['stratum', 'term', 'OR', 'lo', 'hi', 'p', 'n', 'n_err']]
print(Rt.round(4).to_string())
Rt.to_csv(f'{OUT}/v02_independent_exposure_tests.csv', index=False)
D = EM.dropna(subset=['v_labs_12m']).copy(); D['bin'] = pd.cut(D.v_labs_12m, [-1, 0, 2, 9, 49, 1e5], labels=['0', '1-2', '3-9', '10-49', '>=50'])
print(D.groupby('bin').y.agg(['mean', 'size', 'sum']).round(3).to_string())
print(D.groupby(['sars2', 'bin']).y.agg(['mean', 'size', 'sum']).round(3).to_string())
