"""Direct test: is the stated starting model of an erroneous chain a *related* antibody carrying the same error?

Cases: every chain with a classified run (junction, junction_w118out, bulge, cdr12, gap) in the 3-5 A scan.
Controls: a random sample of clean chains (no run of any kind), EM, 2018-2026, with an experimental starting model.
For each chain, the stated starting models (pdbx_initial_refinement_model, experimental PDB entries) are opened, and the
template antibody domain of the same chain type closest in sequence is found. Relationship tiers: identical V region,
close (>= 90%), same V+J gene, same J gene, same V gene, other antibody, no antibody of that type.
The template domain is then run through the SAbR-vs-ANARCI detector (as in the scan). "Same error" = a run overlapping
the case's IMGT window +-3 with the same shift sign (for controls: any run of the same class window, see below).
Output: ../tables/lineage_templates.csv and printed summary.
"""
import sys, os, json, numpy as np, pandas as pd, warnings, logging
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
TP = '/home/user/SAbR/register_error_analysis/training_propagation/scripts'
sys.path.insert(0, TP)
from h19_jobs import dl, entry_domains
from h19_readout import anarci_labels
import h19_readout_full as RF
from Bio import Align
from concurrent.futures import ThreadPoolExecutor
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
O = '/home/user/SAbR/register_error_analysis/retest/lineage/tables'
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lineage_units.py')).read())
al = Align.PairwiseAligner(mode='global', match_score=1, mismatch_score=0, open_gap_score=-1, extend_gap_score=-0.5)
idn = lambda a, b: al.score(a, b) / max(len(a), len(b))
# starting models per entry
SM = {}
for line in open(f'{SP}/retest/modelling.jsonl'):
    e = json.loads(line); irm = e.get('pdbx_initial_refinement_model') or []
    ids = sorted({(x.get('accession_code') or '').lower() for x in irm if (x.get('source_name') or '').upper() == 'PDB' and x.get('accession_code')})
    if ids: SM[e['rcsb_id'].lower()] = ids
M['sm'] = M.pdb.map(SM)
cases = R[['pdb', 'chain', 'cls', 'imgt0', 'imgt1', 'delta']].merge(M[['pdb', 'chain', 'seq', 'V', 'J', 'ctype', 'sm', 'EM', 'year', 'lab']], on=['pdb', 'chain'])
flagged = set(zip(R.pdb, R.chain)) | set()
allruns = pd.read_csv(f'{SP}/unexpl/runs_origin_v2.csv'); anyflag = set(zip(allruns.pdb, allruns.chain))
ctrl_pool = M[M.sm.notna() & M.EM & (M.year >= 2018) & ~pd.Series(list(zip(M.pdb, M.chain)), index=M.index).isin(anyflag) & (M.flag == 0)]
ctrl = ctrl_pool.drop_duplicates(['pdb', 'seq']).sample(n=min(400, len(ctrl_pool)), random_state=0)
print('case runs', len(cases), 'with starting model', cases.sm.notna().sum(), '| controls', len(ctrl), flush=True)
tmpl_ids = sorted({t for l in pd.concat([cases.sm.dropna(), ctrl.sm]) for t in l})
print('template entries to open', len(tmpl_ids), flush=True)
with ThreadPoolExecutor(8) as ex: list(ex.map(dl, tmpl_ids))
TD = {}
for t in tmpl_ids:
    try: TD[t] = entry_domains(t)
    except Exception as e: TD[t] = dict(error=str(e))
def closest(seq, ctype, V, J, sms):
    best = None
    for t in sms or []:
        e = TD.get(t) or {}
        for d in e.get('doms', []):
            if (d['type'] in 'KL') != (ctype in 'KL') or (ctype in 'KL' and d['type'] != ctype): continue
            s = idn(seq, d['seq'])
            if best is None or s > best[0]: best = (s, t, d)
    if best is None: return None
    s, t, d = best
    from anarci import anarci
    n, a, _ = anarci([('q', d['seq'])], scheme='imgt', assign_germline=True)
    g = a[0][0]['germlines'] if a[0] else {}
    sp = g.get('v_gene', [[None, None]])[0][0] if g else None
    gv = lambda x: x.split('*')[0] if x else None
    tV = f"{sp}|{ctype}|{gv(g.get('v_gene', [[None, None]])[0][1])}"; tJ = f"{sp}|{ctype}|{gv(g.get('j_gene', [[None, None]])[0][1])}"
    tier = ('identical' if d['seq'] == seq else 'close (>=90%)' if s >= 0.9 else 'same V+J gene' if (tV == V and tJ == J)
            else 'same J gene' if tJ == J else 'same V gene' if tV == V else 'other antibody')
    return dict(template=t, tchain=d['chain'], tseq=d['seq'], identity=round(s, 3), tier=tier, tV=tV, tJ=tJ, tres=None)
def template_runs(t, chain, ct, vseq):
    lab, _ = anarci_labels(vseq); lab = lab + [(999, '')] * (len(vseq) - len(lab))
    return RF.dep_detector(t, chain, ct, vseq, lab, False)
rows = []
for kind, df in (('case', cases), ('control', ctrl)):
    for r in df.itertuples():
        base = dict(kind=kind, pdb=r.pdb, chain=r.chain, cls=getattr(r, 'cls', None), ctype=r.ctype, V=r.V, J=r.J, EM=r.EM, year=r.year,
                    imgt0=getattr(r, 'imgt0', None), imgt1=getattr(r, 'imgt1', None), delta=getattr(r, 'delta', None),
                    has_sm=isinstance(r.sm, list))
        if not isinstance(r.sm, list): rows.append(dict(base, tier='no experimental starting model recorded')); continue
        c = closest(r.seq, r.ctype, r.V, r.J, r.sm)
        if c is None: rows.append(dict(base, tier='starting model has no antibody of this type', templates=';'.join(r.sm))); continue
        try:
            truns = template_runs(c['template'], c['tchain'], r.ctype, c['tseq'])
        except Exception as e:
            truns = None; c['err'] = str(e)[:80]
        c['template_runs'] = None if truns is None else ';'.join(f'{a}-{b}:{d:+d}' for a, b, d in truns)
        if kind == 'case' and truns is not None and not pd.isna(base['imgt0']):
            same = [x for x in truns if x[0] <= base['imgt1'] + 3 and x[1] >= base['imgt0'] - 3 and (pd.isna(base['delta']) or np.sign(x[2]) == np.sign(base['delta']))]
            c['template_same_error'] = bool(same)
        c['template_any_run'] = None if truns is None else bool(truns)
        rows.append(dict(base, **c))
T = pd.DataFrame(rows); T.drop(columns=['tseq'], errors='ignore').to_csv(f'{O}/lineage_templates.csv', index=False)
pd.set_option('display.width', 250)
cs = T[T.kind == 'case']
print('\n== cases (runs) by class: closest antibody in the stated starting model')
print(pd.crosstab(cs.cls, cs.tier, margins=True).to_string())
print('\n== cases whose related (non-identical) template carries the same error:')
rel = cs[cs.tier.isin(['close (>=90%)', 'same V+J gene', 'same J gene', 'same V gene', 'other antibody'])]
print(rel.groupby(['cls', 'tier']).template_same_error.agg(['size', 'sum']).to_string())
print('\n== template has any run: cases vs controls, by tier')
print(T[T.template_any_run.notna()].groupby(['kind', 'tier']).template_any_run.agg(['size', 'sum', 'mean']).round(3).to_string())
