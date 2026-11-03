"""Select the case/control sample for the structural-unusualness lens.

Cases  : cryo-EM first-occurrence error units (one antibody chain type in one paper); one representative
         chain per unit (the err_first chain of the earliest-deposited entry, then first chain id).
Controls: cryo-EM units whose every chain is in an entry with no disagreement run of any kind, matched to the
         case on chain class (VH / VHH / K / L), deposition-year bin and resolution band, ~3 per case.
         Selection is greedy and prefers entries already selected or already cached, so the total number of
         structures stays near ~350.
Writes tables/sample.csv (one row per unit with role case/control, matched set id, representative chain).
"""
import os, numpy as np, pandas as pd
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
rng = np.random.default_rng(20261102)

d = pd.read_csv(f'{SP}/deeper/chains.csv.gz')
ent = pd.read_csv(f'{SP}/deeper/entries.csv')
em = d[d.method == 'EM'].copy()
em['cclass'] = np.where(em.vhh == 1, 'VHH', em.ctype)
em['ybin'] = pd.cut(em.year, [0, 2019, 2020, 2021, 2022, 2023, 2024, 2100],
                    labels=['<=2019', '2020', '2021', '2022', '2023', '2024', '>=2025']).astype(str)
em['rband'] = pd.cut(em.res, [0, 3.25, 3.5, 3.75, 4.0, 4.4, 9],
                     labels=['<3.25', '3.25-3.5', '3.5-3.75', '3.75-4.0', '4.0-4.4', '>4.4']).astype(str)

cached = {x[:4] for x in os.listdir(f'{SP}/sabr/dl') if x.endswith('.cif')}

# ---- cases
eu = em[em.err_first == 1].sort_values(['deposit', 'pdb', 'chain'])
cases = eu.groupby('unit', sort=False).head(1).copy()
cases['role_cc'] = 'case'
print('case units', len(cases), cases.cclass.value_counts().to_dict())

# ---- control pool: units entirely clean (all chains of the unit in clean entries, err_any False)
clean_entries = set(ent.loc[ent.err_any == 0, 'pdb'])
em['entry_clean'] = em.pdb.isin(clean_entries)
uclean = em.groupby('unit').apply(lambda g: bool(g.entry_clean.all() and (~g.err_any.astype(bool)).all()))
pool = em[em.unit.isin(uclean[uclean].index)].sort_values(['pdb', 'chain'])
pool = pool.groupby(['unit', 'pdb'], sort=False).head(1)  # one chain per unit per entry
print('control pool rows', len(pool), 'units', pool.unit.nunique(), 'entries', pool.pdb.nunique())

# entry-level selection: for each case entry choose control entries (same year bin and resolution band, or
# |dres|<=0.35 A) that contain clean units of every chain class the case entry needs; then assign up to 3 control
# units per case unit from the selected control entries (same class, year bin; nearest resolution).
pool_by_entry = pool.groupby('pdb')
pe = pool.groupby('pdb').agg(ybin=('ybin', 'first'), rband=('rband', 'first'), res=('res', 'first'),
                             classes=('cclass', lambda x: set(x)))
case_entries = cases.groupby('pdb').agg(ybin=('ybin', 'first'), rband=('rband', 'first'), res=('res', 'first'),
                                        classes=('cclass', lambda x: set(x)))
sel = []
TARGET = 190
order = list(rng.permutation(case_entries.index))
k = 0
while len(sel) < TARGET and k < 3:
    for ce in order:
        if len(sel) >= TARGET:
            break
        c = case_entries.loc[ce]
        cand = pe[(pe.ybin == c.ybin) & ((pe.rband == c.rband) | ((pe.res - c.res).abs() <= 0.35))
                  & np.array([c.classes <= x for x in pe.classes], bool) & ~pe.index.isin(sel)]
        if not len(cand):
            cand = pe[(pe.ybin == c.ybin) & ((pe.res - c.res).abs() <= 0.5)
                      & np.array([len(c.classes & x) > 0 for x in pe.classes], bool) & ~pe.index.isin(sel)]
        if not len(cand):
            continue
        cost = (cand.res - c.res).abs().values + np.where(cand.index.isin(cached), 0, 0.15) + rng.random(len(cand)) * 0.3
        sel.append(cand.index[int(np.argmin(cost))])
    k += 1
cpool = pool[pool.pdb.isin(sel)]
print('selected control entries', len(sel), 'control units available', len(cpool))
used_units, rows = set(), []
for k in range(3):
    for i in rng.permutation(len(cases)):
        c = cases.iloc[i]
        cand = cpool[(cpool.cclass == c.cclass) & (cpool.ybin == c.ybin) & ~cpool.unit.isin(used_units)]
        cand = cand[(cand.res - c.res).abs() <= 0.5]
        if not len(cand):
            continue
        cost = (cand.res - c.res).abs().values + rng.random(len(cand)) * 0.2
        pick = cand.iloc[int(np.argmin(cost))]
        used_units.add(pick.unit)
        r = pick.copy(); r['role_cc'] = 'control'; r['set_id'] = c.unit; rows.append(r)

cases['set_id'] = cases.unit
ctrl = pd.DataFrame(rows)
s = pd.concat([cases, ctrl], ignore_index=True)
keep = ['role_cc', 'set_id', 'unit', 'pdb', 'chain', 'ctype', 'cclass', 'vhh', 'year', 'ybin', 'res', 'rband',
        'deposit', 'paper', 'last_author', 'species', 'v_gene', 'j_gene', 'v_id', 'L1', 'L2', 'L3', 'L1_atyp',
        'L2_atyp', 'L3_atyp', 'cdr1_len', 'cdr3_len', 'role', 'sars2', 'irm', 'fit', 'lq', 'lq_z', 'err_modes', 'vseq']
s = s[keep]
s.to_csv(f'{OUT}/sample.csv', index=False)
print(s.role_cc.value_counts().to_dict(), 'entries', s.pdb.nunique(),
      'control entries not cached', len(set(s.pdb) - cached))
print(s.groupby('role_cc').cclass.value_counts().unstack())
print(ctrl.groupby('set_id').size().value_counts().sort_index())
