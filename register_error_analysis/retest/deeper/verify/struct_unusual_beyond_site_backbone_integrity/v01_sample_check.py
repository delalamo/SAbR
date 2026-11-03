"""Check the investigator's case/control sample against the shared chains/entries tables."""
import pandas as pd, numpy as np
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
TAB = '/home/user/SAbR/register_error_analysis/retest/deeper/struct_unusual/tables'
d = pd.read_csv(f'{SP}/deeper/chains.csv.gz', dtype={'chain': str})
ent = pd.read_csv(f'{SP}/deeper/entries.csv')
s = pd.read_csv(f'{TAB}/sample.csv', dtype={'chain': str})
U = pd.read_csv(f'{TAB}/unit_metrics.csv', dtype={'chain': str})
k = s.merge(d[['pdb', 'chain', 'method', 'err_first', 'err_any', 'err_modes', 'unit', 'paper', 'last_author']],
            on=['pdb', 'chain'], how='left', suffixes=('', '_d'))
print('sample rows', len(s), 'joined method', k.method.value_counts().to_dict())
print('cases err_first', k[k.role_cc == 'case'].err_first.value_counts().to_dict())
print('controls err_any', k[k.role_cc == 'control'].err_any.value_counts().to_dict())
ce = ent.set_index('pdb')
print('control entries err_any', ce.loc[k[k.role_cc == 'control'].pdb.unique(), 'err_any'].value_counts().to_dict())
print('case entries err_first', ce.loc[k[k.role_cc == 'case'].pdb.unique(), 'err_first'].value_counts().to_dict())
# all EM first-occurrence units
em = d[d.method == 'EM']
print('EM err_first units', em[em.err_first == 1].unit.nunique(), 'sample cases', (s.role_cc == 'case').sum())
# overlap of entries between cases and controls
print('entries in both roles', len(set(s[s.role_cc == 'case'].pdb) & set(s[s.role_cc == 'control'].pdb)))
# papers in both roles
print('papers in both roles', len(set(s[s.role_cc == 'case'].paper) & set(s[s.role_cc == 'control'].paper)))
# case units dropped from the beyond-site test
m = U[U.role_cc == 'case']
print('case units with metrics', len(m), 'with rama_allowed_out', m.rama_allowed_out.notna().sum())
miss = s[(s.role_cc == 'case') & ~s.unit.isin(m.dropna(subset=['rama_allowed_out']).unit)]
print('cases missing rama_allowed_out:\n', miss[['pdb', 'chain', 'cclass', 'err_modes']].to_string())
mm = m[m.rama_allowed_out.isna()]
print(mm[['pdb', 'chain', 'n_V', 'n_site', 'n_out']].to_string() if len(mm) else 'none in U')
# chains per paper / lab among cases
c = s[s.role_cc == 'case']
print('case units per paper', c.paper.value_counts().value_counts().sort_index().to_dict())
print('case units per last author top', c.last_author.value_counts().head(8).to_dict())
print('ctrl units per last author top', s[s.role_cc == 'control'].last_author.value_counts().head(8).to_dict())
print('case entries', c.pdb.nunique(), 'ctrl entries', s[s.role_cc=='control'].pdb.nunique())
