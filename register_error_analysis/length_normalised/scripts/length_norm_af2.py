"""Does the AlphaFold2 training-set blind spot explain the 2020-22 antibody excess (per residue)?

conkit-validate compares models with AlphaFold2 predictions, and AlphaFold2 was trained on PDB
structures released before 30 April 2018. Structures of proteins it had seen are flagged less
(antibody evidence in ../../retest). If 2020-22 non-antibody structures are more often
re-determinations of training-set proteins than 2020-22 antibodies are, the antibody excess could
be an artifact. Here each chain is labelled "seen" if its protein was in the PDB before the cutoff:
  * non-antibody chains: any SIFTS UniProt accession of the chain appears in an entry deposited
    before 2018-04-30 (deposition date from entries.idx, a proxy for release), or the exact chain
    sequence appears in such an entry;
  * antibody V domains: the first 100 residues of the chain (the V region) appear exactly in a
    sequence deposited before the cutoff.
Then the per-residue hazard model (cloglog, offset log length) is refitted within "unseen" and
"seen" proteins, for 2020-22 and for cryo-EM 2020-22.
"""
import gzip, pandas as pd, numpy as np, warnings
import statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
OUT = '/home/user/SAbR/register_error_analysis/length_normalised/tables'
PW = '/tmp/supp/pdbwide'
CUT = pd.Timestamp('2018-04-30')

# --- deposition dates
dep = {}
with open(f'{PW}/entries.idx', encoding='latin-1') as f:
    for line in f:
        p = line.split('\t')
        if len(p) < 3 or len(p[0]) != 4: continue
        try: dep[p[0].lower()] = pd.to_datetime(p[2], format='%m/%d/%y')
        except Exception: pass
old = {k for k, v in dep.items() if v < CUT}
print('entries deposited before cutoff:', len(old))
# --- sequences deposited before cutoff
old_seqs = set(); cur = None
with gzip.open(f'{PW}/pdb_seqres.txt.gz', 'rt') as f:
    for line in f:
        if line.startswith('>'):
            cur = line[1:5].lower() if 'mol:protein' in line else None
        elif cur in old:
            old_seqs.add(line.strip())
print('unique protein sequences before cutoff:', len(old_seqs))
big = '|'.join(old_seqs)
# --- UniProt accessions seen before cutoff
sifts = pd.read_csv(f'{PW}/pdb_chain_uniprot.tsv.gz', sep='\t', comment='#', usecols=['PDB', 'CHAIN', 'SP_PRIMARY'], dtype=str)
sifts['PDB'] = sifts.PDB.str.lower()
acc_old = set(sifts[sifts.PDB.isin(old)].SP_PRIMARY)
chain_acc = sifts.groupby(['PDB', 'CHAIN']).SP_PRIMARY.agg(set).to_dict()

d = pd.read_pickle('/tmp/analysis/ecod_grouped.pkl'); d = d[d.dlen > 0].copy()
IGL = {'Antibody V', 'Antibody C', 'TCR V', 'TCR C/other', 'Other IgSF (V/C1/C2/I-set; MHC, β2m, receptors)',
       'Fibronectin type III', 'Other Ig/E-set/PapD T-group', 'Other Ig-like X-group'}
d['cat'] = np.select([d.group == 'Antibody V', d.group == 'Antibody C', d.group.isin(IGL)],
                     ['Antibody V', 'Antibody C', 'Other Ig-like (non-antibody)'], 'Non-Ig')
d['era'] = pd.cut(d.year, [0, 2017, 2019, 2023], labels=['≤2017', '2018–19', '2020–22']).astype(str)
d['rb'] = d.res_bin.astype(str)
d = d[d.era == '2020–22'].copy()   # structures deposited after the cutoff only

chains = d[['pdb', 'chain', 'seq', 'cat']].drop_duplicates(['pdb', 'chain'])
seen = {}
for r in chains.itertuples():
    if r.cat == 'Antibody V':
        seen[(r.pdb, r.chain)] = r.seq[:100] in big
    else:
        accs = chain_acc.get((r.pdb, r.chain), set())
        seen[(r.pdb, r.chain)] = bool(accs & acc_old) or (r.seq in old_seqs)
# chains with both V and other domains (antibody chains) take the V-domain rule for all their domains
abv_keys = set(zip(d.pdb[d.cat == 'Antibody V'], d.chain[d.cat == 'Antibody V']))
for k in abv_keys:
    s = chains[(chains.pdb == k[0]) & (chains.chain == k[1])].seq.iloc[0]
    seen[k] = s[:100] in big
d['seen'] = [seen[k] for k in zip(d.pdb, d.chain)]

log = open(f'{OUT}/length_norm_af2.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n')
P('Structures deposited 2020-22 (after the AlphaFold2 training cutoff), conkit flags.')
P('"seen" = protein (UniProt or exact sequence; for antibodies, exact V region) in the PDB before 2018-04-30\n')
t = d[d.cat.isin(['Antibody V', 'Non-Ig', 'Other Ig-like (non-antibody)'])].groupby(['method', 'cat', 'seen']).apply(
    lambda x: pd.Series(dict(domains=len(x), pct_flagged=100 * x.flag.mean(), per10k=1e4 * x.n_err.sum() / x.dlen.sum(), medL=x.dlen.median())))
P(t.round(2).to_string()); t.round(3).to_csv(f'{OUT}/af2_seen_rates_2020_22.csv')

def hr(sub, rhs, label):
    sub = sub.copy(); sub['g'] = pd.factorize(sub.pdb)[0]
    m = smf.glm(f"flag ~ C(cat, Treatment('Non-Ig')) + {rhs}", data=sub, offset=np.log(sub.dlen),
                family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit(cov_type='cluster', cov_kwds={'groups': sub.g})
    k = "C(cat, Treatment('Non-Ig'))[T.Antibody V]"; ci = m.conf_int()
    return dict(model=label, n_abV=int((sub.cat == 'Antibody V').sum()), n_nonIg=int((sub.cat == 'Non-Ig').sum()),
                HR_abV_vs_nonIg=np.exp(m.params[k]), lo=np.exp(ci.loc[k, 0]), hi=np.exp(ci.loc[k, 1]), p=m.pvalues[k])
rows = []
base = d[d.cat.isin(['Antibody V', 'Non-Ig'])]
for meth in ['all', 'EM']:
    b = base if meth == 'all' else base[base.method == 'EM']
    rhs = 'C(method)*C(rb)' if meth == 'all' else 'C(rb)'
    rows.append(hr(b, rhs, f'2020–22 {meth}: all proteins'))
    rows.append(hr(b, rhs + ' + seen', f'2020–22 {meth}: adjusted for seen'))
    rows.append(hr(b[~b.seen], rhs, f'2020–22 {meth}: unseen proteins only'))
    rows.append(hr(b[b.seen], rhs, f'2020–22 {meth}: seen proteins only'))
    bl = b[(b.dlen >= 100) & (b.dlen <= 130)]
    rows.append(hr(bl[~bl.seen], rhs, f'2020–22 {meth}: unseen, length 100–130'))
t2 = pd.DataFrame(rows)
P('\nPer-residue flag hazard ratio, antibody V vs non-Ig (cloglog, offset log length, entry-clustered):')
P(t2.round(3).to_string(index=False)); t2.round(4).to_csv(f'{OUT}/af2_seen_hazard_ratios.csv', index=False)
log.close()
