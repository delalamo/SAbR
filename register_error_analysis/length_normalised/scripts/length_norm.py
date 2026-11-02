"""Length-normalised register-error rates: antibody V domains vs other domains (2024 conkit flags).

Inputs (built by ../../scripts/build_ecod.py, ../../scripts/analyze.py and
../../followup/scripts/filter_released.py): /tmp/analysis/ecod_grouped.pkl (one row per ECOD domain,
released by 2022-04-05, with domain length `dlen`, error count `n_err`, residues in errors `err_res`,
and `flag`), /tmp/analysis/chains.pkl.

Five ways of putting antibody V domains and other proteins on the same footing:
  1. units: entry, chain, domain, residue;
  2. per-residue rates: error events per 10,000 residues, % residues inside errors;
  3. length-matched: only domains of 100-130 residues (the antibody V range), and direct
     standardisation of every comparator to the antibody V length distribution;
  4. per-residue hazard ratio: complementary log-log binomial model with offset log(length),
     P(flag) = 1 - exp(-h * L), adjusted for method x resolution and era, entry-clustered SEs;
     plus a Poisson model of error counts with the same offset;
  5. the same, by era and for cryo-EM alone, and within entries that contain antibodies.
"""
import pandas as pd, numpy as np, warnings
import statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
OUT = '/home/user/SAbR/register_error_analysis/length_normalised/tables'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)

d = pd.read_pickle('/tmp/analysis/ecod_grouped.pkl')
d = d[d.dlen > 0].copy()  # a few ECOD ranges with negative author numbering give dlen <= 0
ch = pd.read_pickle('/tmp/analysis/chains.pkl')
IGL = {'Antibody V', 'Antibody C', 'TCR V', 'TCR C/other', 'Other IgSF (V/C1/C2/I-set; MHC, β2m, receptors)',
       'Fibronectin type III', 'Other Ig/E-set/PapD T-group', 'Other Ig-like X-group'}
d['cat'] = np.select([d.group == 'Antibody V', d.group == 'Antibody C', d.group.isin(IGL)],
                     ['Antibody V', 'Antibody C', 'Other Ig-like (non-antibody)'], 'Non-Ig')
d['era'] = pd.cut(d.year, [0, 2017, 2019, 2023], labels=['≤2017', '2018–19', '2020–22']).astype(str)
d['rb'] = d.res_bin.astype(str)
d['logL'] = np.log(d.dlen.clip(lower=20))
d['g'] = pd.factorize(d.pdb)[0]
# structural comparator inside Non-Ig: non-Ig β-sandwiches
d['cat2'] = np.where((d.cat == 'Non-Ig') & (d.group == 'Other β-sandwiches'), 'Non-Ig β-sandwich', d.cat)
LO, HI = 100, 130  # antibody V length window (5th-95th percentile ~ 105-128)

rng = np.random.default_rng(0)
def boot(sub, f, B=300):
    """bootstrap over entries for a statistic f(sub)"""
    ids = sub.pdb.unique(); grp = {k: v for k, v in sub.groupby('pdb').groups.items()}
    vals = []
    for _ in range(B):
        pick = rng.choice(ids, len(ids)); idx = np.concatenate([grp[k] for k in pick])
        vals.append(f(sub.loc[idx]))
    return np.percentile(vals, [2.5, 97.5])

def rates(sub):
    return pd.Series(dict(domains=len(sub), entries=sub.pdb.nunique(), median_len=sub.dlen.median(),
        pct_flagged=100 * sub.flag.mean(),
        errors_per_10k_res=1e4 * sub.n_err.sum() / sub.dlen.sum(),
        pct_res_in_errors=100 * sub.err_res.sum() / sub.dlen.sum()))

log = open(f'{OUT}/length_norm.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n')

# ---------------------------------------------------------------- 1. units
P('=' * 100); P('1. The same data in different units (conkit flags, released ≤ 2022-04-05)'); P('=' * 100)
ent = ch.groupby('pdb').agg(flag=('flag', 'max'), L=('length', 'sum'), ab=('ab', lambda x: (x == 'antibody').any()))
abV_ent = d[d.cat == 'Antibody V'].groupby('pdb').flag.max()
P(f'entries: {100*ent.flag.mean():.1f}% have ≥1 flagged chain (n={len(ent):,}; median {ent.L.median():.0f} residues per entry)')
P(f'entries containing antibodies: {100*ent[ent.ab].flag.mean():.1f}% have ≥1 flagged chain anywhere (n={ent.ab.sum():,}); '
  f'{100*abV_ent.mean():.1f}% have ≥1 flagged antibody V domain (n={len(abV_ent):,})')
P(f'chains: all {100*ch.flag.mean():.1f}% (median {ch.length.median():.0f} res); antibody chains {100*ch[ch.ab=="antibody"].flag.mean():.1f}% '
  f'(median {ch[ch.ab=="antibody"].length.median():.0f}); other chains {100*ch[ch.ab=="other"].flag.mean():.1f}% (median {ch[ch.ab=="other"].length.median():.0f})')
t1 = d.groupby('cat2').apply(rates)
for c in t1.index:
    sub = d[d.cat2 == c]
    lo, hi = boot(sub, lambda s: 1e4 * s.n_err.sum() / s.dlen.sum(), B=200)
    t1.loc[c, 'per10k_lo'] = lo; t1.loc[c, 'per10k_hi'] = hi
P(t1.round(2).to_string()); t1.round(3).to_csv(f'{OUT}/rates_by_class.csv')

# ---------------------------------------------------------------- 2. length dependence
P('\n' + '=' * 100); P('2. Flag rate vs domain length'); P('=' * 100)
bins = [0, 60, 80, 100, 115, 130, 160, 200, 250, 300, 400, 7000]
d['lbin'] = pd.cut(d.dlen, bins)
t2 = d.pivot_table(index='lbin', columns='cat', values='flag', aggfunc=['mean', 'size'], observed=True)
t2m = (100 * t2['mean']).round(2); t2n = t2['size']
P(pd.concat([t2m.add_suffix(' %'), t2n.add_suffix(' n')], axis=1).to_string())
t2out = d.groupby(['cat', 'lbin'], observed=True).agg(n=('flag', 'size'), pct=('flag', lambda x: 100 * x.mean()),
                                                     per10k=('n_err', 'sum'), res=('dlen', 'sum')).reset_index()
t2out['per10k'] = 1e4 * t2out.per10k / t2out.res; t2out.drop(columns='res').to_csv(f'{OUT}/rates_by_length.csv', index=False)
# empirical length exponent in non-Ig domains: cloglog with logL as free covariate
ni = d[d.cat == 'Non-Ig']
m = smf.glm('flag ~ logL + C(method)*C(rb) + C(era)', data=ni, family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit(
    cov_type='cluster', cov_kwds={'groups': ni.g})
P(f'non-Ig domains: d log(hazard)/d log(length) = {m.params["logL"]:.2f} ({m.conf_int().loc["logL",0]:.2f}–{m.conf_int().loc["logL",1]:.2f}); 1 = flag risk proportional to length')

# ---------------------------------------------------------------- 3. length matched / standardised
P('\n' + '=' * 100); P(f'3. Length-matched ({LO}–{HI} residues) and directly standardised to the antibody V length distribution'); P('=' * 100)
lm = d[(d.dlen >= LO) & (d.dlen <= HI)]
t3 = lm.groupby('cat2').apply(rates)
for c in t3.index:
    lo, hi = boot(lm[lm.cat2 == c], lambda s: 100 * s.flag.mean(), B=300)
    t3.loc[c, 'pct_lo'] = lo; t3.loc[c, 'pct_hi'] = hi
P(t3.round(2).to_string()); t3.round(3).to_csv(f'{OUT}/length_matched_100_130.csv')
# direct standardisation: weight each comparator's 5-residue length bins by the antibody V distribution
sb = np.arange(80, 205, 5)
d['sbin'] = pd.cut(d.dlen, sb)
w = d[d.cat == 'Antibody V'].sbin.value_counts(normalize=True)
rows = []
for c in ['Antibody V', 'Antibody C', 'Other Ig-like (non-antibody)', 'Non-Ig', 'Non-Ig β-sandwich']:
    for era in ['all', '≤2017', '2018–19', '2020–22']:
        for meth in ['all', 'EM', 'X-ray']:
            sub = d[d.cat2 == c] if c == 'Non-Ig β-sandwich' else d[d.cat == c]
            if era != 'all': sub = sub[sub.era == era]
            if meth != 'all': sub = sub[sub.method == meth]
            r = sub.groupby('sbin', observed=False).flag.mean().reindex(w.index)
            ok = r.notna() & (w > 0)
            rows.append(dict(cls=c, era=era, method=meth, n=len(sub), crude=100 * sub.flag.mean(),
                             standardised=100 * (r[ok] * w[ok]).sum() / w[ok].sum(), weight_covered=w[ok].sum()))
t3b = pd.DataFrame(rows)
P(t3b[t3b.method == 'all'].pivot(index='cls', columns='era', values='standardised').round(2).to_string())
P('cryo-EM only:'); P(t3b[t3b.method == 'EM'].pivot(index='cls', columns='era', values='standardised').round(2).to_string())
P('X-ray only:'); P(t3b[t3b.method == 'X-ray'].pivot(index='cls', columns='era', values='standardised').round(2).to_string())
t3b.round(3).to_csv(f'{OUT}/standardised_to_abV_lengths.csv', index=False)

# ---------------------------------------------------------------- 4. per-residue hazard ratios
P('\n' + '=' * 100); P('4. Per-residue hazard ratio vs antibody V domains (cloglog, offset log length) and error-rate ratio (Poisson)'); P('=' * 100)
def hr(sub, formula_rhs, label, ref='Antibody V', col='cat'):
    sub = sub.dropna(subset=['method', 'rb', 'era', 'dlen', 'flag', 'n_err', col]).copy(); sub = sub[sub.dlen > 0]; sub['g'] = pd.factorize(sub.pdb)[0]
    f = f"flag ~ C({col}, Treatment('{ref}')) + {formula_rhs}"
    mc = smf.glm(f, data=sub, offset=np.log(sub.dlen), family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit(
        cov_type='cluster', cov_kwds={'groups': sub.g})
    fp = f"n_err ~ C({col}, Treatment('{ref}')) + {formula_rhs}"
    mp = smf.glm(fp, data=sub, offset=np.log(sub.dlen), family=sm.families.Poisson()).fit(cov_type='cluster', cov_kwds={'groups': sub.g})
    out = []
    for mod, kind in [(mc, 'flag hazard per residue'), (mp, 'error events per residue')]:
        ci = mod.conf_int()
        for k in mod.params.index:
            if k.startswith(f'C({col}'):
                name = k.split('[T.')[1].rstrip(']')
                out.append(dict(model=label, measure=kind, comparator=name, ratio_vs_abV=np.exp(mod.params[k]),
                                lo=np.exp(ci.loc[k, 0]), hi=np.exp(ci.loc[k, 1]), p=mod.pvalues[k]))
    return out
rows = []
adj = "C(method)*C(rb) + C(era)"
rows += hr(d, adj, 'all domains, adjusted')
rows += hr(d, adj, 'all domains, adjusted (β-sandwich split)', col='cat2')
rows += hr(d[(d.dlen >= LO) & (d.dlen <= HI)], adj, f'length {LO}–{HI} only, adjusted')
for era in ['≤2017', '2018–19', '2020–22']:
    rows += hr(d[d.era == era], "C(method)*C(rb)", f'era {era}, adjusted')
for era in ['2018–19', '2020–22']:
    rows += hr(d[(d.era == era) & (d.method == 'EM')], "C(rb)", f'cryo-EM {era}, adjusted')
t4 = pd.DataFrame(rows)
P(t4[t4.measure == 'flag hazard per residue'].round(3).to_string(index=False))
P('\nPoisson (error events per residue):')
P(t4[t4.measure == 'error events per residue'].round(3).to_string(index=False))
t4.round(4).to_csv(f'{OUT}/hazard_ratios_vs_abV.csv', index=False)

# ---------------------------------------------------------------- 5. within entries containing antibodies
P('\n' + '=' * 100); P('5. Within entries that contain antibody V domains: per-residue rates, same map and resolution'); P('=' * 100)
abE = set(d.pdb[d.cat == 'Antibody V'])
wd = d[d.pdb.isin(abE) & d.cat.isin(['Antibody V', 'Non-Ig'])].copy()
for meth in ['EM', 'X-ray']:
    for era in ['all', '2020–22']:
        s = wd[wd.method == meth] if era == 'all' else wd[(wd.method == meth) & (wd.era == era)]
        r = s.groupby('cat').apply(lambda x: pd.Series(dict(domains=len(x), per10k=1e4 * x.n_err.sum() / x.dlen.sum(), pct=100 * x.flag.mean(), medL=x.dlen.median())))
        # Mantel-Haenszel incidence-rate ratio stratified by entry (events per residue), non-Ig vs antibody V
        num = den = 0
        for _, e in s.groupby('pdb'):
            a = e[e.cat == 'Non-Ig']; b = e[e.cat == 'Antibody V']
            if len(a) == 0 or len(b) == 0: continue
            T = e.dlen.sum(); num += a.n_err.sum() * b.dlen.sum() / T; den += b.n_err.sum() * a.dlen.sum() / T
        P(f'{meth}, {era}:'); P(r.round(2).to_string())
        P(f'   Mantel-Haenszel rate ratio (non-Ig / antibody V, stratified by entry) = {num/den if den else float("nan"):.2f}')

# ---------------------------------------------------------------- 6. whole chains
P('\n' + '=' * 100); P('6. Whole chains: antibody chains vs other chains of similar length (200–240 residues)'); P('=' * 100)
c2 = ch[(ch.length >= 200) & (ch.length <= 240)]
P(c2.groupby('ab').agg(n=('flag', 'size'), pct=('flag', lambda x: 100 * x.mean()), per10k=('n_err', 'sum'), L=('length', 'sum')).assign(
    per10k=lambda x: 1e4 * x.per10k / x.L).drop(columns='L').round(2).to_string())
log.close()
