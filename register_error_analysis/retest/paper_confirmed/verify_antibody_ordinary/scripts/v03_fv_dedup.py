"""ABangle contrasts with NCS / repeated-model copies collapsed: one value per distinct (paper, VH seq, VL seq).
Controls are one Fv per entry already; cases include every Fv copy in an entry (e.g. three Fabs on a spike trimer)."""
import numpy as np, pandas as pd
from scipy import stats
AO = '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/antibody_ordinary/tables'
DEEP = '/home/user/SAbR/register_error_analysis/retest/deeper'
F = pd.read_csv(f'{AO}/struct_fv_frame.csv'); F = F[F.kind == 'Fv']
dom = '10.1038/s41422-021-00487-9'
print('case Fv copies per entry (strict):', F[(F.role == 'case') & F.strict].groupby('pdb').size().value_counts().sort_index().to_dict())
print('control Fvs per entry:', F[F.role != 'case'].groupby('pdb').size().value_counts().to_dict())
for meth in ['EM', 'X-ray']:
    ca = F[(F.role == 'case') & F.strict & ((F.method == 'EM') == (meth == 'EM'))]
    co = F[(F.role != 'case') & ((F.method == 'EM') == (meth == 'EM'))]
    for lab, sub in [('all', ca), ('no dominant', ca[ca.paper != dom])]:
        u = sub.groupby(['paper', 'Hseq', 'Lseq']).agg(maha=('maha_dep', 'mean'), out=('outlier99', 'max'),
                                                      core=('core_rms', 'mean'), und=('undistorted', 'min'),
                                                      HC2=('HC2_sabr', 'mean'), LC1=('LC1_sabr', 'mean')).reset_index()
        if len(u) == 0: continue
        k = int(u.out.sum()); n = len(u); c = int(co.outlier99.sum()); m = len(co)
        orr, p = stats.fisher_exact([[k, n - k], [c, m - c]])
        t = stats.ttest_ind(u.maha, co.maha_dep, equal_var=False)
        uu = u[u.und == 1]; cu = co[co.undistorted == 1]
        print(f'{meth} {lab}: distinct Fvs {n} in {u.paper.nunique()} papers; outliers {k}/{n} ({k/n:.0%}) vs {c}/{m} ({c/m:.0%}) '
              f'OR {orr:.2f} Fisher p={p:.2g}; Mahalanobis {u.maha.mean():.2f} vs {co.maha_dep.mean():.2f} diff {u.maha.mean()-co.maha_dep.mean():+.2f} Welch p={t.pvalue:.2g}; '
              f'core {u.core.mean()-co.core_rms.mean():+.3f} p={stats.ttest_ind(u.core, co.core_rms, equal_var=False).pvalue:.2g}; '
              f'undistorted: outliers {int(uu.out.sum())}/{len(uu)} vs {int(cu.outlier99.sum())}/{len(cu)}')
        print('   outlier Fvs:', u[u.out == 1][['paper']].assign(core=u.core.round(2), und=u.und).values.tolist())
# struct_unusual EM controls reference: what is the paper-case contrast against the deeper EM controls?
try:
    SU = pd.read_csv(f'{DEEP}/struct_unusual/tables/tests.csv'); print(SU[SU.iloc[:, 0].astype(str).str.contains('maha|outlier', case=False)].head(10).to_string())
except Exception as e:
    print('struct_unusual tests.csv', e)
