"""Figure: flag rate vs domain length, and per-residue hazard ratios (antibody V vs non-Ig)."""
import pandas as pd, numpy as np, matplotlib, matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
T = '/home/user/SAbR/register_error_analysis/length_normalised/tables'
F = '/home/user/SAbR/register_error_analysis/length_normalised/figures'
BLUE = '#2a78d6'; ORANGE = '#eb6834'; AQUA = '#1baf7a'; GREY = '#7d7c76'; INK = '#0b0b0b'; INK2 = '#52514e'; MUTED = '#8a8983'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlesize': 11, 'legend.frameon': False, 'grid.color': '#e6e5e0'})
L = pd.read_csv(f'{T}/rates_by_length.csv'); H = pd.read_csv(f'{T}/hazard_ratios_vs_abV.csv'); A = pd.read_csv(f'{T}/af2_seen_hazard_ratios.csv')
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'wspace': 0.62, 'width_ratios': [1, 1.15]})
ax = axs[0]
mid = {'(0, 60]': 45, '(60, 80]': 70, '(80, 100]': 90, '(100, 115]': 107.5, '(115, 130]': 122.5, '(130, 160]': 145, '(160, 200]': 180,
       '(200, 250]': 225, '(250, 300]': 275, '(300, 400]': 350, '(400, 7000]': 450}
for cat, col, mk, lab in [('Non-Ig', GREY, 'o', 'Non-immunoglobulin domains'), ('Other Ig-like (non-antibody)', AQUA, 's', 'Other Ig-like domains (TCR, MHC, receptors…)'),
                          ('Antibody V', BLUE, 'D', 'Antibody V domains')]:
    s = L[(L.cat == cat) & (L.n >= 100)].copy(); s['x'] = s.lbin.map(mid)
    se = np.sqrt(s.pct / 100 * (1 - s.pct / 100) / s.n) * 100
    ax.errorbar(s.x, s.pct, yerr=1.96 * se, color=col, marker=mk, ms=6, lw=2 if cat == 'Non-Ig' else 0, elinewidth=1, capsize=2, label=lab)
ax.axvspan(100, 130, color='#e6eef9', zorder=0); ax.text(115, 6.4, 'antibody V\nlength\nrange', ha='center', fontsize=8, color=BLUE)
ax.set_xscale('log'); ax.set_xticks([50, 100, 200, 400]); ax.set_xticklabels(['50', '100', '200', '400']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel('ECOD domain length (residues, log scale)'); ax.set_ylabel('% of domains flagged (95% CI)'); ax.set_ylim(0, 10.5)
ax.yaxis.grid(True); ax.set_axisbelow(True); ax.legend(fontsize=8.3, loc='upper left', bbox_to_anchor=(0, -0.16), ncol=1)
ax.set_title('A. Flag rate rises with length, but antibody V\ndomains sit below same-length non-Ig domains')
ax = axs[1]
def inv(r): return 1 / r.ratio_vs_abV, 1 / r.hi, 1 / r.lo
rows = []
for model, lab in [('all domains, adjusted', 'All years, all methods'), ('length 100–130 only, adjusted', 'All years, length 100–130 only'),
                   ('era ≤2017, adjusted', 'Deposited ≤2017'), ('era 2018–19, adjusted', 'Deposited 2018–19'), ('era 2020–22, adjusted', 'Deposited 2020–22'),
                   ('cryo-EM 2020–22, adjusted', 'Cryo-EM, 2020–22')]:
    r = H[(H.model == model) & (H.measure == 'flag hazard per residue') & (H.comparator == 'Non-Ig')].iloc[0]
    rows.append((lab,) + inv(r) + (BLUE,))
for model, lab in [('2020–22 all: unseen proteins only', '2020–22, proteins new to the PDB'), ('2020–22 EM: unseen proteins only', 'Cryo-EM 2020–22, proteins new to the PDB')]:
    r = A[A.model == model].iloc[0]; rows.append((lab, r.HR_abV_vs_nonIg, r.lo, r.hi, ORANGE))
for i, (lab, o, lo, hi, col) in enumerate(rows):
    ax.plot([lo, hi], [i, i], color=col, lw=2); ax.plot(o, i, 'o', color=col, ms=7, mec='white', mew=1.3)
    sig = lo > 1 or hi < 1
    ax.text(14, i, f'{o:.2f} ({lo:.2f}–{hi:.2f})', va='center', fontsize=8.3, color=INK if sig else MUTED)
ax.axvline(1, color=MUTED, ls='--', lw=1); ax.set_xscale('log'); ax.set_xlim(0.03, 12)
ax.set_xticks([0.05, 0.1, 0.3, 1, 3, 10]); ax.set_xticklabels(['0.05', '0.1', '0.3', '1', '3', '10']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8.6); ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.set_xlabel('← antibody V lower      per-residue hazard ratio      antibody V higher →')
ax.set_title('B. Antibody V vs non-Ig domains, per residue\n(conkit flags; adjusted for method × resolution and era)')
ax.text(14, -0.75, 'HR (95% CI)', fontsize=8.3, color=INK2, fontweight='bold'); ax.set_ylim(len(rows) - 0.5, -1.1)
fig.savefig(f'{F}/length_normalised.png', dpi=200, bbox_inches='tight', pad_inches=0.12, facecolor='white'); print('ok')
