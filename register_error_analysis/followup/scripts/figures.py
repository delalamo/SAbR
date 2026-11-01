import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
A='/tmp/analysis'; O=f'{A}/out'; F='/home/user/SAbR/register_error_analysis/followup/figures'
S1,S2,S3,S4='#2a78d6','#eb6834','#1baf7a','#eda100'; SURF='#fcfcfb'; TP='#0b0b0b'; TS='#52514e'; GRID='#e6e5e0'
plt.rcParams.update({'font.size':10,'axes.edgecolor':GRID,'axes.labelcolor':TS,'xtick.color':TS,'ytick.color':TS,'axes.facecolor':SURF,'figure.facecolor':SURF,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6,'axes.axisbelow':True,'legend.frameon':False})
def title(ax,t,s=None):
    ax.set_title(t,loc='left',color=TP,fontsize=11.5,fontweight='bold',pad=18 if s else 8)
    if s: ax.text(0,1.02,s,transform=ax.transAxes,color=TS,fontsize=9)
# --- Fig 1: EM antibody use by year
T=pd.read_csv(f'{O}/A_em_antibody_use_by_year.csv',index_col=0); T=T.loc[2014:2025]
fig,ax=plt.subplots(figsize=(7.5,4.2))
for col,c,lab in [('pct_EM_with_aux_antibody',S1,'Antibody as auxiliary (fiducial / stabiliser)'),('pct_EM_with_antibody_of_interest',S2,'Antibody is the subject')]:
    ax.plot(T.index,T[col],color=c,lw=2,marker='o',ms=5,markeredgecolor=SURF,markeredgewidth=1.5,label=lab)
    ax.text(T.index[-1]+0.15,T[col].iloc[-1],f'{T[col].iloc[-1]:.1f}%',color=TP,va='center',fontsize=9)
ax.set_ylabel('% of released cryo-EM entries'); ax.set_xticks(range(2014,2026,2)); ax.set_xlim(2013.6,2025.9); ax.set_ylim(0,None)
ax.legend(loc='upper left'); title(ax,'Antibodies in cryo-EM entries, by deposition year','All EM entries in the PDB (any resolution); 2025 incomplete due to release holds')
fig.tight_layout(); fig.savefig(f'{F}/fig1_em_antibody_use.png',dpi=160); plt.close()
# --- Fig 2: auxiliary types stacked
st=pd.read_csv(f'{O}/A_aux_types_by_year.csv',index_col=0).loc[2017:2025]
order=['G-protein stabiliser (Nb35/scFv16)','Other Fab / scFv','Other nanobody','Engineered fiducial/synthetic binder (BRIL-Fab, NabFab, megabody, legobody, sybody, sAB)']
labs=['G-protein stabiliser (Nb35 / scFv16)','Other Fab / scFv','Other nanobody','Engineered fiducial / synthetic binder']
fig,ax=plt.subplots(figsize=(7.5,4.2)); bottom=np.zeros(len(st))
for col,c,l in zip(order,[S1,S2,S3,S4],labs):
    ax.bar(st.index,st[col],bottom=bottom,color=c,width=0.72,edgecolor=SURF,linewidth=2,label=l); bottom+=st[col].values
for x,y in zip(st.index,bottom): ax.text(x,y+8,int(y),ha='center',color=TS,fontsize=8.5)
ax.set_ylabel('Cryo-EM entries'); ax.legend(loc='upper left',fontsize=8.5); ax.grid(axis='x',visible=False)
title(ax,'Auxiliary antibodies in cryo-EM, by type','Entries per deposition year; 2025 incomplete')
fig.tight_layout(); fig.savefig(f'{F}/fig2_aux_types.png',dpi=160); plt.close()
# --- Fig 3: B by era
B=pd.read_csv(f'{O}/B_rates_by_era_method.csv'); eras=['≤2017','2018–19','2020–22']
fig,ax=plt.subplots(figsize=(7.5,4.6))
for cls,c,lab,off in [('Antibody V',S1,'Antibody V domains',-0.06),('Non-antibody, other entries',S2,'All other domains (entries without antibodies)',0.06)]:
    s=B[B.cls==cls].set_index('era').loc[eras]; x=np.arange(3)+off
    ax.errorbar(x,s.flag_pct,yerr=[s.flag_pct-s.lo,s.hi-s.flag_pct],color=c,lw=2,marker='o',ms=6,capsize=0,elinewidth=1.2,markeredgecolor=SURF,label=lab)
    for k,(xi,(r,em)) in enumerate(zip(x,zip(s.flag_pct,s.pct_EM))):
        dy=(0.28 if cls=='Antibody V' else -0.28) if k==2 else 0
        ax.text(xi+0.07,r+dy,f'{r:.1f}%  ({em:.0f}% EM)',color=TP,fontsize=8.5,va='center')
ax.set_xticks(range(3)); ax.set_xticklabels([f'Deposited {e}' for e in eras]); ax.set_ylabel('% of domains with a flagged register error'); ax.set_ylim(0,None); ax.set_xlim(-0.3,2.75)
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.12),ncol=2,fontsize=8.5); ax.grid(axis='x',visible=False)
title(ax,'Register-error rate during the X-ray → cryo-EM transition','3–5 Å entries; 95% CIs from bootstrap over entries; label shows % cryo-EM')
fig.tight_layout(); fig.savefig(f'{F}/fig3_transition.png',dpi=160); plt.close()
# --- Fig 4: role x prior template
ab=pd.read_pickle(f'{A}/ab_prior.pkl'); v=ab[(ab.cls=='Antibody V')&ab.role.isin(['Auxiliary','Antibody of interest'])]
rng=np.random.default_rng(0)
def ci(s):
    g=s.groupby('pdb').flag.agg(['sum','count']); a,c=g['sum'].values,g['count'].values;n=len(g)
    b=[a[i].sum()/c[i].sum() for i in (rng.integers(0,n,n) for _ in range(500))]; return 100*a.sum()/c.sum(),*np.percentile(b,[2.5,97.5])*100
fig,ax=plt.subplots(figsize=(7.5,4.0)); rows=[]
for j,(pr,lab) in enumerate([(0,'No prior high-res structure of this antibody'),(1,'Identical V region already in PDB at <3 Å')]):
    for role,c,off in [('Antibody of interest',S2,-0.08),('Auxiliary',S1,0.08)]:
        s=v[(v.prior_hires==pr)&(v.role==role)]; r,lo,hi=ci(s); rows.append((lab,role,len(s),int(s.flag.sum()),r,lo,hi))
        ax.errorbar(j+off,r,yerr=[[r-lo],[hi-r]],color=c,marker='o',ms=7,lw=0,elinewidth=1.4,markeredgecolor=SURF,label=role if j==0 else None)
        ax.text(j+off+(0.05 if off>0 else -0.05),r,f'{r:.1f}% ({int(s.flag.sum())}/{len(s)})',color=TP,fontsize=8.5,va='center',ha='left' if off>0 else 'right')
ax.set_xticks([0,1]); ax.set_xticklabels(['No prior high-res structure\nof this antibody','Identical V region already\nin PDB at <3 Å']); ax.set_xlim(-0.4,1.6); ax.set_ylim(0,None)
ax.set_ylabel('% antibody V domains flagged'); ax.legend(loc='upper right',fontsize=9); ax.grid(axis='x',visible=False)
title(ax,'What predicts antibody errors: a reusable model, not author interest','3–5 Å entries containing antibodies with a partner chain; 95% bootstrap CIs')
fig.tight_layout(); fig.savefig(f'{F}/fig4_role_prior.png',dpi=160); plt.close()
pd.DataFrame(rows,columns=['prior','role','domains','flagged','pct','lo','hi']).round(2).to_csv(f'{O}/fig4_table.csv',index=False)
print('ok')
