import sys, json, numpy as np, pandas as pd, matplotlib, matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
R='/home/user/SAbR/register_error_analysis'
OUT=f'{SP}/report/img'
BLUE='#2a78d6'; ORANGE='#eb6834'; AQUA='#1baf7a'; YELLOW='#eda100'; MAG='#e87ba4'; GREEN='#008300'; VIOLET='#6250d6'; RED='#e34948'
INK='#0b0b0b'; INK2='#52514e'; MUTED='#8a8983'; GRID='#e6e5e0'; NEUTRAL='#b9b8b1'; PALE='#f0efec'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.edgecolor':'#c9c8c2','axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,
  'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'grid.color':GRID,'grid.linewidth':0.8,'axes.titleweight':'bold','axes.titlesize':11.5,
  'axes.titlecolor':INK,'axes.titlelocation':'left','legend.frameon':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.png',dpi=200,bbox_inches='tight',pad_inches=0.12); plt.close(fig); print('saved',name)
def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return (max(0,c-h)*100,(c+h)*100)
def ygrid(ax): ax.yaxis.grid(True); ax.set_axisbelow(True)
def xgrid(ax): ax.xaxis.grid(True); ax.set_axisbelow(True)
def bars(ax,x,vals,colors,width=0.7,**kw):
    return ax.bar(x,vals,width=width,color=colors,edgecolor='white',linewidth=1.5,**kw)

# ---------------------------------------------------------------- F: concept schematic
def f_concept():
    true='GSWYAFKLT'; vol={'G':60,'S':89,'W':228,'Y':194,'A':89,'F':190,'K':169,'L':167,'T':116}
    fig,axs=plt.subplots(2,1,figsize=(8.4,3.6))
    for ax,(title,model,col) in zip(axs,[('A. Correct register: each residue sits in density of the right size','GSWYAFKLT',BLUE),
                                         ('B. Register error (shift of +1): the same backbone, but every residue is one slot off','SWYAFKLTV',ORANGE)]):
        ax.set_xlim(-2.0,9.6); ax.set_ylim(-1.25,1.35); ax.axis('off'); ax.set_title(title,fontsize=10.5)
        xs=np.arange(9); ax.plot(xs,np.zeros(9),color='#7d7c76',lw=3,zorder=1,solid_capstyle='round')
        for i,(t,m) in enumerate(zip(true,model)):
            r=0.16+0.0019*vol[t]
            ax.add_patch(Circle((i,0.62),r,color='#cfe0f5',ec='#9ec5f4',lw=1,zorder=0))
            ax.plot([i,i],[0,0.45],color='#7d7c76',lw=1.6,zorder=1)
            ok=(m==t); ax.text(i,0.62,m,ha='center',va='center',fontsize=13,fontweight='bold',color=(INK if ok else ORANGE),zorder=3)
            ax.add_patch(Circle((i,0),0.09,color='#4d4c47',zorder=2))
            ax.text(i,-0.42,str(i+1),ha='center',va='center',fontsize=8.5,color=MUTED)
        ax.text(-1.95,0.62,'density +\nmodelled\nresidue',ha='left',va='center',fontsize=7.5,color=INK2)
        ax.text(-1.95,-0.42,'backbone\nposition',ha='left',va='center',fontsize=7.5,color=INK2)
        if col==ORANGE:
            ax.text(4,-1.0,'Backbone geometry is right, so global fit statistics look fine; only side chains (and their contacts) betray the shift.',ha='center',fontsize=8.5,color=INK2)
    save(fig,'f_concept')

# ---------------------------------------------------------------- F: scale (antibody V domains 3-5 A per year)
def f_scale():
    v=pd.read_csv(f'{R}/sabr_anarci_scan/tables/v_domain_flags.csv.gz')
    v=v[v.year>=2010]
    ct=pd.crosstab(v.year,v.method); fl=v.groupby(['year','method']).flag.sum().unstack().fillna(0)
    fig,axs=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={'wspace':0.28})
    ax=axs[0]; yrs=ct.index.values
    ax.bar(yrs,ct['X-ray'],color=ORANGE,width=0.8,edgecolor='white',lw=1,label='X-ray')
    ax.bar(yrs,ct['EM'],bottom=ct['X-ray'],color=BLUE,width=0.8,edgecolor='white',lw=1,label='Cryo-EM')
    ax.set_title('A. Antibody V domains at 3–5 Å, per year'); ygrid(ax); ax.legend(loc='upper left')
    ax.set_ylabel('V domains (chains)'); ax.text(2026,ct.loc[2026].sum()+60,'partial\nyear',ha='center',fontsize=7.5,color=MUTED)
    ax=axs[1]
    ax.bar(yrs,fl['X-ray'],color=ORANGE,width=0.8,edgecolor='white',lw=1,label='X-ray')
    ax.bar(yrs,fl['EM'],bottom=fl['X-ray'],color=BLUE,width=0.8,edgecolor='white',lw=1,label='Cryo-EM')
    ax.set_title('B. …with a detectable register error'); ygrid(ax); ax.set_ylabel('V domains flagged (SAbR vs ANARCI)')
    for a in axs: a.set_xticks(range(2010,2027,4))
    save(fig,'f_scale')

# ---------------------------------------------------------------- F: fiducial use
def f_fiducial():
    a=pd.read_csv(f'{R}/followup/tables/A_em_antibody_use_by_year.csv'); a=a[(a.year>=2014)&(a.year<=2025)]
    fig,axs=plt.subplots(1,2,figsize=(10,3.4),gridspec_kw={'wspace':0.28})
    ax=axs[0]; ax.bar(a.year,a.EM_entries,color=NEUTRAL,width=0.8,edgecolor='white'); ax.set_title('A. Cryo-EM entries released per year'); ygrid(ax)
    ax=axs[1]
    ax.plot(a.year,a.pct_EM_with_aux_antibody,color=ORANGE,lw=2,marker='o',ms=5,label='antibody used as a tool (auxiliary)')
    ax.plot(a.year,a.pct_EM_with_antibody_of_interest,color=BLUE,lw=2,marker='o',ms=5,label='antibody is the subject')
    ax.plot(a.year,a.pct_membrane_like_with_aux,color=ORANGE,lw=1.6,ls='--',label='auxiliary, membrane-protein entries only')
    ax.set_title('B. % of cryo-EM entries with an antibody'); ygrid(ax); ax.set_ylabel('% of entries'); ax.legend(loc='upper left',fontsize=8.5)
    save(fig,'f_fiducial')

# ---------------------------------------------------------------- F: fold classes (conkit)
def f_folds():
    t=pd.read_csv(f'{R}/table_raw.csv')
    keep=['Other Ig-like X-group','alpha complex topology','alpha superhelices','a/b three-layered sandwiches','Other Ig/E-set/PapD T-group','beta barrels','Other β-sandwiches',
          'alpha bundles','alpha arrays','a+b two layers','TCR V','Antibody V','Other IgSF (V/C1/C2/I-set; MHC, β2m, receptors)','Antibody C','extended segments']
    nice={'Other Ig-like X-group':'Other Ig-like folds (receptor LBDs, C2, …)','alpha complex topology':'α complex topology','alpha superhelices':'α superhelices',
          'a/b three-layered sandwiches':'α/β three-layer sandwiches','Other Ig/E-set/PapD T-group':'Other Ig / E-set (non-antibody)','beta barrels':'β-barrels',
          'Other β-sandwiches':'Other β-sandwiches','alpha bundles':'α bundles','alpha arrays':'α arrays','a+b two layers':'α+β two layers','TCR V':'T-cell receptor V',
          'Antibody V':'Antibody V domain','Other IgSF (V/C1/C2/I-set; MHC, β2m, receptors)':'Other Ig-superfamily (MHC, β2m, receptors)','Antibody C':'Antibody C domain','extended segments':'Extended segments'}
    t=t[t.group.isin(keep)].copy(); t=t.sort_values('pct_flagged')
    lo=[];hi=[]
    for c,p in zip(t.ci,t.pct_flagged):
        try: a,b=str(c).split('–'); lo.append(p-float(a)); hi.append(float(b)-p)
        except Exception: lo.append(0); hi.append(0)
    fig,ax=plt.subplots(figsize=(8.2,5.2))
    cols=[BLUE if g.startswith('Antibody') else (AQUA if g in('TCR V','Other IgSF (V/C1/C2/I-set; MHC, β2m, receptors)') else NEUTRAL) for g in t.group]
    y=np.arange(len(t)); ax.barh(y,t.pct_flagged,color=cols,height=0.68,edgecolor='white')
    ax.errorbar(t.pct_flagged,y,xerr=[lo,hi],fmt='none',ecolor='#5f5e59',elinewidth=1,capsize=2)
    ax.set_yticks(y); ax.set_yticklabels([nice[g] for g in t.group]); xgrid(ax); ax.set_xlabel('% of domains with ≥1 flagged register error (95% CI)')
    for yi,(p,n) in enumerate(zip(t.pct_flagged,t.domains)): ax.text(p+0.25+(hi[yi] if hi[yi] else 0),yi,f'{p:.1f}%  (n={n:,})',va='center',fontsize=8,color=INK2)
    ax.set_xlim(0,17.5); ax.set_title('Register-error flag rate by fold class, 3–5 Å X-ray + cryo-EM, PDB to April 2022 (conkit-validate)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE,label='antibody domains'),Patch(color=AQUA,label='closest relatives (TCR, other Ig-superfamily)'),Patch(color=NEUTRAL,label='other fold classes')],loc='lower right',fontsize=8.5)
    save(fig,'f_folds')

# ---------------------------------------------------------------- F: conkit era trend
def f_era_conkit():
    """Per deposition year (2010-2022); data from followup/scripts/by_year.py"""
    from scipy.stats import beta
    t=pd.read_csv(f'{R}/followup/tables/B_by_year.csv')
    ab=t[t.cls=='Antibody V'].set_index('year'); oth=t[t.cls=='Non-antibody, other entries'].set_index('year')
    yrs=ab.index.values
    fig,axs=plt.subplots(3,1,figsize=(10,8.2),sharex=True,gridspec_kw={'height_ratios':[3,2.4,1.05],'hspace':0.28})
    for ax in axs:
        ax.axvspan(2021.5,2022.5,color='#f3f2ee',lw=0,zorder=0)
        ax.axvline(2018.33,color=MUTED,lw=1,ls=':',zorder=1)
    ax=axs[0]
    ax.fill_between(oth.index,oth.lo,oth.hi,color=NEUTRAL,alpha=0.35,lw=0)
    ax.plot(oth.index,oth.flag_pct,color='#7d7c76',lw=2.2,marker='o',ms=6,label='All other domains (entries without antibodies)')
    lo=ab.lo.copy(); hi=ab.hi.copy()
    z=ab.flagged==0
    hi[z]=100*(1-0.025**(1/ab.domains[z]))   # Clopper-Pearson upper bound when nothing was flagged
    ax.errorbar(yrs,ab.flag_pct,yerr=[ab.flag_pct-lo,hi-ab.flag_pct],color=BLUE,lw=2.2,marker='o',ms=6,elinewidth=1.2,capsize=2.5,label='Antibody V domains')
    for y in [2019,2020,2021]: ax.text(y+0.1,ab.flag_pct[y]+(0.75 if y==2019 else 0.35),f"{ab.flag_pct[y]:.1f}%",fontsize=8.5,color=BLUE)
    ax.set_ylabel('% of domains flagged (95% CI)'); ax.set_ylim(0,11); ygrid(ax); ax.legend(loc='upper center',bbox_to_anchor=(0.47,1.0),fontsize=8.6)
    ax.set_title('A. Flag rate by deposition year (2024 conkit flags)')
    ax.text(2022,10.4,'2022:\nJan–Apr\nonly',fontsize=7.8,color=MUTED,va='top',ha='center')
    ax=axs[1]
    ok=ab.observed>0
    lo_c=np.maximum(ab.OE_lo[ok],0.02); ax.errorbar(yrs[ok],ab.OE[ok],yerr=[ab.OE[ok]-lo_c,ab.OE_hi[ok]-ab.OE[ok]],fmt='o',color=BLUE,ms=7,mec='white',mew=1.2,elinewidth=1.5,capsize=2.5)
    floor=0.025
    for y in yrs[~ok]:
        ax.plot([y,y],[floor,ab.OE_hi[y]],color=BLUE,lw=1.5); ax.plot(y,floor,marker='v',color=BLUE,ms=7,mfc='white',mew=1.3)
    for y in yrs: ax.text(y,0.0115,f"{int(ab.observed[y])}/{ab.expected[y]:.0f}",ha='center',fontsize=7.4,color=INK2)
    ax.axhline(1,color=MUTED,lw=1,ls='--'); ax.set_yscale('log'); ax.set_ylim(0.009,4)
    ax.set_yticks([0.03,0.1,0.3,1,3]); ax.set_yticklabels(['0.03','0.1','0.3','1','3']); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel('observed / expected\n(95% CI, log scale)'); ygrid(ax)
    ax.set_title('B. Antibody V flags vs same-year, same-method/resolution, same-length other domains')
    ax.text(2009.6,1.12,'antibodies worse ↑',fontsize=7.8,color=MUTED); ax.text(2009.6,0.78,'antibodies better ↓',fontsize=7.8,color=MUTED,va='top')
    ax=axs[2]
    ax.bar(yrs,ab.pct_EM,color=BLUE,alpha=0.55,width=0.7,edgecolor='white')
    for y in yrs: ax.text(y,ab.pct_EM[y]+4,f"{int(ab.domains[y]):,}",ha='center',fontsize=7.2,color=INK2)
    ax.set_ylim(0,125); ax.set_yticks([0,50,100]); ax.set_ylabel('% cryo-EM'); ygrid(ax)
    ax.text(2018.25,108,'AlphaFold2 training cut-off (Apr 2018)',fontsize=7.6,color=MUTED,ha='right',va='center')
    ax.set_title('C. Antibody V domains: share from cryo-EM (number of domains above each bar)',fontsize=10.5)
    ax.set_xticks(yrs); ax.set_xticklabels([str(y) for y in yrs]); ax.set_xlim(2009.4,2022.6)
    save(fig,'f_era_conkit')

# ---------------------------------------------------------------- ladders
sys.path.insert(0,f'{SP}/scripts')
def draw_ladder(ax, positions, rows, title, note=None, poslabel='Structural position (IMGT, from the reference)', L=4.6):
    import textwrap
    n=len(positions); ax.set_xlim(-L,n+0.1); ax.set_ylim(-0.95,len(rows)+0.75); ax.axis('off')
    ax.set_title(title,fontsize=10.5)
    for j,p in enumerate(positions): ax.text(j+0.5,len(rows)+0.0,p,ha='center',va='bottom',fontsize=8,color=MUTED)
    ax.text(-L+0.05,len(rows)+0.0,poslabel,ha='left',va='bottom',fontsize=7.2,color=MUTED)
    for i,(lab,cells) in enumerate(rows):
        y=len(rows)-1-i
        ax.text(-L+0.05,y+0.45,lab,ha='left',va='center',fontsize=8.6,color=INK)
        for j,c in enumerate(cells):
            if c is None:
                ax.add_patch(Rectangle((j+0.06,y+0.08),0.88,0.78,fc='white',ec='#c9c8c2',ls='--',lw=0.8)); continue
            letter,sub,state=c
            fc={'ok':'#e8eef7','ref':'#eef6f1','bad':'#fbe0d4','bad2':'#f6c3c2','na':'#f3f2ee'}[state]
            ec={'ok':'#b7c9e4','ref':'#bfe0cd','bad':ORANGE,'bad2':RED,'na':'#dcdbd5'}[state]
            ax.add_patch(FancyBboxPatch((j+0.08,y+0.1),0.84,0.74,boxstyle='round,pad=0.0,rounding_size=0.08',fc=fc,ec=ec,lw=1.2))
            ax.text(j+0.5,y+(0.57 if sub else 0.47),letter,ha='center',va='center',fontsize=11 if len(letter)<3 else 9,fontweight='bold',color=INK)
            if sub: ax.text(j+0.5,y+0.22,sub,ha='center',va='center',fontsize=6.0,color=INK2)
    if note: ax.text(-L+0.05,-0.15,'\n'.join(textwrap.wrap(note,int((n+L)*7.2))),ha='left',va='top',fontsize=7.8,color=INK2)

def ladder_rows(ref,q,lo,hi,qname,refname):
    from ladder import ladder
    rows,rms,n=ladder(ref,q,lo,hi)
    pos=[str(r['pos'][0])+r['pos'][1] for r in rows]
    refc=[(r['ref'],'',  'ref') for r in rows]
    qc=[]
    for r in rows:
        if r['q'] is None: qc.append(None); continue
        ql=r['qlab']; sub=f"seq#{ql[0]}{ql[1]}" if ql else ''
        st='ok' if (ql is not None and ql==r['pos']) else ('bad2' if (ql and abs(ql[0]-r['pos'][0])>=2) else 'bad')
        qc.append((r['q'],sub,st))
    return pos,[(refname,refc),(qname,qc)],rms,n

def f_ladder_1b2():
    from ladder import ladder
    pos,rows1,rms1,n1=ladder_rows('6mlk_L','6c9u_L',1,14,'6C9U: Fab 1B2 (2.09 Å X-ray, 2018)','6MLK: Fab 3A6 (2.45 Å X-ray, 2018)')
    _,rows2,rms2,n2=ladder_rows('6mlk_L','7m7f_D',1,14,'7M7F: Fab 1B2 (3.5 Å cryo-EM, 2021)','')
    rows=rows1+[rows2[1]]
    fig,ax=plt.subplots(figsize=(0.62*(len(pos)+4.6),3.0))
    draw_ladder(ax,pos,rows,'Light-chain FR1 shared by Fabs 1B2 and 3A6: which residue occupies each backbone position',
        note=f'Superposed on framework Cα outside this window (6C9U: {rms1:.2f} Å over {n1}; 7M7F: {rms2:.2f} Å over {n2}). Small text = the residue\'s own sequence position (ANARCI). Orange = residue sits one slot from where its sequence says it belongs.')
    save(fig,'f_ladder_1b2')

def f_ladder_7m7f():
    from render import labels
    L=labels('7m7f_D'); items=sorted(L.items())[:14]
    pos=[str(i+1) for i in range(len(items))]
    seq=[(v[0],) for k,v in items]
    from render import AA1
    an=[];sb=[]
    for k,(aa,la,ls) in items:
        a=f'{la[0]}{la[1]}' if la else '–'; s=f'{ls[0]}{ls[1]}'
        st='ok' if la==ls else 'bad'
        an.append((a,'', 'ref')); sb.append((s,'',st))
    letters=[(AA1.get(v[0],'X'),'', 'na') for k,v in items]
    fig,ax=plt.subplots(figsize=(0.62*(14+4.6),3.1))
    draw_ladder(ax,[str(k[0]) for k,v in items],[('Residue (7M7F chain D)',letters),('ANARCI number (from sequence)',an),('SAbR number (from structure)',sb)],
        'The detector on one chain: Fab 1B2 light chain in 7M7F (3.5 Å cryo-EM)',poslabel='Author residue number',
        note='ANARCI aligns the sequence to the IMGT template; SAbR places each residue by where its backbone actually sits. Eight consecutive disagreements (orange) form one register-error run: the first seven residues are each built one slot too early.')
    save(fig,'f_ladder_7m7f')

def f_ladder_mech():
    fig,axs=plt.subplots(3,1,figsize=(0.62*(14+4.6),8.4),gridspec_kw={'hspace':0.6})
    pos,rows,rms,n=ladder_rows('7yv1_L','6bpe_C',1,14,'6BPE chain C (3.7 Å cryo-EM)','7YV1 κ (1.45 Å X-ray)')
    draw_ladder(axs[0],pos,rows,'A. Chain end: the N-terminal strand slides by one or two positions',note=f'Different antibodies, so letters differ; what matters is the small sequence number. Superposed on framework: {rms:.2f} Å over {n} Cα. Pro8 is missing (−).')
    pos,rows,rms,n=ladder_rows('7yv1_H','7v2a_G',114,125,'7V2A chain G (3.4 Å cryo-EM)','7YV1 heavy (1.45 Å X-ray)')
    draw_ladder(axs[1],pos,rows,'B. CDR3–FR4 junction: the strand after CDR-H3 starts one residue early',note=f'Val117 occupies the slot of the conserved Trp118; the Trp is pushed one or two slots along the strand. {rms:.2f} Å over {n} Cα.')
    pos,rows,rms,n=ladder_rows('7yv1_H','7mkl_H',116,126,'7MKL chain H (3.9 Å cryo-EM)','7YV1 heavy (1.45 Å X-ray)')
    draw_ladder(axs[2],pos,rows,'C. FR4 bulge: in register through Trp118–Gly119, then the strand jumps a slot',note=f'Gln120 is stretched across two slots; everything after it sits one position ahead. {rms:.2f} Å over {n} Cα.')
    save(fig,'f_ladder_mech')

def f_ladder_7mxl():
    import numpy as np
    from render import atoms,pdbtext,kabsch,AA1
    aB=atoms(pdbtext('7mxl_B')); aK=atoms(pdbtext('7mxl_K'))
    def seq(a,lo,hi):
        out={}
        for l in pdbtext(a).splitlines():
            if l.startswith('ATOM') and l[12:16].strip()=='CA': out[int(l[22:26])]=AA1.get(l[17:20],'X')
        return out
    sB=seq('7mxl_B',0,0); sK=seq('7mxl_K',0,0)
    best=max(range(-4,5),key=lambda o: sum(1 for n in sB if n>=110 and (n+o) in sK and sK[n+o]==sB[n]))
    # superpose B onto K on C-kappa outside 136..176
    com=[n for n in sB if 110<=n<=210 and not (136<=n<=176) and (n+best) in sK and (n,'') in aB and (n+best,'') in aK]
    P=np.array([aK[(n+best,'')]['CA'] for n in com]); Q=np.array([aB[(n,'')]['CA'] for n in com]); Rm,pc,qc=kabsch(P,Q)
    rms=float(np.sqrt((((Q-qc)@Rm.T+pc-P)**2).sum(1).mean()))
    Bca={n:(aB[(n,'')]['CA']-qc)@Rm.T+pc for n in sB if (n,'') in aB}
    pos=[];refc=[];qcl=[]
    for nk in range(138+best,176+best):
        if (nk,'') not in aK: continue
        pos.append(str(nk-best)); refc.append((sK[nk],'','ref'))
        d=min((np.linalg.norm(Bca[n]-aK[(nk,'')]['CA']),n) for n in Bca if abs(n-(nk-best))<=6)
        if d[0]>3.2: qcl.append(None); continue
        nb=d[1]; sh=nb-(nk-best)
        qcl.append((sB[nb],f'#{nb}', 'ok' if sh==0 else ('bad2' if abs(sh)>=2 else 'bad')))
    fig,ax=plt.subplots(figsize=(0.5*(len(pos)+4.6),2.8))
    draw_ladder(ax,pos,[('7MXL chain K (correct)',refc),('7MXL chain B (flagged)',qcl)],
        'REGN5713 light chain Cκ (7MXL, 3.2 Å cryo-EM): residues either side of an unbuilt stretch are shifted',
        poslabel='Position (chain-B numbering)',note=f'Same map, identical Cκ sequence. Chain B superposed on chain K using Cκ Cα outside the window ({rms:.2f} Å over {len(com)}). Dashed = no chain-B residue within 3.2 Å (149–158 not built).')
    save(fig,'f_ladder_7mxl')
    print('7mxl offset',best)

# ---------------------------------------------------------------- validation
def f_validation():
    fig,axs=plt.subplots(1,2,figsize=(10.4,3.6),gridspec_kw={'wspace':0.55,'width_ratios':[1.05,1]})
    ax=axs[0]
    labs=['All 134 flagged V-region errors','Framework errors (≥75% FR)','Mixed FR/CDR errors','Errors mostly inside a CDR','Errors of ≤6 residues','Errors of 7–10 residues','Errors of >10 residues']
    vals=[73.1,84,77,16,56,74,92]; cols=[BLUE,BLUE,BLUE,NEUTRAL,AQUA,AQUA,AQUA]
    y=np.arange(len(vals)); ax.barh(y,vals,color=cols,height=0.66,edgecolor='white'); ax.set_yticks(y); ax.set_yticklabels(labs,fontsize=8.5); ax.invert_yaxis()
    for yi,v in zip(y,vals): ax.text(v+1.5,yi,f'{v:g}%',va='center',fontsize=8.5,color=INK2)
    ax.text(23,3,'blind spot: CDR numbering is length-based',va='center',fontsize=7.5,color=MUTED)
    ax.set_xlim(0,108); xgrid(ax); ax.set_xlabel('% detected by SAbR vs ANARCI'); ax.set_title('A. Sensitivity against the 2024 conkit flags')
    ax=axs[1]
    grp=['X-ray\n≤Apr 2022','X-ray\nnewer','EM\n≤Apr 2022','EM\nnewer']
    det=[17.0,17.3,34.9,35.6]; ctl=[38.2,35.4,45.1,40.8]
    x=np.arange(4); w=0.36
    bars(ax,x-w/2,det,ORANGE,width=w,label='disagreement runs'); bars(ax,x+w/2,ctl,NEUTRAL,width=w,label='same positions, clean V domains')
    for xi,a,b in zip(x,det,ctl): ax.text(xi-w/2,a+1.2,f'{a:.0f}',ha='center',fontsize=8.5,color=INK2); ax.text(xi+w/2,b+1.2,f'{b:.0f}',ha='center',fontsize=8.5,color=INK2)
    ax.axhline(50,color=MUTED,lw=1,ls=':'); ax.text(3.45,51,'typical\nresidue',fontsize=7.5,color=MUTED,ha='right',va='bottom')
    ax.set_xticks(x); ax.set_xticklabels(grp,fontsize=8.5); ax.set_ylim(0,62); ygrid(ax); ax.set_ylabel('median percentile of map fit\nwithin its own V domain'); ax.legend(fontsize=8,loc='upper left')
    ax.set_title('B. Flagged segments fit the map worse')
    save(fig,'f_validation')

# ---------------------------------------------------------------- SAbR trend
def f_trend_sabr():
    h=[('2018H1',10.2,157),('2018H2',9.1,297),('2019H1',13.1,183),('2019H2',1.2,164),('2020H1',3.9,309),('2020H2',13.2,552),('2021H1',9.5,803),('2021H2',3.4,726),
       ('2022H1',2.6,1095),('2022H2',3.0,812),('2023H1',3.3,635),('2023H2',4.3,739),('2024H1',7.6,907),('2024H2',3.2,820),('2025H1',6.3,942),('2025H2',2.5,812),('2026H1',4.0,377)]
    xs=np.array([int(a[:4])+(0.25 if a.endswith('H1') else 0.75) for a,_,_ in h]); p=np.array([b for _,b,_ in h]); n=np.array([c for *_,c in h])
    ci=np.array([wilson(round(pp*nn/100),nn) for pp,nn in zip(p,n)])
    fig,ax=plt.subplots(figsize=(9.6,3.9))
    ax.fill_between(xs,ci[:,0],ci[:,1],color=BLUE,alpha=0.14,lw=0); ax.plot(xs,p,color=BLUE,lw=2,marker='o',ms=5,label='Cryo-EM, half-year (95% CI)')
    xr=[(2017.0,1.9),(2019.0,1.6),(2021.0,1.7),(2023.0,1.3),(2025.0,1.9)]
    ax.plot([a for a,_ in xr],[b for _,b in xr],color=ORANGE,lw=2,marker='s',ms=5,label='X-ray, 2-year bins')
    ax.axvline(2021.54,color=INK2,lw=1,ls='--'); ax.text(2021.6,19.2,'AlphaFold2 code +\nmodels released\n(Jul 2021)',fontsize=7.8,color=INK2,va='top')
    ax.set_ylim(0,20); ygrid(ax); ax.set_ylabel('% of V domains with ≥1 disagreement run'); ax.legend(loc='upper left',fontsize=8.5); ax.set_title('Register-error rate of antibody V domains over time, SAbR vs ANARCI (3–5 Å)')
    ax.set_xticks(range(2017,2027)); ax.set_xticklabels([str(y) for y in range(2017,2027)],fontsize=8.5)
    save(fig,'f_trend_sabr')

def f_decomp():
    fig,ax=plt.subplots(figsize=(8.4,2.9))
    labs=['≤2019 → 2020–21','2020–21 → 2022–26']
    comp=[2.92,0.48]; within=[0.00,-2.91]
    y=np.arange(2)
    ax.barh(y+0.18,comp,height=0.34,color=VIOLET,edgecolor='white',label='method/resolution mix (composition)')
    ax.barh(y-0.18,within,height=0.34,color=AQUA,edgecolor='white',label='within method × resolution')
    for yi,(c,w) in enumerate(zip(comp,within)):
        ax.text(c+(0.1 if c>=0 else -0.1),yi+0.18,f'{c:+.2f} pp',va='center',ha='left' if c>=0 else 'right',fontsize=8.5,color=INK2)
        ax.text(w+(0.1 if w>=0 else -0.1),yi-0.18,f'{w:+.2f} pp',va='center',ha='left' if w>=0 else 'right',fontsize=8.5,color=INK2)
    ax.axvline(0,color=MUTED,lw=1); ax.set_yticks(y); ax.set_yticklabels(['≤2019 → 2020–21\n3.1% → 6.1%','2020–21 → 2022–26\n6.1% → 3.6%'],fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(-4.2,4.2); xgrid(ax); ax.set_xlabel('change in flag rate (percentage points)'); ax.legend(fontsize=8.3,loc='upper left',bbox_to_anchor=(1.01,1)); ax.set_title('What drove the rise and the fall: method mix vs within-method change')
    save(fig,'f_decomp')

# ---------------------------------------------------------------- forest: time-pressure hypotheses
def forest(ax, items, title, xlim=(0.05,12), colors=(BLUE,ORANGE), labels=('≤ Apr 2022 (snapshot)','after Apr 2022 (newer)')):
    yl=[]; y=0; ticks=[]
    for name,vals in items:
        for k,v in enumerate(vals):
            if v is None: continue
            o,l,h=v; yy=y+(k-0.5*(len(vals)-1))*0.32
            ax.plot([max(l,xlim[0]),min(h,xlim[1])],[yy,yy],color=colors[k],lw=2,solid_capstyle='round')
            ax.plot(o,yy,'o',color=colors[k],ms=7,mec='white',mew=1.3)
            sig=(l>1 or h<1)
            ax.text(xlim[1]*1.05,yy,f'{o:.2f} ({l:.2f}–{h:.2f})'+(' *' if sig else ''),va='center',fontsize=7.6,color=INK if sig else MUTED)
        ticks.append(y); yl.append(name); y+=1
    ax.set_yticks(ticks); ax.set_yticklabels(yl,fontsize=8.6); ax.invert_yaxis(); ax.set_xscale('log'); ax.set_xlim(*xlim)
    ax.axvline(1,color=MUTED,lw=1,ls='--'); xgrid(ax); ax.set_title(title)
    tk=[t for t in [0.03,0.1,0.3,1,3,10,30] if xlim[0]<=t<=xlim[1]]; ax.set_xticks(tk); ax.set_xticklabels([f'{t:g}' for t in tk])
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],color=c,marker='o',lw=2,label=lb) for c,lb in zip(colors,labels)],loc='upper center',fontsize=8.3,bbox_to_anchor=(0.5,-0.13),ncol=2)

def f_forest_pressure():
    items=[('Antibody is only a tool (auxiliary)\nvs antibody is the subject',[(0.30,0.17,0.54),(0.79,0.36,1.71)]),
           ('Published in Nature / Science / Cell\nvs other journals',[(0.94,0.55,1.60),(0.53,0.23,1.23)]),
           ('Published in a CNS sister journal\nvs other journals',[(0.92,0.59,1.44),(0.57,0.35,0.92)]),
           ('More entries per paper\n(per doubling)',[(1.04,0.86,1.27),(0.85,0.69,1.04)]),
           ('1–2 competing labs on the same target\nwithin ±6 months (vs none)',[(1.06,0.61,1.83),(0.49,0.28,0.83)]),
           ('3–9 competing labs (vs none)',[(1.22,0.68,2.19),(0.64,0.30,1.35)]),
           ('Novel target\n(first year in the PDB)',[(1.60,1.03,2.47),(0.91,0.53,1.54)]),
           ('Released within 30 days\n(vs 91–180 days on hold)',[(1.26,0.58,2.74),(0.34,0.08,1.45)]),
           ('Released in 31–90 days\n(vs 91–180 days)',[(1.05,0.62,1.78),(0.17,0.07,0.44)]),
           ('SARS-CoV-2 antigen',[(0.67,0.20,2.22),(0.25,0.09,0.75)])]
    fig,ax=plt.subplots(figsize=(8.8,6.6))
    forest(ax,items,'Adjusted odds ratios for an antibody V-domain register error (SAbR vs ANARCI)',xlim=(0.05,6))
    ax.set_xlabel('odds ratio (log scale)    ← fewer errors      more errors →')
    save(fig,'f_forest_pressure')

def f_forest_mech():
    items=[('Cryo-EM vs X-ray at the same resolution',[(5.80,3.54,9.51),(3.29,1.79,6.05)]),
           ('Local map quality of the V domain\n(per SD better)',[(1.10,0.87,1.39),(1.04,0.85,1.26)]),
           ('A different antibody chain in the same\nentry is also flagged',[(2.70,1.55,4.70),None]),
           ('Atypical CDR2 length for its chain type\n(CDR1/CDR2-boundary runs)',[(9.2,3.7,22.9),None]),
           ('Entry records an AlphaFold starting model\n(vs none recorded)',[None,(0.91,0.40,2.07)]),
           ('EM model built ab initio\n(vs fitting not recorded)',[None,(0.22,0.08,0.58)]),
           ('Lab\'s earlier excess error rate\n(O/E > 1.5, snapshot → newer)',[None,(1.27,0.45,3.54)])]
    fig,ax=plt.subplots(figsize=(8.8,5.0))
    forest(ax,items,'What does, and does not, predict register errors (adjusted odds ratios)',xlim=(0.05,30),labels=('≤ Apr 2022 / pooled','after Apr 2022'))
    ax.set_xlabel('odds ratio (log scale)')
    save(fig,'f_forest_mech')

# ---------------------------------------------------------------- template effect
def f_template():
    fig,axs=plt.subplots(1,2,figsize=(10,3.7),gridspec_kw={'wspace':0.32})
    ax=axs[0]
    labs=['none\n(n=12,349)','yes, clean\n(n=3,651)','yes, flagged\n(n=140)','yes, flagged;\nno HIV bnAbs\n(n=39)']
    vals=[4.1,0.8,93,74]; ns=['n=12,349','n=3,651','n=140','n=39']; cols=[NEUTRAL,BLUE,ORANGE,ORANGE]
    x=np.arange(4); bars(ax,x,vals,cols)
    for xi,v,nn in zip(x,vals,ns): ax.text(xi,v+2,f'{v:g}%',ha='center',fontsize=9,color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=8); ax.set_ylim(0,105); ygrid(ax); ax.set_ylabel('% of 3–5 Å V domains flagged'); ax.set_title('A. Earlier <3 Å structure of the same V region?')
    ax=axs[1]
    labs=['clean structure\n(n=1,140)','flagged structure\n(n=58)']
    vals=[0.3,84.5]; x=np.arange(2); bars(ax,x,vals,[BLUE,ORANGE],width=0.55)
    for xi,v in zip(x,vals): ax.text(xi,v+2,f'{v:g}%',ha='center',fontsize=9,color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=8.5); ax.set_ylim(0,105); ygrid(ax)
    ax.set_title('B. Recorded starting model is a…')
    save(fig,'f_template')

# ---------------------------------------------------------------- 1B2 timeline
def f_timeline_1b2():
    ent=[(2018.1,'6C9U','2.09 Å X-ray','Khosla','f',False,'below'),(2018.6,'6MLK (Fab 3A6)','2.45 Å X-ray','Khosla','o',False,'above'),(2020.7,'7KRA, 7KTX (Fab DH4)','3.2–4.3 Å EM','Frost','o',False,'below'),
         (2021.3,'7M7E–J (×6)','3.2–4.3 Å EM','Khosla','f',False,'above'),(2021.8,'7S6C, 7S6D','3.1 Å EM','Fromme','f',True,'below'),(2022.5,'8EE0','2.65 Å X-ray','Khosla','f',True,'below'),
         (2023.6,'8TPW, 8TPX','3.4–3.5 Å EM','Khosla','f',True,'above'),(2025.4,'9PAT, 9PAV, 9PC6','3.2–4.0 Å EM','Chen','f',False,'below'),(2026.3,'11SZ','2.94 Å EM','Cogan','f',False,'below')]
    labs=['Khosla','Frost','Fromme','Chen','Cogan']
    fig,ax=plt.subplots(figsize=(10,3.6))
    for t,e,res,lab,reg,cites,where in ent:
        y=labs.index(lab); c=ORANGE if reg=='f' else BLUE
        if cites: ax.plot(t,y,'o',ms=19,mfc='none',mec=INK,mew=1.2,zorder=2)
        ax.plot(t,y,'o',ms=11,color=c,mec='white',mew=1.5,zorder=3)
        if where=='below': ax.text(t,y+0.3,f'{e}\n{res}',ha='center',va='top',fontsize=7.2,color=INK2)
        else: ax.text(t,y-0.3,f'{e}\n{res}',ha='center',va='bottom',fontsize=7.2,color=INK2)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels([l+' lab' for l in labs]); ax.set_ylim(len(labs)-0.1,-1.0)
    ax.set_xlim(2017.5,2026.9); ax.set_xticks(range(2018,2027)); xgrid(ax)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color=ORANGE,ms=9,label='6C9U register (flagged)'),Line2D([],[],marker='o',ls='',color=BLUE,ms=9,label='other register (agrees with AlphaFold2)'),
                       Line2D([],[],marker='o',ls='',mfc='none',mec=INK,ms=13,label='entry names 6C9U as its starting model')],loc='upper center',fontsize=8.3,ncol=3,bbox_to_anchor=(0.5,-0.1))
    ax.set_title('One light-chain FR1 sequence, two registers: Fab 1B2 (orange) and two other Fabs that share the sequence (blue)')
    save(fig,'f_timeline_1b2')

# ---------------------------------------------------------------- AF2 artifact
def f_af2():
    fig,axs=plt.subplots(1,2,figsize=(10,3.5),gridspec_kw={'wspace':0.32})
    ax=axs[0]; x=np.arange(3); w=0.36
    pre=[10.0,8.3,16.7]; post=[40.5,61.5,37.8]; npre=[30,24,6]; npost=[227,26,201]
    bars(ax,x-w/2,pre,ORANGE,width=w,label='released before 30 Apr 2018\n(in AlphaFold2\'s training set)'); bars(ax,x+w/2,post,BLUE,width=w,label='released after')
    for xi,a,b,na,nb in zip(x,pre,post,npre,npost):
        ax.text(xi-w/2,a+1.5,f'{a:.0f}%\nn={na}',ha='center',fontsize=7.5,color=INK2); ax.text(xi+w/2,b+1.5,f'{b:.0f}%\nn={nb}',ha='center',fontsize=7.5,color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(['all','X-ray','cryo-EM']); ax.set_ylim(0,80); ygrid(ax); ax.legend(fontsize=7.8,loc='upper left')
    ax.set_ylabel('% of SAbR framework-shift errors\nalso flagged by conkit'); ax.set_title('A. conkit-validate sensitivity, by release date')
    ax=axs[1]; bars(ax,[0,1],[1.4,1.6],[ORANGE,BLUE],width=0.5)
    for xi,v,nn in zip([0,1],[1.4,1.6],[2090,1553]): ax.text(xi,v+0.05,f'{v}%  (n={nn:,})',ha='center',fontsize=8.5,color=INK2)
    ax.set_xticks([0,1]); ax.set_xticklabels(['before cutoff','after cutoff']); ax.set_ylim(0,2.2); ygrid(ax); ax.set_ylabel('% flagged by SAbR vs ANARCI')
    ax.set_title('B. SAbR\'s X-ray flag rate is flat across the cutoff')
    save(fig,'f_af2')

# ---------------------------------------------------------------- SARS-CoV-2
def f_sars():
    fig,axs=plt.subplots(1,2,figsize=(10,3.5),gridspec_kw={'wspace':0.3})
    w=0.36
    ax=axs[0]; x=np.arange(2)
    s=[6.40,2.87]; o=[1.39,1.54]
    bars(ax,x-w/2,s,RED,width=w,label='SARS-CoV-2 antigen'); bars(ax,x+w/2,o,NEUTRAL,width=w,label='other viral antigens')
    for xi,a,b in zip(x,s,o): ax.text(xi-w/2,a+0.2,f'{a:.1f}%',ha='center',fontsize=8.5,color=INK2); ax.text(xi+w/2,b+0.2,f'{b:.1f}%',ha='center',fontsize=8.5,color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(['2020','2021–22']); ax.set_ylim(0,14); ygrid(ax); ax.legend(fontsize=8.5); ax.set_ylabel('% of antibody V domains flagged')
    ax.set_title('A. conkit flags (X-ray + EM)')
    ax=axs[1]; x=np.arange(3)
    s=[11.2,3.6,1.7]; o=[11.0,7.9,3.8]; ns=[(376,218),(696,228),(1654,4175)]
    bars(ax,x-w/2,s,RED,width=w,label='SARS-CoV-2 antigen'); bars(ax,x+w/2,o,NEUTRAL,width=w,label='other antigens')
    for xi,a,b,(na,nb) in zip(x,s,o,ns): ax.text(xi-w/2,a+0.2,f'{a:.1f}%',ha='center',fontsize=8.5,color=INK2); ax.text(xi+w/2,b+0.2,f'{b:.1f}%',ha='center',fontsize=8.5,color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(['2020\n(vs other viral)','2021\n(vs other viral)','after Apr 2022\n(vs all others)'],fontsize=8.5); ax.set_ylim(0,14); ygrid(ax)
    ax.set_title('B. SAbR vs ANARCI (cryo-EM, antibody = subject)')
    save(fig,'f_sars')

# ---------------------------------------------------------------- role
def f_role():
    fig,axs=plt.subplots(1,2,figsize=(10,3.5),gridspec_kw={'wspace':0.3,'width_ratios':[1.6,1]})
    ax=axs[0]; w=0.36; x=np.arange(4)
    i=[8.1,2.1,3.2,1.6]; a=[3.3,0.9,3.8,1.8]; ni=[2497,2229,5829,1168]; na=[448,549,1347,170]
    bars(ax,x-w/2,i,BLUE,width=w,label='antibody is the subject'); bars(ax,x+w/2,a,ORANGE,width=w,label='antibody is a tool (auxiliary)')
    for xi,p,q in zip(x,i,a): ax.text(xi-w/2,p+0.15,f'{p:.1f}',ha='center',fontsize=8.5,color=INK2); ax.text(xi+w/2,q+0.15,f'{q:.1f}',ha='center',fontsize=8.5,color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(['EM\n≤ Apr 2022','X-ray\n≤ Apr 2022','EM\nnewer','X-ray\nnewer'],fontsize=8.5); ygrid(ax); ax.set_ylabel('% of V domains flagged'); ax.legend(fontsize=8.5)
    ax.set_title('A. Flag rate by the antibody\'s role in the study')
    ax=axs[1]; bars(ax,[0,1],[0,4.5],[ORANGE,BLUE],width=0.5)
    ax.text(0,0.25,'0 / 67',ha='center',fontsize=9,color=INK2); ax.text(1,4.75,'3 / 67',ha='center',fontsize=9,color=INK2)
    ax.set_xticks([0,1]); ax.set_xticklabels(['reused reagent\n(Nb35, scFv16, NabFab…)','other antibody chains\nin the same entries'],fontsize=8); ax.set_ylim(0,7); ygrid(ax)
    ax.set_title('B. Within the same 40 entries')
    save(fig,'f_role')

# ---------------------------------------------------------------- run location
def f_runloc():
    r=pd.read_csv(f'{R}/sabr_anarci_scan/tables/disagreement_runs.csv')
    fig,axs=plt.subplots(2,1,figsize=(10,4.6),sharex=True,gridspec_kw={'hspace':0.35})
    regions=[(1,26,'FR1'),(27,38,'CDR1'),(39,55,'FR2'),(56,65,'CDR2'),(66,104,'FR3'),(105,117,'CDR3'),(118,128,'FR4')]
    for ax,(ct,name) in zip(axs,[(['H'],'Heavy chains'),(['K','L'],'Light chains (κ + λ)')]):
        s=r[r.chain_type.isin(ct)]
        cnt=s.imgt_start.clip(1,128).value_counts().reindex(range(1,129),fill_value=0)
        for lo,hi,nm in regions:
            c=PALE if nm.startswith('FR') else '#fdf0d9'
            ax.axvspan(lo-0.5,hi+0.5,color=c,lw=0,zorder=0); ax.text((lo+hi)/2,1.02,nm,transform=ax.get_xaxis_transform(),ha='center',fontsize=7.8,color=INK2)
        ax.bar(cnt.index,cnt.values,width=0.85,color=BLUE,zorder=2)
        ax.set_ylabel('runs starting here'); ax.text(0.995,0.85,f'{name}  (n={len(s):,} runs)',transform=ax.transAxes,ha='right',fontsize=9,fontweight='bold',color=INK)
        ygrid(ax)
    axs[1].set_xlabel('IMGT position where the disagreement run starts'); axs[1].set_xlim(0,129)
    save(fig,'f_runloc')

# ---------------------------------------------------------------- gaps / N-term
def f_gaps():
    fig,axs=plt.subplots(1,3,figsize=(10.4,3.3),gridspec_kw={'wspace':0.45})
    ax=axs[0]; bars(ax,[0,1],[11.9,0.9],[ORANGE,NEUTRAL],width=0.55)
    for xi,v in zip([0,1],[11.9,0.9]): ax.text(xi,v+0.3,f'{v}%',ha='center',fontsize=9,color=INK2)
    ax.set_xticks([0,1]); ax.set_xticklabels(['disagreement\nruns','matched windows,\nclean domains'],fontsize=8); ygrid(ax); ax.set_ylabel('% within 3 residues\nof an unbuilt gap'); ax.set_title('A. Next to unbuilt residues')
    ax=axs[1]; bars(ax,[0,1],[29.9,2.8],[ORANGE,NEUTRAL],width=0.55)
    for xi,v in zip([0,1],[29.9,2.8]): ax.text(xi,v+0.6,f'{v}%',ha='center',fontsize=9,color=INK2)
    ax.set_xticks([0,1]); ax.set_xticklabels(['disagreement\nruns','random windows,\nclean domains'],fontsize=8); ygrid(ax); ax.set_ylabel('% starting at the\nN-terminus'); ax.set_title('B. At the chain start')
    ax=axs[2]; bars(ax,[0,1],[18.1,3.4],[ORANGE,BLUE],width=0.55)
    for xi,v,nn in zip([0,1],[18.1,3.4],['n=635','n=15,230']): ax.text(xi,v+0.4,f'{v}%\n{nn}',ha='center',fontsize=8.5,color=INK2)
    ax.set_xticks([0,1]); ax.set_xticklabels(['first residues\nnot built','complete\nN-terminus'],fontsize=8); ax.set_ylim(0,23); ygrid(ax); ax.set_ylabel('% of V domains flagged'); ax.set_title('C. Domains by N-terminus')
    save(fig,'f_gaps')

# ---------------------------------------------------------------- W118 burial
def f_w118():
    w=pd.read_csv(f'{R}/retest/unexplained/tables/w118_burial.csv')
    groups=[('high-res template (<3 Å)','High-res templates (<3 Å)',NEUTRAL),('clean V domain | X-ray','Clean, X-ray 3–5 Å',NEUTRAL),('clean V domain | EM','Clean, cryo-EM 3–5 Å',NEUTRAL),
            ('FR4: SAbR skips IMGT 120 after W/F118-G119 | EM','FR4 bulge runs, EM',YELLOW),
            ('CDR3→FR4: conserved W/F118 displaced | X-ray','CDR3–FR4 junction runs, X-ray',ORANGE),('CDR3→FR4: conserved W/F118 displaced | EM','CDR3–FR4 junction runs, EM',ORANGE)]
    fig,ax=plt.subplots(figsize=(9,3.9)); rng=np.random.default_rng(1)
    for i,(g,nm,c) in enumerate(groups):
        v=w[w.label==g].ca_within_10A.dropna().values
        ax.scatter(v+rng.uniform(-0.25,0.25,len(v)),i+rng.uniform(-0.22,0.22,len(v)),s=10,color=c if c!=NEUTRAL else '#8f8e88',alpha=0.55,lw=0)
        ax.plot([np.median(v)]*2,[i-0.33,i+0.33],color=INK,lw=2)
        ax.text(31.2,i,f'{(v<14).mean()*100:.0f}% exposed  (n={len(v)})',va='center',fontsize=8.3,color=INK2)
    ax.axvspan(0,14,color='#fbe0d4',alpha=0.55,lw=0,zorder=0); ax.text(7,-0.75,'exposed (<14 neighbours)',ha='center',fontsize=8,color=ORANGE)
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([g[1] for g in groups]); ax.set_xlim(0,31); ax.set_ylim(len(groups)-0.5,-1.0)
    ax.set_xlabel('Cα atoms within 10 Å of the side chain of Trp/Phe118 (ANARCI\'s assignment)'); xgrid(ax)
    ax.set_title('Burial of the conserved FR4 anchor Trp/Phe118: a structure-only test, no map or numbering tool')
    save(fig,'f_w118')

# ---------------------------------------------------------------- CDR length
def f_cdrlen():
    fig,axs=plt.subplots(1,2,figsize=(10,3.4),gridspec_kw={'wspace':0.3})
    ax=axs[0]
    k=[(5,1.1,268),(6,2.7,3018),(7,0.0,600),(10,0.3,303),(11,5.4,445),(12,4.9,285)]
    l=[(4,0.0,48),(6,6.2,520),(8,12.3,480),(9,1.9,986)]
    xs=np.arange(len(k)); bars(ax,xs,[a[1] for a in k],BLUE,width=0.6)
    xs2=np.arange(len(l))+len(k)+0.8; bars(ax,xs2,[a[1] for a in l],VIOLET,width=0.6)
    for x,(L,p,n) in zip(list(xs)+list(xs2),k+l): ax.text(x,p+0.3,f'{p:.1f}%',ha='center',fontsize=7.8,color=INK2)
    ax.set_xticks(list(xs)+list(xs2)); ax.set_xticklabels([str(a[0]) for a in k+l]); ax.set_xlabel('CDR1 length (κ: blue, λ: violet)'); ygrid(ax); ax.set_ylim(0,14.5)
    ax.set_ylabel('% of V domains flagged'); ax.set_title('A. Light-chain CDR1 length')
    ax=axs[1]
    bars(ax,[0,1],[2.6,0.4],[ORANGE,NEUTRAL],width=0.55)
    for x,v in zip([0,1],[2.6,0.4]): ax.text(x,v+0.07,f'{v}%',ha='center',fontsize=9,color=INK2)
    ax.set_xticks([0,1]); ax.set_xticklabels(['rare CDR2 length\nfor the chain type','common CDR2\nlength'],fontsize=8.5); ygrid(ax); ax.set_ylim(0,3.3)
    ax.set_ylabel('% of heavy chains with a run\ntouching CDR2'); ax.set_title('B. Heavy-chain CDR2 (adjusted OR 9.2, 3.7–22.9)')
    save(fig,'f_cdrlen')

# ---------------------------------------------------------------- origins
def f_origins():
    cats=['Chain end / next to an\nunbuilt loop','CDR3–FR4 junction','FR4 bulge (IMGT 120)','CDR1/CDR2 boundary','Copied from an earlier\nstructure','HIV / FR3-insertion\n(ambiguous)','Still unexplained']
    snap=[36,35,5,10,9,22,6]; new=[70,34,14,20,11,11,20]
    cols=[BLUE,ORANGE,YELLOW,AQUA,VIOLET,'#b9b8b1',RED]
    fig,ax=plt.subplots(figsize=(10,2.9))
    for yi,(lab,v) in enumerate([('≤ Apr 2022\n(123 events)',snap),('after Apr 2022\n(180 events)',new)]):
        tot=sum(v); left=0
        for c,(val,col) in enumerate(zip(v,cols)):
            w=val/tot*100; ax.barh(yi,w,left=left,color=col,edgecolor='white',linewidth=2,height=0.62)
            if w>=5.5: ax.text(left+w/2,yi,f'{w:.0f}%',ha='center',va='center',fontsize=8.5,color='white' if col not in (YELLOW,'#b9b8b1',AQUA) else INK)
            left+=w
    ax.set_yticks([0,1]); ax.set_yticklabels(['≤ Apr 2022\n(123 events)','after Apr 2022\n(180 events)']); ax.invert_yaxis(); ax.set_xlim(0,100); ax.set_xlabel('% of independent error events')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c,label=l.replace('\n',' ')) for c,l in zip(cols,cats)],ncol=4,fontsize=7.8,loc='upper center',bbox_to_anchor=(0.5,-0.3))
    ax.set_title('Where each independent register error comes from (one event = paper × antibody × position × shift)')
    save(fig,'f_origins')

# ---------------------------------------------------------------- persistence
def f_persist():
    fig,axs=plt.subplots(1,3,figsize=(10.4,3.3),gridspec_kw={'wspace':0.5})
    ax=axs[0]; bars(ax,[0,1],[9,91],[BLUE,NEUTRAL],width=0.55)
    for x,v in zip([0,1],[9,91]): ax.text(x,v+2,f'{v}%',ha='center',fontsize=9,color=INK2)
    ax.set_xticks([0,1]); ax.set_xticklabels(['coordinates\never revised','never\nrevised'],fontsize=8.5); ax.set_ylim(0,105); ygrid(ax); ax.set_title('A. 87 conkit-flagged\nantibody entries')
    ax=axs[1]; bars(ax,[0,1],[31,62],[BLUE,ORANGE],width=0.55)
    for x,v,n in zip([0,1],[31,62],['n=16','n=149']): ax.text(x,v+2,f'{v}%\n{n}',ha='center',fontsize=8.5,color=INK2)
    ax.set_xticks([0,1]); ax.set_xticklabels(['revised','unrevised'],fontsize=8.5); ax.set_ylim(0,80); ygrid(ax); ax.set_ylabel('% of flagged segments\nstill wrong today'); ax.set_title('B. Is the flagged segment\nstill wrong today?')
    ax=axs[2]; bars(ax,[0,1],[86,14],[ORANGE,NEUTRAL],width=0.55)
    for x,v in zip([0,1],[86,14]): ax.text(x,v+2,f'{v}%',ha='center',fontsize=9,color=INK2)
    ax.set_xticks([0,1]); ax.set_xticklabels(['all copies\nflagged','only some\ncopies'],fontsize=8.5); ax.set_ylim(0,105); ygrid(ax); ax.set_title('C. 170 groups of identical\ncopies in one entry')
    save(fig,'f_persist')

if __name__=='__main__':
    which=sys.argv[1:] or [k for k in globals() if k.startswith('f_')]
    for k in which: globals()[k]()
