"""Render Figure 26 (Trp118 displacement, 7V2A chain G vs 7YV1 heavy chain).

Uses the helpers in render.py (3Dmol.js page + Playwright screenshot via shot.js).
Per-chain PDB files and numbering.json are produced by rep_struct_prep.py.
"""
import numpy as np
from render import superpose, pdbtext, atoms, imgt_map, labels, apply, vresi, render

BLUE = '#1f5fb0'; ORANGE = '#d9531e'
fr = lambda l: (l[0] <= 26 or 39 <= l[0] <= 55 or 66 <= l[0] <= 104) and not l[1]
tG, rms, n = superpose('7v2a_G', '7yv1_H', use=fr)
tR = pdbtext('7yv1_H'); aR = atoms(tR); mR = imgt_map('7yv1_H'); mG = imgt_map('7v2a_G'); aG = atoms(tG)
def sc(a, k): return np.mean([v for nm, v in a[k].items() if nm not in ('N', 'CA', 'C', 'O', 'CB')], axis=0)
vR = [aR[k]['CA'] for k, v in labels('7yv1_H').items() if v[1] is not None and 'CA' in aR[k]]; c = np.mean(vR, axis=0)
pR = sc(aR, mR[(118, '')]); pG = sc(aG, mG[(118, '')])
x = (pG - pR); x /= np.linalg.norm(x); y = (pR - c); y -= y.dot(x) * x; y /= np.linalg.norm(y); z = np.cross(x, y); R = np.vstack([x, y, z])
tR2 = apply(tR, R, c); tG2 = apply(tG, R, c); aR2 = atoms(tR2); aG2 = atoms(tG2)
vresR = vresi('7yv1_H')
cdr3fr4R = [mR[l][0] for l in mR if l[0] >= 105]; cdr3fr4G = [mG[l][0] for l in mG if l[0] >= 105]
wR = mR[(118, '')][0]; wG = mG[(118, '')][0]
jsR = f"v.setStyle({{model:MID}},{{}});v.setStyle({{model:MID,resi:{vresR}}},{{cartoon:{{color:'#c3d3ea',opacity:0.8}}}});v.setStyle({{model:MID,resi:{cdr3fr4R}}},{{cartoon:{{color:'{BLUE}'}}}});v.addStyle({{model:MID,resi:[{wR}]}},{{stick:{{color:'{BLUE}',radius:0.38}}}});"
jsG = f"v.setStyle({{model:MID}},{{}});v.setStyle({{model:MID,resi:{cdr3fr4G}}},{{cartoon:{{color:'{ORANGE}'}}}});v.addStyle({{model:MID,resi:[{wG}]}},{{stick:{{color:'{ORANGE}',radius:0.38}}}});"
pR2 = sc(aR2, mR[(118, '')]); pG2 = sc(aG2, mG[(118, '')])
def lab(t, pos, col, al): return f"v.addLabel('{t}',{{position:{{x:{pos[0]:.1f},y:{pos[1]:.1f},z:{pos[2]:.1f}}},alignment:'{al}',fontSize:15,fontColor:'{col}',backgroundColor:'white',backgroundOpacity:0.92,borderThickness:1,borderColor:'{col}',inFront:true,font:'Helvetica'}});"
post = lab('Trp118, 7YV1 (1.45 A X-ray): in the core', pR2 + np.array([0, -3, 0]), BLUE, 'topCenter') + lab('Trp118, 7V2A (3.4 A EM): 11 A out', pG2 + np.array([3, 0, 0]), ORANGE, 'centerLeft')
render('s_cdr3fr4', [dict(pdb=tR2, js=jsR), dict(pdb=tG2, js=jsG)], w=860, h=560, view=f"v.zoomTo({{model:0,resi:{vresR}}}); v.zoom(1.1); v.translate(-90,0);", post=post)
