"""V04: is ANARCI's FR3 gap position (81 vs 84) set by single residues rather than by structure?
Point-mutate the DE-loop of the 'clean' rabbit control (9p4c, ANARCI gap 81) and of flagged chains (30ad, 9bop; gap 84)
and record where ANARCI (scan setting: human+mouse HMMs) puts the gap. SAbR places a one-residue DE deletion at 81 by
rule (sabr/corrections.py de_loop_positions), so any sequence-driven move of the ANARCI gap to 84 creates a 3-residue
'run' at 81-83 without any change in structure."""
from anarci import run_anarci
SEQ = {
 '9p4c (control, gap 81)': 'QSLEESGGGLVTPGGILTLTCTASGFDISRHYMTWVRQAPGEGLEWIGSIYGGSTYYASWAKGRFTISKTSSTTVDLKMTSLTTEDTATYFCVRGVGGGLWGPGTLVTVSS',
 '30ad (flagged, gap 84)': 'QEQLEESGGDLVKPEGSLTLTCTASGFSVSSTDWIWWVRQAPGKGLEWIGCIVSGSENTYYASWAEGRFIISKASSTTVTLQMTSLTAADTATYFCARDGSDSGDYRLDLWGQGTLVTVSS',
 '9bop (flagged, gap 84)': 'QSVEESGGRLVTPGTPLTLTCIVSGFSLSNYYMSWVRQVPGKGLEWIGVIGWSGTSSYASWAKGRFTISKTASTTVDLKITSPTTEDTATYFCARVLYIGGGFNYYDAFDPWGPGTLVTVS',
}
MUT = {'9p4c (control, gap 81)': [('ISKTSSTTV', 'ISKASSTTV'), ('ISKTSSTTV', 'ISRASSTTV'), ('ISKTSSTTV', 'ISKTASTTV')],
       '30ad (flagged, gap 84)': [('ISKASSTTV', 'ISKTSSTTV')],
       '9bop (flagged, gap 84)': [('ISKTASTTV', 'ISKTSSTTV')]}
def de(seq, species=('human', 'mouse')):
    an = run_anarci([('q', seq)], scheme='imgt', allowed_species=list(species) if species else None)
    num = [(p, i.strip(), a) for (p, i), a in an[1][0][0][0] if a != '-']
    pos = {p for p, i, a in num if i == ''}
    gap = sorted(k for k in range(74, 101) if k not in pos)
    return ' '.join(f'{a}{p}' for p, i, a in num if 78 <= p <= 86), gap
rows = []
for name, s in SEQ.items():
    w, g = de(s); wa, ga = de(s, None)
    print(f'{name:26s} wild type       : {w:40s} gap {g}   | all-species HMMs gap {ga}')
    for a, b in MUT[name]:
        assert a in s
        m = s.replace(a, b)
        w, g = de(m); wa, ga = de(m, None)
        print(f'{"":26s} {a}->{b}: {w:40s} gap {g}   | all-species HMMs gap {ga}')
